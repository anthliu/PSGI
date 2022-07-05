"""Expand-filter version of Inductive Logic Programming (ILP) module implementation."""

from typing import Optional, Dict, List

import os
import numpy as np
import math
import time
from acme import specs
from psgi.utils import log_utils

from psgi.graph.ilp import ILP
from psgi.utils import graph_utils
from psgi.utils import log_utils
from psgi.utils.graph_utils import SubtaskGraph, dotdict, GraphVisualizer
try:
  from psgi.graph.expand import expand_and_filter
except ImportError:
  expand_and_filter = None   # module needs compilation, but throw lazily


class ExpandILP(ILP):
  """Expand-filter version of ILP."""
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      num_adapt_steps: int,
      branch_neccessary_first: bool = True,
      bonus_mode: int = 0,  # TODO: Change this to string
      visualize: bool = False,
      directory: str = 'visualize',
      environment_id: Optional[str] = None,
      verbose: bool = False,
      tr_epi: int = 20):  # TODO: replace this with iteration
    super().__init__(
        environment_spec=environment_spec,
        num_adapt_steps=num_adapt_steps,
        branch_neccessary_first=branch_neccessary_first,
        bonus_mode=bonus_mode,
        visualize=visualize,
        directory=directory,
        environment_id=environment_id,
        verbose=verbose,
        tr_epi=tr_epi,
    )
    self._preprocess_mode = 'best_fit'
    self._strict_inference = True # If true, expect perfect classification of eligibility
    self._use_prior = False

    # duplication filtering for prior
    self._prior_unique_indices_by_batch_ind = []

    if not expand_and_filter:
      # Check the existence of the extension lazily to avoid import errors
      # in a case where MTSGI is actually not being used.
      raise ImportError("psgi.graph.expand.expand_and_filter has not been built.")

  def update_prior(self, batch_prior_data: List[Dict]):
    assert self._step_count == 0, 'Currently, we expect this to be executed right after reset()'
    self._use_prior = True

    # Re-arrange subtasks in prior data such that they are aligned with current subtasks
    self._prior_unique_indices_by_batch_ind = []
    prior_completion, prior_eligibility = [], []
    reward_sum, reward_count = [], []
    self._prior_index_to_pool = []
    for i, prior_data in enumerate(batch_prior_data):
      redundant_pool_id = [pid for pid in prior_data['index_to_pool'] if pid not in self._index_to_pool[i]] # In prior, not in current
      missing_pool_id = [pid for pid in self._index_to_pool[i] if pid not in prior_data['index_to_pool']] # In current, not in prior
      self._prior_index_to_pool.append(prior_data['index_to_pool'])

      ### 1. compute mapping prior -> current
      curr_idx_to_pool = self._index_to_pool[i]
      prior_pool_to_idx = prior_data['pool_to_index']
      prior_to_current_mapping = np.array([prior_pool_to_idx[pool_id] for pool_id in curr_idx_to_pool])
      ### 2. Map prior -> current
      prior_comp = graph_utils.batched_mapping_expand(prior_data['completion'].astype(np.int8), prior_to_current_mapping, default_val=-1)
      prior_elig = graph_utils.batched_mapping_expand(prior_data['eligibility'].astype(np.int8), prior_to_current_mapping, default_val=-1)
      prior_rew = graph_utils.batched_mapping_expand(prior_data['reward_sum'], prior_to_current_mapping, default_val=0)
      prior_rew_count = graph_utils.batched_mapping_expand(prior_data['reward_count'], prior_to_current_mapping, default_val=0)
      assert len(missing_pool_id) == 0 or np.any(prior_comp < 0), 'prior_completion is wrongly mapped'

      if len(missing_pool_id) >= 5:
        print("Warning! Too many subtasks are missing in prior. # missing subtasks = ", len(missing_pool_id))

      # After this point, only [common, missing] remains; redundant subtasks are gone!
      # Stack
      prior_completion.append(prior_comp) # dtype=np.int8
      prior_eligibility.append(prior_elig)# dtype=np.int8
      reward_sum.append(prior_rew)
      reward_count.append(prior_rew_count)
    self.prior_completion = prior_completion # dtype=
    self.prior_eligibility = prior_eligibility
    self.reward_sum = np.stack(reward_sum)
    self.reward_count = np.stack(reward_count)

  def _randomly_fill_dc(self, completion, max_num_dc=4):
    prior_dc_all = np.all(completion == -1, axis=0)
    dc_indices = np.flatnonzero(prior_dc_all).tolist()
    num_dc = len(dc_indices)
    if num_dc > max_num_dc:
      fillout_indices = np.random.permutation(dc_indices)[:num_dc - max_num_dc]
      completion[:, fillout_indices] = np.random.randint(low=0, high=2, size=(completion.shape[0], num_dc - max_num_dc))
    return completion

  def infer_task(self) -> List[SubtaskGraph]:
    if not self._use_prior:
      return super().infer_task()
    graphs = []
    for i in range(self._batch_size):
      if self._duplication_filter_flag: # Filter-out duplicated result
        curr_idx = self._unique_indices_by_batch_ind[i]
      else:
        curr_idx = list(range(self._step_count))

      #0. initialize graph
      graph = SubtaskGraph(
          index_to_pool=self._index_to_pool[i],
          pool_to_index=self._pool_to_index[i])

      #1. Infer reward (prior reward is already merged in "update_prior()")
      graph.subtask_reward = self._infer_reward(self.reward_count[i], self.reward_sum[i]) #mean-reward-tracking

      #2. Infer precondition
      kmap_set, tind_by_layer = self._infer_precondition_with_prior(
          completion=self._completion_buffer[curr_idx, i, :],
          eligibility=self._eligibility_buffer[curr_idx, i, :],
          prior_completion=self.prior_completion[i],
          prior_eligibility=self.prior_eligibility[i],
      )

      #3. Fill-out graph edges from precondition kmap
      matrices = self._kmap_to_mat(kmap_set, tind_by_layer)
      graph.fill_edges(**matrices)
      graphs.append(graph)
    return graphs

  def _infer_precondition_with_prior(
      self,
      completion: np.ndarray,   # (T, num_subtasks), dtype=np.bool
      eligibility: np.ndarray,  # (T, num_subtasks), dtype=np.bool
      prior_completion: np.ndarray,   # (T, num_subtasks), dtype=np.int8
      prior_eligibility: np.ndarray): # (T, num_subtasks), dtype=np.int8
    # 0. assertions
    assert completion.shape == eligibility.shape and completion.ndim == 2, "shape error"
    assert prior_completion.shape == prior_eligibility.shape and prior_completion.ndim == 2, "shape error"
    assert completion.dtype == eligibility.dtype and completion.dtype == np.bool, "completion, eligibility should be np.bool type"
    assert prior_completion.dtype == prior_eligibility.dtype and prior_completion.dtype == np.int8, "priors should be np.int8 type"

    # 1. Find first layer: either always elig or never elig, while ignoring DC
    always_elig = np.all(eligibility, axis=0)
    never_elig = np.all(np.logical_not(eligibility), axis=0)
    prior_always_elig = np.all(prior_eligibility == 1, axis=0)
    prior_never_elig = np.all(prior_eligibility == 0, axis=0)
    prior_dc = np.all(prior_eligibility == -1, axis=0)
    first_layer_mask = (always_elig & (prior_always_elig | prior_dc)) | (never_elig & (prior_never_elig | prior_dc))

    # Layer-wise ILP with prior
    curr_layer_ind_list = np.flatnonzero(first_layer_mask).tolist()
    tind_by_layer = [curr_layer_ind_list]
    cand_ind_list = curr_layer_ind_list.copy()
    subtask_layer = np.full((self.ntasks), fill_value=-1, dtype=np.int16)
    subtask_layer[first_layer_mask] = 0 # assign first layer
    kmap_set = [None] * self.ntasks
    for layer_ind in range(1, self.ntasks):
      curr_layer_ind_list = []
      curr_comp = completion[:, cand_ind_list]
      prior_comp = prior_completion[:, cand_ind_list]
      _, curr_unique_idx = np.unique(curr_comp, return_index=True, axis=0)

      for ind in range(self.ntasks):
        if subtask_layer[ind] >= 0: # already assigned
          continue
        curr_elig = eligibility[:, ind]

        # 1. check if curr is valid
        if self._check_validity(curr_comp, curr_elig):
          # Valid --> safely use unique data only
          curr_comp_uniq = curr_comp[curr_unique_idx, :]
          curr_elig_uniq = eligibility[curr_unique_idx, ind]
        else:
          continue

        # 2. Expand & filter the prior
        if prior_dc[ind]:
          # If elig[ind] is all D.C -> cannot use prior. Only use curr
          final_comp = curr_comp
          final_elig = curr_elig
        else:
          prior_elig = prior_eligibility[:, ind]
          # 0. Remove duplicated data in prior to make expand-filter faster.
          prior_comp_uniq, prior_elig_uniq = self._remove_duplicate(prior_comp, prior_elig, skip_elig=False)

          prior_comp_uniq = self._randomly_fill_dc(prior_comp_uniq, max_num_dc=6)

          #print('#input=', prior_comp_uniq.shape[0])
          # 1. Expand & filter
          expanded_comp, expanded_elig = expand_and_filter.expand_and_filter(
            prior_comp_uniq,
            np.expand_dims(prior_elig_uniq, axis=-1).astype(np.int8),
            curr_comp_uniq.astype(np.int8),
            np.expand_dims(curr_elig_uniq, axis=-1).astype(np.int8)
          )
          #print('#output=', expanded_comp.shape[0])
          assert np.all(expanded_comp >= 0) and np.all(expanded_elig >= 0), "Error: There still exists DC after expansion"

          # 2. Remove incompatible ones
          if expanded_comp.shape[0] > 5000:
            rand_idx = np.random.permutation(expanded_comp.shape[0])[:5000]
            expanded_comp = expanded_comp[rand_idx]
            expanded_elig = expanded_elig[rand_idx]
          codes = graph_utils.batch_bin_encode(expanded_comp.astype(np.bool))
          _, refine_indices = np.unique(codes, return_index=True, axis=0)
          refined_prior_comp = expanded_comp[refine_indices]
          refined_prior_elig = expanded_elig[refine_indices, 0]

          # 3. Concat
          final_comp = np.concatenate([curr_comp_uniq, refined_prior_comp], axis=0).astype(np.bool) # T'' x ntasks (T'' = T + T')
          final_elig = np.concatenate([curr_elig_uniq, refined_prior_elig], axis=0).astype(np.bool) # T'' x ntasks
          assert self._check_validity(final_comp, final_elig), "Error: final_comp, final_elig is invalid! Either expand-filter is wrong or incompatible data removal is wrong"

        assert final_comp.dtype == np.bool and final_elig.dtype == np.bool, "Error: dtype error"

        # CART (97~99%)
        mask = np.ones(final_comp.shape[-1], dtype=np.bool)
        root = self.cart_train(mask, final_comp, final_elig) # 97~99%
        kmap_tensor_org = self.decode_cart(root, final_comp.shape[-1]) # 0.5%
        kmap_tensor = self.simplify_Kmap(kmap_tensor_org) # 0.6%

        assert kmap_tensor.ndim == 2, "kmap_tensor should be 2 dimension"
        kmap_set[ind] = kmap_tensor

        curr_layer_ind_list.append(ind)
        subtask_layer[ind] = layer_ind

      assert len(curr_layer_ind_list) > 0, f"Error: layer-wise ILP failed to find precondition for subtasks {np.nonzero(subtask_layer==-1)}"
      cand_ind_list.extend(curr_layer_ind_list)
      tind_by_layer.append(curr_layer_ind_list)
      if len(cand_ind_list) == self.ntasks:
        break
    return kmap_set, tind_by_layer

  def _remove_duplicate(self, comp, elig, skip_elig=False):
    # dtype can be anything. Usually np.bool or np.int8
    assert comp.ndim == 2 and elig.ndim == 1 and comp.shape[0] == elig.shape[0], "shape error"
    if skip_elig:
      data = comp
    else:
      data = np.concatenate((comp, np.expand_dims(elig, axis=-1)), axis=1)
    _, uniq_indices = np.unique(data, return_index=True, axis=0)
    return comp[uniq_indices], elig[uniq_indices]
