"""Bayesian version of Inductive Logic Programming (ILP) module implementation."""

from typing import Optional, Dict, List

import os
import numpy as np
import math

from acme import specs

from psgi.graph.ilp import ILP
from psgi.utils import graph_utils
from psgi.utils import log_utils
from psgi.utils.graph_utils import SubtaskGraph, dotdict, GraphVisualizer

class BayesILP(ILP):
  """ Bayesian version of ILP
  """
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      num_adapt_steps: int,
      bonus_mode: int = 0,  # TODO: Change this to string
      visualize: bool = False,
      directory: str = 'visualize',
      environment_id: Optional[str] = None,
      verbose: bool = False,
      tr_epi: int = 20):  # TODO: replace this with iteration
    super().__init__(
        environment_spec=environment_spec,
        num_adapt_steps=num_adapt_steps,
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
  
  def update_prior(self, batch_prior_data: List[Dict]):
    assert self._step_count == 0, 'Currently, we expect this to be executed right after reset()'
    self._use_prior = True
    #self.prior_step_count = prior_data[0]['step_count'] # not used. possibly remove
    
    # Re-arrange subtasks in prior data such that they are aligned with current subtasks
    self.prior_completion, self.prior_eligibility = [], []
    self.reward_sum, self.reward_count = [], []
    self.noisy_flag = []
    for i, prior_data in enumerate(batch_prior_data):
      # self._index_to_pool = [1, 8, 11, 5, 4, 7] # Testing
      # prior_pool_to_index = [3, 2, -1, 0, -1, -1, -1, 1, 6, 5, -1, 4] # Testing
      num_redundant = len([pid for pid in prior_data['index_to_pool'] if pid not in self._index_to_pool[i]]) # In prior, not in current
      num_missing = len([pid for pid in self._index_to_pool[i] if pid not in prior_data['index_to_pool']]) # In current, not in prior
      self.noisy_flag.append(num_missing > 0)
      """
      print(f'Redundant={num_redundant}, Missing={num_missing}')
      print('prior=', prior_data['index_to_pool'])
      print('current=', self._index_to_pool[0])"""
      ### 1. compute mapping prior -> current
      curr_idx_to_pool = self._index_to_pool[i]
      prior_pool_to_idx = prior_data['pool_to_index']
      prior_to_current_mapping = np.array([prior_pool_to_idx[pool_id] for pool_id in curr_idx_to_pool])
      ### 2. Map prior -> current
      prior_comp = graph_utils.batched_mapping_expand(prior_data['completion'].astype(np.int8), prior_to_current_mapping, default_val=-1)
      prior_elig = graph_utils.batched_mapping_expand(prior_data['eligibility'].astype(np.int8), prior_to_current_mapping, default_val=-1)
      prior_rew = graph_utils.batched_mapping_expand(prior_data['reward_sum'], prior_to_current_mapping, default_val=0)
      prior_rew_count = graph_utils.batched_mapping_expand(prior_data['reward_count'], prior_to_current_mapping, default_val=0)
      assert num_missing == 0 or np.any(prior_comp < 0), 'prior_completion is wrongly mapped'

      # Optionally fill-out missing values
      if self._preprocess_mode == 'random':
        raise NotImplementedError
        """
        # Assign random value
        missing_column_mask = prior_comp.sum(axis=0) < 0
        num_missing_columns = sum(missing_column_mask)
        if num_missing_columns > 0:
          prior_comp[:, missing_column_mask] = np.random.binomial(n=1, p=0.5, size=(prior_comp.shape[0], num_missing_columns))"""
      elif self._preprocess_mode == 'best_fit':
        # keep -1 as -1
        pass
      # Stack
      self.prior_completion.append(prior_comp)
      self.prior_eligibility.append(prior_elig)
      self.reward_sum.append(prior_rew)
      self.reward_count.append(prior_rew_count)
    self.prior_completion = np.swapaxes(np.stack(self.prior_completion), 0, 1)
    self.prior_eligibility = np.swapaxes(np.stack(self.prior_eligibility), 0, 1)
    self.reward_sum = np.stack(self.reward_sum)
    self.reward_count = np.stack(self.reward_count)

  def infer_task(self) -> SubtaskGraph:
    completions = self._completion_buffer[:self._step_count, :, :]      # T x batch x 13
    eligibilities = self._eligibility_buffer[:self._step_count, :, :]   # T x batch x 13
    rew_counts = self.reward_count        # batch x 13
    rew_tensors = self.reward_sum         # batch x 13
    if self._use_prior:
      completions = np.concatenate([completions, self.prior_completion], axis=0) # concat along T axis
      eligibilities = np.concatenate([eligibilities, self.prior_eligibility], axis=0) # concat along T axis
      #
      """ TODO: expansion ver
      expanded_prior_comp, expanded_prior_elig = self.expand_dc(self.prior_completion, self.prior_eligibility)
      expanded_completions = np.concatenate([completions, expanded_prior_comp], axis=0) # concat along T axis
      expanded_eligibilities = np.concatenate([eligibilities, expanded_prior_elig], axis=0) # concat along T axis"""

    graphs = []
    for i in range(self._batch_size):
      completion = completions[:, i, :]      # (T, num_subtasks)
      eligibility = eligibilities[:, i, :]   # (T, num_subtasks)
      rew_count = rew_counts[i]          # (num_subtasks,)
      rew_tensor = rew_tensors[i]        # (num_subtasks,)

      #0. initialize graph
      graph = SubtaskGraph()
      graph.num_data = self._step_count
      graph.numP, graph.numA = [self.ntasks], []
      graph.index_to_pool = self._index_to_pool[i]
      graph.pool_to_index = self._pool_to_index[i]

      #1. update subtask reward
      graph.subtask_reward = self._infer_reward(rew_count, rew_tensor) #mean-reward-tracking

      #2. Infer preconditions
      if self.noisy_flag[i]:
        W_a, W_o, ANDmat, ORmat, tind_by_layer = self._infer_noisy_precondition(completion, eligibility)
      else:
        W_a, W_o, ANDmat, ORmat, tind_by_layer = self._infer_precondition(completion, eligibility)

      #5. fill-out params.
      graph.W_a = W_a
      graph.W_o = W_o
      graph.ANDmat = ANDmat
      graph.ORmat = ORmat
      graph.tind_by_layer = tind_by_layer
      graph.numP, graph.numA = [], []
      for i in range( len(tind_by_layer) ):
        graph.numP.append( len(tind_by_layer[i]) )
      for i in range( len(graph.W_a) ):
        graph.numA.append( graph.W_a[i].shape[0] )
      graphs.append(graph)
    self._inferred_graphs = graphs
    return graphs

  """def expand_dc(self, completions, eligibilities):
    dc_mask = completions.sum(axis=0) < 0
    num_dc = dc_mask.sum()
    dc_list = list(map(list, itertools.product([0, 1], repeat=num_dc)))
    return completions, eligibilities"""

  def _infer_noisy_precondition(self, completion, eligibility):
    #0. find the correct layer for each node
    is_not_flat, subtask_layer, tind_by_layer = self._noisy_locate_node_layer(completion, eligibility) #20%

    if is_not_flat:
      # 0. Assign layers to each subtask
      ever_elig = eligibility.sum(0)
      #'subtask_layer' is already filled out in 'update()'
      Kmap_set, cand_ind_list = [None] * self.ntasks, []
      max_layer = subtask_layer.max()

      if self._verbose:
        print('ever_elig=', ever_elig)

      for layer_ind in range(1, max_layer + 1):
        cand_ind_list = cand_ind_list + tind_by_layer[layer_ind - 1]  # previous layers
        nFeature = len(cand_ind_list)

        for ind in range(self.ntasks):
          if subtask_layer[ind] == layer_ind:
            if ever_elig[ind] > 0:
              inputs = completion[:, cand_ind_list]
              targets = eligibility[:, ind]
              mask = np.ones(nFeature, dtype=np.int)
              root = self.cart_train(mask, inputs, targets) #1.8
              Kmap_tensor_org = self.decode_cart(root, nFeature) #0.08
              Kmap_tensor = self.simplify_Kmap(Kmap_tensor_org) #0.12
              # TODO: add helpful assert message
              assert Kmap_tensor.ndim == 2
              Kmap_set[ind] = Kmap_tensor
            else:
              print('ind=', ind)
              print('ever_elig=',ever_elig)
              # TODO: add helpful assert message
              assert False
      return self._kmap_to_mat(Kmap_set, tind_by_layer)
    else:
      return [], [], np.zeros(shape=(0)), np.zeros(shape=(0)), tind_by_layer

  def _noisy_locate_node_layer(self, completion, eligibility):
    subtask_layer = np.ones(self.ntasks, dtype=np.int) * (-1)

    # Main idea: if there exists at least one assignment for -1's, it is valid.

    #1. update subtask_layer / tind_by_layer
    cand_ind_list, cur_layer = [], []
    infer_flag = False
    for i in range(self.ntasks):
      comp = completion[:, i]
      elig = eligibility[:, i]
      comp = comp[comp >= 0] # Mask-out DC's
      elig = elig[elig >= 0] # Mask-out DC's
      #
      num_comp = len(comp)
      num_elig = len(elig)
      comp_count = comp.sum()
      elig_count = elig.sum()
      if i==23:
        import ipdb; ipdb.set_trace()
      if comp_count == 0 or comp_count == num_comp:
        # Rule 1. Remove from feature (i.e., layer=-2) if
        # always/never completed (i.e., we can replace the feature with True/False)      
        subtask_layer[i] = -2 # Ignore
        cur_layer.append(i)

      elif elig_count == 0 or elig_count == num_elig:
        # Rule 2. Assign first layer if
        # always/never eligible (i.e., precondition = True/False)
        subtask_layer[i] = 0 # First layer
        cur_layer.append(i)
        cand_ind_list.append(i) # Consider as feature for other subtasks

      else:
        # Rule 3. Else, TBD later
        subtask_layer[i] = -1 # Not determined yet
        infer_flag = True

    tind_by_layer = [cur_layer]
    for layer_ind in range(1, self.ntasks):
      cur_layer = []
      for tind in range(self.ntasks):
        if subtask_layer[tind] == -1:  # among remaining tind
          inputs = completion[:, cand_ind_list]
          targets = eligibility[:, tind]

          assert inputs.ndim == 2 and targets.ndim == 1
          if np.any(targets < 0):
            # If ? exists in eligibility, remove that data point for testing validity
            mask = targets >= 0
            inputs = inputs[mask, :]
            targets = targets[mask]
            
          if np.any(inputs < 0):
            is_valid = self._check_noisy_validity_simple(inputs, targets)
            # is_valid = self._check_noisy_validity(inputs, targets) # TODO
          else:
            is_valid = self._check_validity(inputs, targets)
          if is_valid:  # add to cur layer
            subtask_layer[tind] = layer_ind
            cur_layer.append(tind)

      if len(cur_layer) > 0:
        tind_by_layer.append(cur_layer)
        cand_ind_list = cand_ind_list + cur_layer
      else:  # no subtask left.
        nb_subtasks_left = (subtask_layer == -1).astype(dtype=np.int32).sum().item()
        if nb_subtasks_left > 0:
          assert False, 'Error! There is a bug in the precondition!'
        assert nb_subtasks_left == 0
        break

    # result:
    if self._verbose:
      print('subtask_layer:', subtask_layer)
      print('tind_by_layer:', tind_by_layer)
    return infer_flag, subtask_layer, tind_by_layer

  def _check_noisy_validity_simple(self, inputs, targets):
    # 1. remove data point with -1
    mask = np.all(inputs >= 0, axis=0)
    refined_inputs = inputs[mask]
    refined_targets = targets[mask]

    # 2. run regular validity
    return self._check_validity(refined_inputs, refined_targets)
  
  def _check_noisy_validity(self, inputs, targets):
    # check if there exists any i1 and i2 such that inputs[i1]==inputs[i2] and targets[i1]!=targets[i2]
    # if there exists, it means the node is not valid, and it should be in the higher layer in the graph.
    tb = {}
    nstep = inputs.shape[0] #new
    code_batch = graph_utils.batch_bin_encode(inputs)
    for i in range(nstep):
      code = code_batch[i]
      target = targets[i].item()
      if code in tb:
        if tb[code] != target:
          return False
      else:
        tb[code] = target

    return True

  @property
  def ilp_data(self) -> dict:
    ilp_data = [dict(
        index_to_pool=self.index_to_pool[i],
        pool_to_index=self.pool_to_index[i],
        reward_sum=self.reward_sum[i],
        reward_count=self.reward_count[i],
        step_count=self._step_count,
        completion=self._completion_buffer[:self._step_count, i, :],
        eligibility=self._eligibility_buffer[:self._step_count, i, :],
    ) for i in range(self._batch_size)]
    return ilp_data


