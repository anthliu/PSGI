"""Inductive Logic Programming (ILP) module implementation."""

from typing import Optional, Dict, List

import os
import numpy as np
import math
import time
import itertools
from psgi.utils import log_utils

import dm_env
from acme import specs

import psgi
from psgi.utils import graph_utils, predicate_utils
from psgi.utils.graph_utils import SubtaskGraph, dotdict, OptionSubtaskGraphVisualizer
from psgi.envs.predicate_graph import PredicateLogicGraph

DUMMY_ACTION = -1
MAX_GINI = 2.0
GINI_REWEIGHTING = True

MAX_CYCLE = 1

class FeatureSet(object):
  def __init__(self, nparams, param_emb_size, param_to_idx, feature_labels):
    self.nparams = nparams
    self.param_emb_size = param_emb_size
    self.nfeatures = param_emb_size
    self.param_to_idx = param_to_idx
    self.feature_labels = feature_labels

  def forward(self, parameter_embeddings):
    assert self.nparams == parameter_embeddings.shape[1]
    assert self.param_emb_size == parameter_embeddings.shape[2]
    self.parameter_embedding = parameter_embeddings[0] > 0.5# Assume embeddings are the same for now
    return parameter_embeddings > 0.5

class PILP:

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
    # Create ILP parameters.
    self._action_spec = environment_spec.actions
    self._total_timesteps = num_adapt_steps * 2 # since we store T+1 states for episode of length T, it can be up to * 2
    self._bonus_mode = bonus_mode
    self._verbose = verbose

    # Default setting for PSGI
    self.infer_pred_subtasks = True
    self.infer_part_subtasks = True
    self.infer_lit_subtasks = True
    self.infer_pred_options = True
    self.infer_lit_options = False
    self.only_pred_effects = True

    # TODO: Remove this.
    # these are shared among batch
    self._tr_epi = tr_epi
    self._environment_id = environment_id

    # Extra features
    ### duplication filtering
    self._duplication_filter_flag = False
    self._unique_indices_by_batch_ind = None # Initialized in reset()
    self._hash_table_by_batch = None # Initialized in reset()
    ### neccessary-first
    self._necessary_first = branch_neccessary_first


    # Create subtask graph visualizer.
    self._visualize = visualize
    if self._visualize:
      assert environment_id is not None, \
        'Please specify environment id for graph visualization.'
      self._graph_visualizer = OptionSubtaskGraphVisualizer()
      self._visualize_directory = directory

    # Count steps and trials.
    self._step_count = 0
    self._trial_count = 0
    self._graphs = None
    self._kmap_sets = None

  def switch_to_msgi(self):
    self.infer_pred_subtasks = False
    self.infer_part_subtasks = False                                                     
    self.infer_lit_subtasks = True
    self.infer_pred_options = False                                                      
    self.infer_lit_options = True
    self.only_pred_effects = False

  @property
  def graphs(self):
    return self._graphs

  @property
  def trajectory(self) -> List[Dict]:
    trajectory = []
    for i in range(self._batch_size):
      unique_indices = self._unique_indices_by_batch_ind[i]
      data = dict(
          reward_sum=self.reward_sum[i],
          reward_count=self.comp_count[i],
          step_count=self._step_count,
          completion=self._completion_buffer[unique_indices, i, :],
          eligibility=self._eligibility_buffer[unique_indices, i, :],
      )
      trajectory.append(data)
    return trajectory

  @property
  def kmaps(self):
    return self._kmap_sets

  def reset(self, environment: dm_env.Environment):
    # Reset/increment counts.
    self._batch_size = environment.batch_size
    self._step_count = 0
    self._trial_count += 1

    # Create ILP buffers.
    self.n_literal_options = max(environment.num_options)  # XXX each env may have different # of subtasks
    self.n_literal_subtasks = max(environment.num_subtasks)
    assert 'parameter_embeddings' in environment.observation_spec(), 'Environment observations must include parameter_embeddings'
    self.nparams, self.param_emb_size = environment.observation_spec()['parameter_embeddings'].shape
    self._completion_buffer = np.zeros((self._total_timesteps, self._batch_size, self.n_literal_subtasks), dtype=np.bool)
    self._eligibility_buffer = np.zeros((self._total_timesteps, self._batch_size, self.n_literal_options), dtype=np.bool)
    self._action_buffer = np.full((self._total_timesteps, self._batch_size, 1), DUMMY_ACTION, dtype=np.int32)# XXX need to track actions to infer effect
    self._p_embeddings_buffer = np.zeros((self._total_timesteps, self._batch_size, self.nparams, self.param_emb_size))

    self.reward_sum = np.zeros((self._batch_size, self.n_literal_subtasks))
    self.comp_count = np.zeros((self._batch_size, self.n_literal_subtasks), dtype=np.long)

    self.option_param = environment.envs[0]._environment._environment.option_param# TODO Allow wrappers to access subtask param. Should we make this different for different batches, or assume same graph?
    self.subtask_param = environment.envs[0]._environment._environment.subtask_param
    self.param_name_to_idx = environment.envs[0]._environment._environment.parameter_name_to_index# TODO update wrappers for this
    self.option_param = environment.envs[0]._environment._environment.option_param
    self.feature_labels = environment.envs[0]._environment._environment.config.feature_func_names

    self._feature_set = FeatureSet(nparams=self.nparams, param_emb_size=self.param_emb_size, param_to_idx=self.param_name_to_idx, feature_labels=self.feature_labels)
    self.nfeatures = self._feature_set.nfeatures

    self._pcandidate_graph = predicate_utils.PCandidateGraph(
        self.option_param, self.subtask_param,
        infer_literal_subtasks=self.infer_lit_subtasks,
        infer_partial_subtasks=self.infer_part_subtasks,
        infer_predicate_subtasks=self.infer_pred_subtasks,
        infer_literal_options=self.infer_lit_options,
        infer_predicate_options=self.infer_pred_options
    )
    self.precomputed_indices = self._pcandidate_graph.precompute_pilp_indices(num_parameters=self.nparams, feature_dim=self.nfeatures, feature_labels=self.feature_labels, param_to_idx=self.param_name_to_idx)
    self.n_p_subtasks = self._pcandidate_graph.all_p_subtask_size
    self.n_p_options = self._pcandidate_graph.all_p_option_size
    self.feature_start_idx = self._pcandidate_graph.feature_start_idx
    self._option_label = [p_option.pretty() for p_option in self._pcandidate_graph.all_p_options]
    self._literal_option_label = [p_option.pretty() for p_option in self._pcandidate_graph.literal_p_options]
    self._subtask_label = [p_subtask.pretty() for p_subtask in self._pcandidate_graph.all_p_subtasks]
    self._literal_subtask_label = [p_subtask.pretty() for p_subtask in self._pcandidate_graph.literal_p_subtasks]
    self._parameter_label = {i: name for name, i in self.param_name_to_idx.items()}# TODO just use environment property for this

    if self._bonus_mode > 0:
      self.hash_table = [set() for i in range(self._batch_size)]
      # TODO: Remove _tr_epi
      # TODO double check that this is still correct with n_literal_options
      self.base_reward = min(10.0 / self._tr_epi / self.n_literal_options, 1.0)
      if self._bonus_mode == 2:
        self.pn_count = np.zeros((self._batch_size, self.n_literal_options, 2))

    # Get task index.
    #self._task_indices = [c.seed for c in environment.task] # XXX: not compatible with mining
    self._task_indices = [0] * self._batch_size

    # Reset hash table.
    if self._duplication_filter_flag:
      raise NotImplementedError
      self._hash_table_by_batch = [set() for _ in range(self._batch_size)]
      self._unique_indices_by_batch_ind = [[] for _ in range(self._batch_size)]

  def insert(
      self,
      is_valid: np.ndarray,  # 1-dim (batch_size,)
      completion: np.ndarray,   # 2-dim (batch_size, num_subtasks)
      eligibility: np.ndarray,  # 2-dim (batch_size, num_subtasks)
      parameter_embeddings: np.ndarray, # 3-dim (batch_size, num_params, p_embedding_size)
      action_id: Optional[np.ndarray],  # 2-dim (batch_size, 1)
      rewards: Optional[np.ndarray] = None      # 2-dim (batch_size, 1)
  ):
    """Store the experiences into ILP buffer."""
    # Get unique indices & update hash table
    if self._duplication_filter_flag:
      raise NotImplementedError
      batch_codes = graph_utils.batch_bin_encode(completion.astype(np.bool))
      for batch_index in range(self._batch_size):
        code = batch_codes[batch_index]
        hash_table = self._hash_table_by_batch[batch_index]
        is_unique = code not in hash_table
        if is_unique:
          hash_table.add(code)
          self._unique_indices_by_batch_ind[batch_index].append(self._step_count)

    self._completion_buffer[self._step_count] = completion.copy()
    self._eligibility_buffer[self._step_count] = eligibility.copy()
    self._p_embeddings_buffer[self._step_count] = parameter_embeddings.copy()

    if action_id is not None:
      active = np.expand_dims(is_valid.astype(dtype=np.int32), -1)
      assert active.ndim == 2 and active.shape[-1] == 1

      # For inactive batches, replace with DUMMY_ACTION to avoid error in _get_index_from_pool()
      action_id[active < 0.5] = DUMMY_ACTION
      #act_inds = (self._get_index_from_pool(action_id) * active).astype(np.int32)
      act_inds = action_id# XXX Local is global now, so action_id is same

      self._action_buffer[self._step_count] = action_id
    else: # initial state
      self._action_buffer[self._step_count] = DUMMY_ACTION

    # reward
    if rewards is not None:
      delta_comp = completion - self.prev_completion
      delta_comp = delta_comp.astype(dtype=np.int32) * active
      # HACK: only extract positive delta. 
      delta_comp = np.clip(delta_comp, a_min=0, a_max=1)

      # XXX Estimate option reward with more than two positive effects by even split
      # rewards: [batch_size x 1]
      # delta_comp: [batch_size x num_subtask] (one-hot or all-zero)
      reward_mat = delta_comp * rewards / np.clip(delta_comp.sum(axis=1, keepdims=True), a_min=1, a_max=None)

      #assert np.all(delta_comp.sum(axis=1) <= 1), "Only one positive effect is allowed!"

      self.reward_sum += reward_mat
      self.comp_count += delta_comp.astype(np.long)

    # Increment step count.
    self._step_count += 1
    self.prev_completion = completion

  def infer_task(self) -> List[SubtaskGraph]:
    assert self._check_data(), "ilp data is invalid!"
    graphs, kmap_sets, effect_mats = [], [], []
    for i in range(self._batch_size):
      # TODO Add a duplication filter for every augmented table
      if self._duplication_filter_flag:
        raise NotImplementedError
      # TODO currently implemented for batch size 1 only
      completion, unique_idx = np.unique(self._completion_buffer[:self._step_count, i, :], return_index=True, axis=0)
      eligibility = self._eligibility_buffer[unique_idx, i, :]
      p_embeddings = self._p_embeddings_buffer[unique_idx, i, :, :]

      #0. initialize graph
      graph = PredicateLogicGraph('Inferred graph')

      #1. Infer reward (prior reward is already merged in "update_prior()")
      literal_subtask_reward = self._infer_reward(self.comp_count[i], self.reward_sum[i]) #mean-reward-tracking

      completion, eligibility = self._build_comp_elig_table(
          completion_literals=completion,
          eligibility_literals=eligibility,
          parameter_embeddings=p_embeddings,
      )

      #1.5 Infer effect
      effect_mat, effect_any_mat = self._infer_effect(
          actions=self._action_buffer[:self._step_count, i, :],
          completion_literals=self._completion_buffer[:self._step_count, i, :],
      )

      #2. Infer precondition
      kmap_set, tind_by_layer, subtask_layer = self._infer_precondition(
          completion=completion,
          eligibility=eligibility,
          effect_any_mat=effect_any_mat,
      )
      if self._visualize and i == 0:
        dot = self._graph_visualizer.visualize(kmap_set, effect_mat, subtask_label=self._subtask_label, option_label=self._option_label)
        filepath = f"{self._visualize_directory}/graph_step{self._step_count}"
        self._graph_visualizer.render_and_save(dot, path=filepath)

      #3. Fill-out graph edges from precondition kmap
      graph.initialize_from_kmap(
        kmap_set, effect_mat, self._subtask_label, 
        self._option_label, self._literal_subtask_label, literal_subtask_reward)
      kmap_sets.append(kmap_set)
      graphs.append(graph)
      effect_mats.append(effect_mat)
    self._graphs = graphs
    self._kmap_sets = kmap_sets
    self._effect_mats = effect_mats
    return graphs

  def _check_data(self):
    for batch_index in range(self._batch_size):
      completion = self._completion_buffer[:self._step_count, batch_index, :]
      eligibility = self._eligibility_buffer[:self._step_count, batch_index, :]
      for i in range(completion.shape[1]):
        comp = np.concatenate((completion[:, :i], completion[:, i+1:]), axis=1)
        elig = eligibility[:, i]
        if not self._check_validity(comp, elig, debug=True):
          return False
    return True

  def _build_comp_elig_table(
      self,
      completion_literals: np.ndarray,   # (T, num_subtasks), dtype=np.bool
      eligibility_literals: np.ndarray,  # (T, num_subtasks), dtype=np.bool
      parameter_embeddings: np.ndarray):# (T, num_params, p_embedding_size)

    # 0. assertions
    assert completion_literals.shape[0] == eligibility_literals.shape[0] and completion_literals.ndim == 2, "shape error"
    assert completion_literals.dtype == eligibility_literals.dtype and completion_literals.dtype == np.bool, "completion, eligibility should be np.bool type"
    assert parameter_embeddings.shape[0] == completion_literals.shape[0] and parameter_embeddings.ndim == 3, "shape error in parameter_embeddings"

    num_samples, num_literal_subtasks = completion_literals.shape
    # 1 Pilp step --- Compute feature values from parameter_embeddings
    feature_values = self._feature_set.forward(parameter_embeddings)
    assert feature_values.shape[0] == num_samples \
        and feature_values.shape[1] == self.nparams \
        and feature_values.shape[2] == self.nfeatures

    num_samples, num_literal_subtasks = completion_literals.shape
    eligibility = np.zeros((num_samples * self.precomputed_indices['nx_rows'], self.n_p_options), dtype=np.bool)
    completion = np.zeros((num_samples * self.precomputed_indices['nx_rows'], self.n_p_subtasks), dtype=np.bool)
    for stack_idx in range(self.precomputed_indices['nx_rows']):
      start_idx = stack_idx * num_samples
      end_idx = start_idx + num_samples
      eligibility[start_idx:end_idx, :] = eligibility_literals[:, self.precomputed_indices['eligibility_idx_mat'][stack_idx]]
      completion[start_idx:end_idx, :self.feature_start_idx] = completion_literals[:, self.precomputed_indices['completion_idx_mat'][stack_idx]]
      completion[start_idx:end_idx, self.feature_start_idx:] = feature_values[:, self.precomputed_indices['feature_param_mat'][stack_idx], self.precomputed_indices['feature_idx_mat'][stack_idx]]

    return completion, eligibility

  def _infer_effect(
      self,
      actions: np.ndarray,               # (T, 1, dtype=np.int32)
      completion_literals: np.ndarray):  # (T, num_literal_subtasks), dtype=np.bool
    assert actions.shape[0] == completion_literals.shape[0]
    assert actions[0] == DUMMY_ACTION, "first timesteps after reset should have no action"
    assert actions[actions != DUMMY_ACTION].max() < self.n_literal_options and actions[actions != DUMMY_ACTION].min() >= 0, "invalid actions"
    actions = actions[1:]
    valid_action_mask = actions.squeeze() != DUMMY_ACTION
    comp_literals_before = completion_literals[:-1][valid_action_mask]
    comp_literals_after = completion_literals[1:][valid_action_mask]
    actions = actions[valid_action_mask]
    assert comp_literals_before.shape == comp_literals_after.shape and actions.shape[0] == comp_literals_before.shape[0], "completion mismatch with actions"
    num_samples = actions.shape[0]

    p_actions = np.zeros((num_samples, self.n_p_options), dtype=np.bool)
    comp_before = np.zeros((num_samples, self.n_p_subtasks), dtype=np.bool)
    comp_after = np.zeros((num_samples, self.n_p_subtasks), dtype=np.bool)

    _actions_squeezed = actions.squeeze()
    p_actions = self.precomputed_indices['option_parents'][_actions_squeezed]
    candidate_effect_mask = self.precomputed_indices['effect_mask_mat'][_actions_squeezed]
    _p_effect_idxs = self.precomputed_indices['effect_idx_mat'][_actions_squeezed]
    comp_before = np.take_along_axis(comp_literals_before, _p_effect_idxs, axis=1)
    comp_after = np.take_along_axis(comp_literals_after, _p_effect_idxs, axis=1)

    comp_changed = np.logical_xor(comp_before, comp_after)
    comp_changed = np.logical_and(comp_changed, candidate_effect_mask)# only consider changed subtasks within the mask
    comp_positive = np.logical_and(comp_changed, comp_after)
    comp_negative = np.logical_and(comp_changed, comp_before)
    # effect_mat(i, j) = 1 if p_option_i always effects p_subtask_j
    effect_mat = np.zeros((self.n_p_options, self.n_p_subtasks), dtype=np.int32)
    # effect_any(i, j) = 1 if p_option_i may effects p_subtask_j
    effect_any_mat = np.zeros((self.n_p_options, self.n_p_subtasks), dtype=np.bool)

    for p_option_idx in range(self.n_p_options):
      sample_mask = p_actions[:, p_option_idx]
      if not np.any(sample_mask):
        if self.infer_pred_options: # skip this for MSGI
          print(f'no samples found for: {self._option_label[p_option_idx]}! Skipping effect inference')
        # if no actions with samples yet, skip
        continue

      # candidate: there is some sample that changed from negative to positive
      # final: all samples ended on positive
      # if candidate and final, set effect to 1
      CHANGE_THRESH = 1
      #PURE_THRESH = 0.99
      cand_comp_changed = np.any(comp_changed[sample_mask], axis=0)

      cand_comp_positive = np.sum(comp_positive[sample_mask], axis=0) > CHANGE_THRESH
      #final_comp_positive = np.mean(comp_after[sample_mask], axis=0) > PURE_THRESH
      final_comp_positive = np.all(comp_after[sample_mask], axis=0)

      cand_comp_negative = np.sum(comp_negative[sample_mask], axis=0) > CHANGE_THRESH
      #final_comp_negative = np.mean(np.logical_not(comp_after[sample_mask]), axis=0) > PURE_THRESH
      final_comp_negative = np.all(np.logical_not(comp_after[sample_mask]), axis=0)

      effect_mat[p_option_idx, np.logical_and(cand_comp_positive, final_comp_positive)] = 1
      effect_mat[p_option_idx, np.logical_and(cand_comp_negative, final_comp_negative)] = -1

      effect_any_mat[p_option_idx, cand_comp_changed] = True
      effect_any_mat[p_option_idx] = effect_any_mat[p_option_idx].dot(self.precomputed_indices['subtask_intersection'])# if pickup(A) changes, so does pickup(B)
      #print(self._option_label[p_option_idx], [f'{effect_mat[p_option_idx,i]}{self._subtask_label[i]}' for i in np.flatnonzero(effect_mat[p_option_idx])])
      #print(self._option_label[p_option_idx], [self._subtask_label[i] for i in np.flatnonzero(effect_any_mat[p_option_idx])])


    if self.only_pred_effects:
      # Only infer effects for full parameratized subtask completions
      priority = self.precomputed_indices['completion_priority']
      mask = priority < priority.max()
      effect_mat[:, mask] = 0


    return effect_mat, effect_any_mat

  def _infer_precondition_option(
      self,
      completion: np.ndarray,    # (T, num_p_subtasks), dtype=np.bool
      eligibility: np.ndarray,   # (T, num_p_options), dtype=np.bool
      completion_mask: np.ndarray,
      option: int):

    candidate_mask = self.precomputed_indices['candidate_mask_mat'][option]
    mask = np.logical_and(completion_mask, candidate_mask)

    if not np.any(mask):
      return None

    comp = completion[:, mask]
    elig = eligibility[:, option]
    _, unique_idx = np.unique(comp, return_index=True, axis=0) # 80% time
    is_valid = self._check_validity(comp, elig)

    if is_valid: # NOTE: MUST use non-compact-ver for checking validity
      #print('*'* 10 + f'Inferring option: {self._option_label[option]}')
      feat_dim = comp.shape[-1]
      mask_labels = [l for i, l in enumerate(self._subtask_label) if mask[i]]
      root = self.cart_train(np.ones(feat_dim, dtype=np.bool), comp[unique_idx], elig[unique_idx], mask_labels=mask_labels) # 20% time
      kmap_tensor_org = self.decode_cart(root, feat_dim) # 0.5%
      kmap_tensor = self.simplify_Kmap(kmap_tensor_org) # 0.6%
      assert kmap_tensor.ndim == 2, "kmap_tensor should be 2 dimension"
      kmap_tensor_remapped = np.zeros((kmap_tensor.shape[0], completion.shape[1]), dtype=kmap_tensor.dtype)
      kmap_tensor_remapped[:, mask] = kmap_tensor
      return kmap_tensor_remapped
    else:
      return None

  def _precondition_mask(
      self,
      zero_layer_comp_mask: np.ndarray,# (num_p_subtasks)
      effect_any_mat: np.ndarray,#(num_p_options, num_p_subtasks) dtype=np.bool
      inferred_options: List[int],
      priority: int):
    mask = zero_layer_comp_mask.copy()
    mask = mask | np.any(effect_any_mat[inferred_options])

    completion_priority = self.precomputed_indices['completion_priority']
    priority_mask = completion_priority >= priority
    mask = mask & priority_mask
    return mask

  def _infer_precondition(
      self,
      completion: np.ndarray,    # (T, num_p_subtasks), dtype=np.bool
      eligibility: np.ndarray,   # (T, num_p_options), dtype=np.bool
      effect_any_mat: np.ndarray):#(num_p_options, num_p_subtasks) dtype=np.bool    
    assert completion.shape[0] == eligibility.shape[0] and completion.ndim == 2, "shape error"
    assert completion.shape[1] == self.n_p_subtasks and eligibility.shape[1] == self.n_p_options, "shape error"
    assert completion.dtype == eligibility.dtype and completion.dtype == np.bool, "completion, eligibility should be np.bool type"
    assert effect_any_mat.shape[1] == completion.shape[1] and effect_any_mat.shape[0] == eligibility.shape[1] and effect_any_mat.dtype == np.bool, "effect any mat input error"

    # 1. Find first layer: either always elig or never elig, while ignoring DC
    always_elig = np.all(eligibility, axis=0)
    never_elig = np.all(np.logical_not(eligibility), axis=0)
    pickup_option_mask = np.array(['op_pickup' in label for label in self._option_label]) # XXX: HACK
    zero_layer_option_mask = always_elig | never_elig | pickup_option_mask

    # Find first layer completion -- all completions that are constant across resets
    zero_layer_comp_mask = np.all(completion, axis=0) | np.all(~completion, axis=0)
    zero_layer_comp_mask[self.feature_start_idx:] = True

    inferred_options = set(np.flatnonzero(zero_layer_option_mask).tolist())
    all_options = set(range(self.n_p_options))
    option_layer = np.full((self.n_p_options), fill_value=-1, dtype=np.int16)
    option_layer[zero_layer_option_mask] = 0
    kmap_set = [None] * self.n_p_options
    for option_index in range(len(zero_layer_option_mask)):
      if always_elig[option_index] or pickup_option_mask[option_index]:
        kmap_set[option_index] = True
      if never_elig[option_index]:
        kmap_set[option_index] = False
    tind_by_layer = [list(inferred_options)]

    # Layer-wise-layer-wise, try to infer precond with only higher priority completions first
    completion_priority = self.precomputed_indices['completion_priority']
    max_priority = completion_priority.max()
    min_priority = completion_priority.min()

    # memoize completions required for each option
    self.prev_completion_mask = {}

    for layer in range(1, self.n_p_options + 1):
      if len(inferred_options) == len(all_options):
        break

      option_kmaps = {}
      leftover_options = all_options - inferred_options

      for priority in range(max_priority, min_priority - 1, -1):
        for assume_n in range(0, MAX_CYCLE+1):
          for assume_options in itertools.combinations(leftover_options, assume_n):
            assume_options = set(assume_options)
            temp_inferred_options = inferred_options | assume_options
            temp_leftover_options = leftover_options - assume_options

            for target_option in temp_leftover_options:
              self.prev_completion_mask[target_option] = self._precondition_mask(zero_layer_comp_mask, effect_any_mat, list(temp_inferred_options), priority)
              result = self._infer_precondition_option(completion, eligibility, self.prev_completion_mask[target_option], target_option) # 99% time
              if result is not None:
                option_kmaps[target_option] = result

            if len(option_kmaps) > 0:
              inferred_options.update(option_kmaps.keys())
              for assume_target_option in assume_options:
                self.prev_completion_mask[assume_target_option] = self._precondition_mask(zero_layer_comp_mask, effect_any_mat, list(inferred_options), priority)
                result = self._infer_precondition_option(completion, eligibility, self.prev_completion_mask[assume_target_option], assume_target_option) # 99% time
                if result is not None:
                  inferred_options.add(assume_target_option)
                  option_kmaps[assume_target_option] = result
              break
            """else:
              if assume_n > 0:
                print(f'cyclic layer-wise ILP failed with priority {priority}, assume {[self._option_label[i] for i in assume_options]}, options {[self._option_label[i] for i in np.flatnonzero(option_layer==-1)]}. retrying with different assumptions.')"""

          if len(option_kmaps) > 0:
            break
          """else:
            print(f'acyclic layer-wise ILP failed with priority {priority}, options {[self._option_label[i] for i in np.flatnonzero(option_layer==-1)]}. retrying with cycles.')"""

        if len(option_kmaps) > 0:
          break
        """else:
          print(f'layer-wise ILP failed with priority {priority}, options {[self._option_label[i] for i in np.flatnonzero(option_layer==-1)]}. retrying with lower priority.')"""

      assert len(option_kmaps) > 0, f"layer-wise ILP failed to find precondition for subtasks even with cycle conditions AND priority {[self._option_label[i] for i in np.flatnonzero(option_layer==-1)]}"
      for target_option, kmap_tensor in option_kmaps.items():
        kmap_set[target_option] = kmap_tensor
        option_layer[target_option] = layer
      tind_by_layer.append(list(option_kmaps.keys()))

    assert np.all(option_layer >= 0)
    return kmap_set, tind_by_layer, option_layer


  def _check_validity(self, inputs, targets, debug=False):
    # inputs: np.arr((T, ntasks), dtype=bool) (only 0/1)
    # targets: np.arr((T,), dtype=bool) (only 0/1)
    assert inputs.dtype == np.bool and targets.dtype == np.bool, "type error"
    assert inputs.ndim == 2 and targets.ndim == 1, "shape errror"

    # check if there exists any i1 and i2 such that inputs[i1]==inputs[i2] and targets[i1]!=targets[i2]
    # if there exists, it means the node is not valid, and it should be in the higher layer in the graph.
    if np.all(targets) == False and np.any(targets) == True:
      code_list = np.array(graph_utils.batch_bin_encode(inputs), dtype=np.int64)
      eligible_code_set = set(code_list[targets])
      ineligible_code_set = set(code_list[np.logical_not(targets)])
      is_valid = eligible_code_set.isdisjoint(ineligible_code_set)
      if not is_valid and debug:
        print('Error!! Not valid')
        import ipdb; ipdb.set_trace()
      return is_valid
    else:
      return True

  def _infer_reward(self, reward_count, reward_tensor): #. mean-reward
    # reward_count: Ntasks
    # reward_tensor: Ntasks
    subtask_reward = np.zeros(self.n_literal_subtasks)
    nonzero_mask = reward_count > 0
    # self.reward_sum
    subtask_reward[nonzero_mask] = reward_tensor[nonzero_mask] / reward_count[nonzero_mask]
    rmean = subtask_reward[nonzero_mask].sum() / len(nonzero_mask)
    # Optimism in the face of uncertainty
    rmean = rmean + abs(rmean)*0.1 if abs(rmean) > 0. else rmean + 0.1
    #subtask_reward[np.logical_not(nonzero_mask)] = rmean
    subtask_reward[np.logical_not(nonzero_mask)] = 0.0  # default to zero

    return subtask_reward

  # CART functions. Don't touch unless necessary
  def decode_cart(self, root, nFeature):
    Kmap = []
    stack = []
    instance = dotdict()
    instance.lexp = np.zeros((nFeature), dtype=np.int8)
    instance.node = root
    stack.append(instance)

    while len(stack) > 0:
      node = stack[0].node
      lexp = stack[0].lexp
      stack.pop(0)
      featId = node.best_ind

      if node.gini > 0:  # leaf node && positive sample
        # TODO: add helpful assert message
        assert featId >= 0

        if node.left.best_ind >= 0:  # negation
          instance = dotdict()
          instance.lexp = lexp.copy()
          instance.lexp[featId] = -1  # negative
          instance.node = node.left
          stack.append(instance)
        elif node.left.sign > 0:
          n_lexp = lexp.copy()
          n_lexp[featId] = -1
          Kmap.append(n_lexp[None, :])

        if node.right.best_ind >= 0:  # positive
          instance = dotdict()
          instance.lexp = lexp.copy()
          instance.lexp[featId] = 1  # positive
          instance.node = node.right
          stack.append(instance)
        elif node.right.sign > 0:
          n_lexp = lexp.copy()
          n_lexp[featId] = 1
          Kmap.append(n_lexp[None, :])

      elif node.sign == 0:
        lexp[featId] = -1
        Kmap.append(lexp[None, :])
      else:
        lexp[featId] = 1
        Kmap.append(lexp[None, :])

    kmap_tensor = np.concatenate(Kmap, axis=0)

    return kmap_tensor

  def cart_train(self, mask, inputs, targets, mask_labels=None, print_pref=''):
    assert inputs.dtype == np.bool and targets.dtype == np.bool, 'type error'
    assert inputs.ndim == 2, 'inputs should be 2 dim'
    assert inputs.shape[0] > 0, "Error: data is empty!"
    nstep, ncand = inputs.shape
    root = dotdict()
    minval = MAX_GINI + 0.1  # gini range: 0~2.0
    assert (mask.sum() > 0), 'Error: No feature left but classification is incomplete. Data is not separable'

    #if mask_labels:
      #print(f'subtask candidates: {mask_labels}')
    best_ind = None
    for i in range(ncand):
      if mask[i] == True:
        # Compute gini
        left, right, counts = self.compute_gini(inputs[:, i], targets)
        gini = left.gini + right.gini
        #if mask_labels and print_pref == '':
          #print(f'{mask_labels[i]:<20} gini: {gini:>10.6f}, nn:{counts.nn:>6.2f} np:{counts.np:>6.2f} pn:{counts.pn:>6.2f} pp:{counts.pp:>6.2f}')
          #print(f'{mask_labels[i]:<20} gini: {gini:>10.6f}, left-p0:{left.p0:>6.2f} left-p1:{left.p1:>6.2f} right-p0:{right.p0:>6.2f} right-p1:{right.p1:>6.2f}')
        # Test whether neccessary
        bias = 0.
        if self._necessary_first:
          test_input = np.concatenate((inputs[:, :i], inputs[:, i+1:]), axis=1) # form data without target subtask
          if not self._check_validity(test_input.astype(np.bool), targets.astype(np.bool)): # not valid without target subtask -> target subtask is neccessary!
            bias = -1 * (MAX_GINI + 0.1) # higher priority than unneccessary ones
        noise = 1e-5 * np.random.random() # to break ties

        if minval > gini + bias + noise and left.gini < 1.0 and right.gini < 1.0:
          minval = gini + bias + noise
          best_gini = gini
          best_ind = i
          best_left = left
          best_right = right
    assert best_ind is not None, 'Error! no feature has been chosen!'

    root.best_ind = best_ind
    root.gini = best_gini
    mask[best_ind] = False
    #print(f'{print_pref}best st: {mask_labels[best_ind]}')

    assert  best_left.gini < 1.0 and best_right.gini < 1.0, "Error! Left or Right branch is empty. It means best split has no gain"

    if best_gini > 0: # means somethings left for further branch expansion
      if best_left.gini > 0:  # means there exists both 0 and 1
        left_mask = inputs[:, best_ind] == 0
        left_input = inputs[left_mask, :]
        left_targets = targets[left_mask]
        #print(f'{print_pref}{mask_labels[best_ind]}=F => {{')
        root.left = self.cart_train(mask, left_input, left_targets, mask_labels, print_pref=print_pref + '  ')
        #print(f'{print_pref}}}')
      else:
        root.left = dotdict()
        root.left.gini = 0
        root.left.sign = best_left.p1
        root.left.best_ind = -1
        #print(f'{print_pref}{mask_labels[best_ind]}=F => E={best_left.p1>0.5}')

      if best_right.gini > 0:
        right_mask = inputs[:, best_ind] != 0
        right_input = inputs[right_mask, :]
        right_targets = targets[right_mask]
        #print(f'{print_pref}{mask_labels[best_ind]}=T => {{')
        root.right = self.cart_train(mask, right_input, right_targets, mask_labels, print_pref=print_pref + '  ')
        #print(f'{print_pref}}}')
      else:
        root.right = dotdict()
        root.right.gini = 0
        root.right.sign = best_right.p1
        root.right.best_ind = -1
        #print(f'{print_pref}{mask_labels[best_ind]}=T => E={best_right.p1>0.5}')
    else:
      root.sign = best_right.p1  # if right is all True,: sign=1
      #print(f'{print_pref}{mask_labels[best_ind]} <=> E={best_right.p1>0.5}')

    mask[best_ind] = True
    return root

  def compute_gini(self, input_feat, targets):
    assert input_feat.dtype == np.bool and targets.dtype == np.bool, "type error!"
    neg_input = np.logical_not(input_feat)
    neg_target = np.logical_not(targets)
    nn_count = np.logical_and(neg_input, neg_target).sum()   # count[0]
    np_count = np.logical_and(neg_input, targets).sum()      # count[1]
    pn_count = np.logical_and(input_feat, neg_target).sum()  # count[2]
    pp_count = np.logical_and(input_feat, targets).sum()     # count[3]
    assert (nn_count + np_count + pn_count + pp_count == input_feat.shape[0]), "error in computing nn_count, np_count, pn_count, pp_count"

    if GINI_REWEIGHTING:
      _nn_count = nn_count / max(nn_count + pn_count, 1)
      _np_count = np_count / max(np_count + pp_count, 1)
      _pn_count = pn_count / max(nn_count + pn_count, 1)
      _pp_count = pp_count / max(np_count + pp_count, 1)
      nn_count, np_count, pn_count, pp_count = _nn_count, _np_count, _pn_count, _pp_count

    counts = dotdict()
    counts.nn = nn_count
    counts.np = np_count
    counts.pn = pn_count
    counts.pp = pp_count

    left, right = dotdict(), dotdict()
    if nn_count + np_count > 1e-8:
      p0_left = nn_count / (nn_count + np_count)
      p1_left = np_count / (nn_count + np_count)
      left.gini = 1 - pow(p0_left, 2) - pow(p1_left, 2)
      assert left.gini >= 0 and left.gini <= 0.5, "Error: Gini value is out of range [0, 0.5]"
      left.p0 = p0_left
      left.p1 = p1_left
    else:
      left.gini = 1
      left.p0 = 1
      left.p1 = 1

    if pn_count + pp_count > 0:
      p0_right = pn_count / (pn_count + pp_count)
      p1_right = pp_count / (pn_count + pp_count)
      right.gini = 1 - pow(p0_right, 2) - pow(p1_right, 2)
      assert right.gini >= 0 and right.gini <= 0.5, "Error: Gini value is out of range [0, 0.5]"
      right.p0 = p0_right
      right.p1 = p1_right
    else:
      right.gini = 1
      right.p0 = 1
      right.p1 = 1

    return left, right, counts

  def simplify_Kmap(self, kmap_tensor):
    """
      This function performs the following two reductions
      A + AB  -> A
      A + A'B -> A + B

      Kmap_bin: binarized Kmap. (i.e., +1 -> +1, 0 -> 0, -1 -> +1)
    """
    numAND = kmap_tensor.shape[0]
    mask = np.ones(numAND)
    max_iter = 20

    for jj in range(max_iter):
      done = True
      remove_list = []
      Kmap_bin = np.not_equal(kmap_tensor,0).astype(np.uint8)

      for i in range(numAND):
        if mask[i] == 1:
          kbin_i = Kmap_bin[i]

          for j in range(i + 1, numAND):
            if mask[j] == 1:
              kbin_j = Kmap_bin[j]
              kbin_mul = kbin_i * kbin_j

              if np.all(kbin_mul == kbin_i):  # i subsumes j. Either 1) remove j or 2) reduce j.
                done = False
                Kmap_common_j = kmap_tensor[j] * kbin_i  # common parts in j.
                difference_tensor = Kmap_common_j != kmap_tensor[i]  # (A,~B)!=(A,B) -> 'B'
                num_diff_bits = np.sum(difference_tensor)

                if num_diff_bits == 0:  # completely subsumes --> remove j.
                  mask[j] = 0
                else:  # turn off the different bits
                  dim_ind = np.nonzero(difference_tensor)[0]
                  kmap_tensor[j][dim_ind] = 0

              elif np.all(kbin_mul == kbin_j):  # j subsumes i. Either 1) remove i or 2) reduce i.
                done = False
                Kmap_common_i = kmap_tensor[i] * kbin_j
                difference_tensor = Kmap_common_i != kmap_tensor[j]
                num_diff_bits = np.sum(difference_tensor)

                if num_diff_bits == 0:  # completely subsumes--> remove i.
                  mask[i] = 0
                else:  # turn off the different bit.
                  dim_ind = np.nonzero(difference_tensor)[0]
                  kmap_tensor[i][dim_ind] = 0

      if done:
        break

    if mask.sum() < numAND:
      kmap_tensor = kmap_tensor[mask.nonzero()[0],:]
      #kmap_tensor = kmap_tensor.index_select(0,mask.nonzero().view(-1))

    return kmap_tensor

  ### Util
  def _get_index_from_pool(self, input_ids):
    assert False, 'get_index_from_pool should no longer be used.'
    if input_ids.ndim == 1:
      input_ids = np.expand_dims(input_ids, -1)
    assert self.pool_to_index is not None
    return graph_utils.get_index_from_pool(
      pool_ids=input_ids,
      pool_to_index=self.pool_to_index,
    )

  @property
  def environment_id(self) -> str:
    return self._environment_id

  @property
  def pool_to_index(self) -> np.ndarray:
    assert False, 'pool_to_index should no longer be used.'
    return self._pool_to_index

  @property
  def index_to_pool(self) -> np.ndarray:
    assert False, 'index_to_pool should no longer be used.'
    return self._index_to_pool

  ####
  def save(self, filename):
    data = (
      self._completion_buffer,
      self._eligibility_buffer,
      self.reward_sum,
      self.comp_count,
      self._step_count,
    )
    np.save(filename, data)

  def load(self, filename):
    self._completion_buffer, \
    self._eligibility_buffer, \
    self.reward_sum, \
    self.comp_count, \
    self._step_count = np.load(filename, allow_pickle=True)
    assert self.n_literal_options == self.comp_count.shape[-1]
