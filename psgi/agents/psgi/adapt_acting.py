"""Baselines Actor implementations."""

from typing import Dict, List, Sequence
from collections import defaultdict

import abc
import numpy as np

import dm_env
import acme
from acme import specs, types
from acme.tf import utils as tf2_utils

from psgi.utils import graph_utils, tf_utils
from psgi.agents.base import UCBActor, CountBasedActor
from psgi.utils.graph_utils import SubtaskGraph
from psgi.graph.option_grprop import CycleGRProp

class SubtaskCountActor(CountBasedActor):
  """An Actor that chooses an action using UCB."""
  def observe_task(self, tasks: List[SubtaskGraph]):
    self.task_reset_flag = True
    # Debugging
    self.debug_graphs = tasks
    self.debug_feasible_option_name_list = [[node.name for node in task.graph.option_nodes] for task in tasks] # for debugging
    self.debug_option_name_list = [task.graph.all_option_names for task in tasks] # for debugging
    self.debug_subtask_name_list = [[node.name for node in task.graph.subtask_nodes] for task in tasks] # for debugging

  @property
  def num_subtask_completed(self):
    return (self._comp_counts>0).sum()
  @property
  def num_option_eligible(self):
    return (self._elig_counts>0).sum()
  @property
  def num_option_executed(self):
    return (self._option_counts>0).sum()

  def _hash_completion(self, completion):
    comp_hash = graph_utils.batch_bin_encode(completion > 0.5) # [(0, 1, 13), ] * batch_size
    for i in range(self.batch_size):
      self.comp_hash_set[i].add(comp_hash[i])

  def observe_first(
      self,
      timestep: dm_env.TimeStep,
  ): # called in the very first step after resetting task
    self._bonus_weight = 1.0 # XXX: move to init and make it argument
    if self.task_reset_flag:
      # option-based
      eligibility = timestep.observation['eligibility']
      num_batch, num_option = eligibility.shape
      self.batch_size = num_batch
      self._option_reward_sum = np.zeros_like(eligibility) # this is not perfect. we should use subtask reward.
      self._option_counts = np.zeros_like(eligibility)
      self._elig_counts = eligibility.copy() # consider the first elig
      #self._total_elig_counts = np.full(shape=(num_batch), fill_value=num_option)

      # subtask-based
      completion = timestep.observation['completion']
      self.comp_hash_set = [set() for _ in range(self.batch_size)]
      self._hash_completion(completion)
      num_batch, num_subtask = completion.shape
      self._subtask_reward_sum = np.zeros_like(completion)
      self._comp_counts = completion.copy() # only for measuring agent's exploration
      self._delta_comp_counts = np.zeros_like(completion)
      #self._total_comp_counts = np.full(shape=(num_batch), fill_value=num_subtask)
      #
      #self.effect_subtask_index = np.full_like(eligibility, fill_value=-1, dtype=np.int) # only subtask index of single positive effect
      self.effect_matrices = np.zeros(shape=(num_batch, num_option, num_subtask), dtype=np.int) # store entire effect
      self.prev_completion = completion

      # else
      self.task_reset_flag = False

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    next_rewards = next_timestep.reward
    is_first = next_timestep.first()
    completion = next_timestep.observation['completion']
    eligibility = next_timestep.observation['eligibility']
    delta_comps = (completion - self.prev_completion)
    positive_delta_comps = delta_comps > 0

    # Update
    self._hash_completion(completion)
    self._comp_counts += completion
    self._elig_counts += eligibility

    # Update remaining
    for i, positive_delta_comp in enumerate(positive_delta_comps):
      delta_comp = delta_comps[i]
      delta_subtask_indices = delta_comp.nonzero()[0]
      pos_delta_subtask_indices = positive_delta_comp.nonzero()[0]
      option = action[i]
      # option
      if not is_first[i]:
        self._option_reward_sum[i][option] += next_rewards[i]
        self._option_counts[i][option] += 1
        
      # subtask reward & single positive effect
      if not is_first[i] and len(pos_delta_subtask_indices) > 0:
        #assert len(pos_delta_subtask_indices) == 1, "Error! Does not support options with more than two positive effects"
        # XXX Estimate option reward with more than two positive effects by even split
        '''
        # Old code
        subtask_index = pos_delta_subtask_indices[0]
        self._subtask_reward_sum[i][subtask_index] += next_rewards[i]
        self._delta_comp_counts[i][subtask_index] += 1
        self.effect_subtask_index[i][option] = subtask_index
        '''
        N_effects = len(pos_delta_subtask_indices)
        for subtask_index in pos_delta_subtask_indices:
          self._subtask_reward_sum[i][subtask_index] += next_rewards[i] / N_effects
          self._delta_comp_counts[i][subtask_index] += 1
      
      # full effect matrix
      for subtask_index in delta_subtask_indices:
        if self.effect_matrices[i, option, subtask_index] != 0:
          assert self.effect_matrices[i, option, subtask_index] == delta_comp[subtask_index], "Error: effect has changed!"
        self.effect_matrices[i, option, subtask_index] = delta_comp[subtask_index]

    # Update prev_completion
    self.prev_completion = completion
    

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['eligibility'].shape[1] == observation['mask'].shape[1]

    # C0: never_delta_comp=T : discover reward! (only in transfer. most important)
    # C1: delta_comp=T : help reward inference & maybe precond in the next step.
    # C2:   & next_comp_unique=T: help precondition in the next step.
    # C3: opt_never_exec=T : help effect & maybe precond in the next step. TIP: bonus = 1 + \propto 1/#ever_elig makes it prioritize executing option that is rarely eligible
    # C4: opt_never_exec=T & delta_comp=T : verify effect (only possible when effect is transferred)

    # Possible cases in priority (no transfer)
    # C3: opt_never_exec=T: help effect & reward & maybe precond in the next step. propto 1/#elig_count. bonus=3~4
    # C2: (delta_comp=T &) next_comp_unique=T  : help reward inference & precond in the next step. bonus=2
    # C1: delta_comp=T: help reward inference & "maybe" precond in the next step. bonus=1
    # else (i.e., delta_comp=F, opt_never_exec=F): nothing to gain. bonus=0
    masked_eligibilities = observation['mask'] * observation['eligibility']
    logits = np.full_like(observation['eligibility'], fill_value=-np.infty)
    for i in range(self.batch_size):
      if observation['termination'][i]:
        continue
      elig_options = masked_eligibilities[i].nonzero()[0]
      logits[i][elig_options] = 0 # starts from 0
      #
      completion = observation['completion'][i]
      elig_count = self._elig_counts[i][elig_options]
      option_count = self._option_counts[i][elig_options]
      effect_mat = self.effect_matrices[i]
      # C3: opt_never_exec=T
      bonus1 = np.zeros_like(elig_count)
      option_never_exec = option_count < 1
      if option_never_exec.sum() > 0: # if exists
        option_elig_count = elig_count[option_never_exec]
        min_count, max_count = option_elig_count.min(), option_elig_count.max()
        if option_never_exec.sum() == 1 or min_count == max_count:
          rarity_bonus = 0.5
        else:
          # count = min_count ~ max_count --> bonus = 1 ~ 0
          rarity_bonus = 1.  - (option_elig_count - min_count) / (max_count - min_count)
          assert rarity_bonus.min()==0 and rarity_bonus.max()==1, "bonus error!"
        # 
        bonus1[option_never_exec] = 3.0 + 1.0 * rarity_bonus
      
      # C2: (delta_comp=T &) next_comp_unique=T : bonus = 2
      # C1: delta_comp=T : bonus = 1
      # If effect is None: bonus = 1
      bonus2 = []
      for ind, option in enumerate(elig_options):
        if np.any(effect_mat[option] != 0): # at least one effect is discovered
          delta_comp = effect_mat[option]
          next_comp = np.clip(completion + delta_comp, 0, 1)
          hash_code = graph_utils.batch_bin_encode(completion > 0.5)
          is_next_comp_unique = hash_code in self.comp_hash_set[i]
          is_comp_changed = np.any(next_comp != completion)
          bonus2.append(is_next_comp_unique + is_comp_changed)
        else: # no effect -> under our assumption, this means option never executed. so should be covered in bonus1
          if bonus1[ind] > 3.0:
            bonus2.append(0.)
          else:
            bonus2.append(1.)
      bonus2 = np.array(bonus2)
      logits[i][elig_options] = bonus1 + bonus2
    return logits * self._temperature

class CountGRPropActor(SubtaskCountActor):
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      grprop: CycleGRProp,
      temperature: float = 10.0,
      verbose_level: int = 0,
  ):
    super().__init__(
      environment_spec=environment_spec,
      temperature=temperature,
      verbose_level=verbose_level,
    )
    self._grprop = grprop

  def observe_task(self, tasks: List[SubtaskGraph]):
    super().observe_task(tasks)
    self._graph_initialized = False

  def _get_logits_from_grprop(self, grprop, observation):
    masked_eligibility = observation['mask'] * observation['eligibility']
    self._all_to_feasible = grprop._all_to_feasible
    self._feasible_to_all = grprop._feasible_to_all
    self.debug_feasible_option_name_list = grprop.debug_feasible_option_name_list

    # Compute subtask bonus
    subtask_bonus = np.copy(self._avg_rewards)
    feasible_option_bonus = np.zeros_like(self._feasible_to_all)
    option_extra_bonus = np.full_like(observation['eligibility'], fill_value=-np.infty)
    for i, masked_elig in enumerate(masked_eligibility):
      all_to_feasible = self._all_to_feasible[i]
      feasible_to_all = self._feasible_to_all[i]
      if not observation['termination'][i]:
        options = masked_elig.nonzero()[0]
        is_infeasible = all_to_feasible == -1

        # infeasible -> extra & 3.0 (added to logits later)
        option_extra_bonus[i][is_infeasible] = 3.0

        # feasible + never executed -> +1 for option_bonus (going into grprop)
        feasible_option_count = self._option_counts[i][feasible_to_all]
        feasible_never_executed = feasible_option_count <= 1 # 1 means just got increased in current step.
        feasible_option_bonus[i][feasible_never_executed] = 1.0

        # feasible + never eligible -> +2 for option_bonus (going into grprop)
        feasible_elig_count = self._elig_counts[i][feasible_to_all]
        feasible_never_eligible = feasible_elig_count <= 1 # 1 means just got increased in current step.
        feasible_option_bonus[i][feasible_never_eligible] = 2.0
        print(feasible_never_eligible)
        """
        print('never eligible=')
        np_option_names = np.array(self.debug_feasible_option_name_list[i])
        print(np_option_names[feasible_never_eligible])
    print(feasible_option_bonus)
    import ipdb; ipdb.set_trace()"""
    print('bonus')
    print(feasible_option_bonus)
    
    # run GRProp
    logits_feasible_option = grprop.get_raw_logits(
      observation=observation, 
      subtask_reward_list=subtask_bonus, 
      option_reward_list=feasible_option_bonus, 
      temperature=1.0
    )
    logits = graph_utils.map_index_arr_to_pool_arr(
        logits_feasible_option,
        pool_to_index=self._all_to_feasible,
        default_val=-np.infty # Mask out infeasible options
    )
    logits = np.maximum(logits, option_extra_bonus)
    #self._debug_print_logits(logits, masked_eligibility)
    return logits
  

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['eligibility'].shape[1] == observation['mask'].shape[1]
    if self._grprop._graph_initialized:
      logits = self._get_logits_from_grprop(self._grprop, observation)
    else:
      logits = self._get_logits_from_count(observation)
    assert self._action_spec.num_values == logits.shape[1], str(logits.shape)
    return logits * self._temperature

  def _debug_UCB_bonus(self, option_extra_bonus, feasible_option_bonus):
    # option_extra_bonus: (1, 400)
    # feasible_option_bonus: (1, 152)
    print('=== UCB bonus ===')
    batch_size = option_extra_bonus.shape[0]
    for i in range(batch_size):
      option_names = self.debug_option_name_list[i]
      feas_option_names = self.debug_feasible_option_name_list[i]
      extra_bonus = option_extra_bonus[i]
      feasible_bonus = feasible_option_bonus[i]
      #
      indices = (extra_bonus != -np.infty).nonzero()[0]
      """
      for index in indices:
        print(f"{option_names[index]}: infeasible. bonus={extra_bonus[index]}")
        if 'cook' in option_names[index]:
          import ipdb; ipdb.set_trace()"""
      #
      indices = (feasible_bonus != 0.).nonzero()[0]
      for index in indices:
        print(f"{feas_option_names[index]}: bonus={feasible_bonus[index]}")

  def _debug_print_logits(self, logits, masked_eligibility):
    print('=== GRprop logits ===')
    for i, logit in enumerate(logits):
      mask = masked_eligibility[i]
      logit[logit == -np.infty] = 0.
      logit[mask==0] = 0.
      option_indices = logit.nonzero()[0]
      for index in option_indices:
        print(f"{self.debug_option_name_list[i][index]}: logit = {logit[index]}")

class MTCountGRPropActor(CountGRPropActor):
  """An Multi-Task UCB actor."""
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      prior_grprop: CycleGRProp,
      grprop: CycleGRProp,
      temperature: float = 10.0,
      verbose_level: int = 0,
  ):
    super().__init__(
      environment_spec=environment_spec,
      grprop=grprop,
      temperature=temperature,
      verbose_level=verbose_level,
    )
    self._prior_grprop = prior_grprop
  
  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['eligibility'].shape[1] == observation['mask'].shape[1]
    if self._grprop._graph_initialized:
      use_prior = True
    else:
      use_prior = True
    
    if use_prior:
      logits = self._get_logits_from_grprop(self._prior_grprop, observation)
    else:
      logits = self._get_logits_from_grprop(self._grprop, observation)
      
    assert self._action_spec.num_values == logits.shape[1], str(logits.shape)
    return logits * self._temperature
