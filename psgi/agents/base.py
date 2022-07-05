"""Baselines Actor implementations."""

from typing import Dict, List, Sequence, Optional
from collections import defaultdict

import abc
import numpy as np

import dm_env
import acme
from acme import specs, types
from acme.tf import utils as tf2_utils
from acme.utils import loggers

from psgi import agents
from psgi.agents import base
from psgi.agents import meta_agent
from psgi.utils import graph_utils, tf_utils
from psgi.utils.graph_utils import SubtaskGraph


class BaseActor(acme.Actor):
  """Actor Base class
  """
  __metaclass__ = abc.ABC

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      verbose_level: int = 0,
  ):
    self._observation_spec: types.NestedArray = environment_spec.observations
    self._action_spec: specs.DiscreteArray = environment_spec.actions
    self._verbose_level = verbose_level

  @property
  def num_option_executed(self):
    return (self._option_counts>0).sum()

  def observe_task(self, tasks: List[SubtaskGraph]):
    pass # Do nothing upon starting a new task

  def observe_first(self, timestep: dm_env.TimeStep):
    eligibility = timestep.observation['eligibility']
    self._option_counts = np.zeros_like(eligibility)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    for i, option in enumerate(action):
      self._option_counts[i][option] += 1

  def update(self):
    pass  # Do nothing.

  def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    masked_eligibility = np.multiply(observation['mask'], observation['eligibility'])

    # Choose from stochastic policy
    termination = observation['termination']
    logits = self._get_raw_logits(observation)
    assert not np.any(np.isnan(logits)), 'Error! Nan in logits'

    # When the environment is done, replace with uniform policy. This action
    # will be ignored by the environment
    if np.any(termination):
      masked_eligibility[termination, :] = 1
      logits[termination, :] = 1.

    # Masking
    probs = self._mask_softmax(logits, masked_eligibility)
    actions = tf_utils.categorical_sampling(probs=probs).numpy()
    """
    if self._verbose_level > 0:
      print(f'actions (pool)={actions[0].item()}')
      print('probs (pool)=', probs[0])
    if self._verbose_level > 1:
      print('masked_eligibility (pool)= ', masked_eligibility[0])"""

    return actions

  @abc.abstractmethod
  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    raise NotImplementedError

  def _mask_logits(self, logits: np.array, masked_eligibility: np.array) -> np.ndarray:
    # Numerically stable masking
    # Ex) If Mask = [1, 0, 1]. logits=[1, 0, -5]->[6, 5, 0]->[6, -infty, 0]->[0, -infty, -6]
    # Ex) If Mask = [1, 1, 0]. logits=[1, 0, -5]->[6, 5, 0]->[6, 5, -infty]->[0, -1, -infty]
    if logits.min() > -np.infty:
      out_logits = logits - logits.min(axis=-1, keepdims=True)
    else:
      out_logits = logits.copy()
    out_logits[masked_eligibility == 0] = -np.infty
    if logits.max() > -np.infty:
      out_logits -= out_logits.max(axis=-1, keepdims=True)
    else:
      import ipdb; ipdb.set_trace()
      out_logits.fill(0)
    assert not np.any(np.isnan(out_logits)), 'Error! Nan in logits'
    return out_logits

  def _mask_softmax(self, logits: np.array, masked_eligibility: np.ndarray) -> np.ndarray:
    # Numerically stable masked-softmax
    masked_logits = self._mask_logits(logits, masked_eligibility)
    exps = np.exp(masked_logits)
    probs_masked = exps / exps.sum(axis=-1, keepdims=True)

    # Make sure by masking probs once more
    probs_masked = probs_masked * masked_eligibility
    probs_masked = probs_masked / probs_masked.sum(axis=-1, keepdims=True)
    return probs_masked


class FixedActor(BaseActor):
  """An Actor that chooses a predefined set of actions."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      option_sequence: Sequence[np.ndarray],
      verbose_level: int = 0,
  ):
    super().__init__(environment_spec=environment_spec, verbose_level=verbose_level)
    self.step_count = 0
    self.option_seq = option_sequence

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    action = self.option_seq[self.step_count]
    self.step_count += 1
    logits = np.zeros_like(observation['mask'])
    logits[:, action] = 1.
    return logits

class RandomActor(BaseActor):
  """An Actor that chooses random action."""

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['mask'].ndim == 2
    if 'eligibility' in observation:
      assert observation['eligibility'].ndim == 2
    logits = np.ones_like(observation['mask'], dtype=np.float)
    return logits

class UCBActor(BaseActor):
  """An Actor that chooses an action using UCB."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temperature: float,
      verbose_level: int = 0,
  ):
    super().__init__(environment_spec=environment_spec, verbose_level=verbose_level)
    self._temperature = temperature

  def observe_task(self, tasks: List[SubtaskGraph]):
    self.debug_graphs = tasks
    self.debug_feasible_option_name_list = [[node.name for node in task.graph.option_nodes] for task in tasks] # for debugging
    self.debug_option_name_list = [task.graph.all_option_names for task in tasks] # for debugging
    self.debug_subtask_name_list = [[node.name for node in task.graph.subtask_nodes] for task in tasks] # for debugging
    num_options = self._action_spec.num_values
    self._avg_rewards = np.zeros((len(tasks), num_options))
    self._counts = np.ones((len(tasks), num_options))
    self._total_counts = np.full(shape=(len(tasks)), fill_value=num_options)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    action = np.expand_dims(action, axis=-1)
    next_rewards = np.expand_dims(next_timestep.reward, axis=-1)
    is_first = np.expand_dims(next_timestep.first(), axis=-1)  # for mask
    #
    avg_rewards = np.take_along_axis(self._avg_rewards, action, axis=-1)
    counts = np.take_along_axis(self._counts, action, axis=-1)

    # Compute & update avg rewards.
    update_values = 1 / counts * (next_rewards - avg_rewards)
    next_avg_rewards = avg_rewards + np.where(is_first, 0, update_values)  # skip first timestep.
    np.put_along_axis(self._avg_rewards, action, values=next_avg_rewards, axis=-1)

    # Update counts.
    np.put_along_axis(self._counts, action, values=counts + (1 - is_first), axis=-1)
    self._total_counts += (1 - is_first).squeeze()

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['eligibility'].shape[1] == observation['mask'].shape[1]
    logits = np.zeros_like(observation['mask'])
    masked_eligibility = observation['mask'] * observation['eligibility']
    for i, masked_elig in enumerate(masked_eligibility):
      if not observation['termination'][i]:
        options = masked_elig.nonzero()[0]
        rewards = self._avg_rewards[i][options]
        counts = self._counts[i][options]
        utility = rewards + np.sqrt(2 * np.log(self._total_counts[i]) / counts)
        logits[i][options] = self._temperature * utility
    return logits

class MTUCBActor(UCBActor):
  """An Multi-Task UCB actor."""

  def observe_task(self, tasks: List[SubtaskGraph]):
    pool_to_index = np.stack([g.pool_to_index for g in tasks], axis=0)
    prior_rewards = np.stack([g.subtask_reward for g in tasks], axis=0)
    prior_counts = np.stack([g.reward_count for g in tasks], axis=0)

    # Use prior to initialize the avg rewards.
    self._avg_rewards = graph_utils.map_index_arr_to_pool_arr(
        arr_by_index=prior_rewards,
        pool_to_index=pool_to_index
    )

    # Initialize counts.
    self._counts = graph_utils.map_index_arr_to_pool_arr(
        arr_by_index=prior_counts,
        pool_to_index=pool_to_index
    )
    self._counts += 1  # avoid nan
    self._total_counts = np.sum(self._counts, axis=-1)


class CountBasedActor(UCBActor):
  """An Actor that chooses an action using simple count-based strategy."""
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temperature: float,
      verbose_level: int = 0,
  ):
    super().__init__(environment_spec=environment_spec, temperature=temperature, verbose_level=verbose_level)

  def observe_task(self, tasks: List[SubtaskGraph]):
    self.task_reset_flag = True
    super().observe_task(tasks)
    
  def observe_first(
      self,
      timestep: dm_env.TimeStep,
  ): # called in the very first step after resetting task
    if self.task_reset_flag:
      completion = timestep.observation['completion']
      eligibility = timestep.observation['eligibility']
      num_batch, num_subtask = completion.shape
      _, num_option = eligibility.shape
      #
      self.batch_size = num_batch
      #
      self._option_reward_sum = np.zeros_like(eligibility)
      self._elig_counts = eligibility.copy()
      self._option_counts = np.ones_like(eligibility)
      #
      self.task_reset_flag = False
  
  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    next_rewards = next_timestep.reward
    is_first = next_timestep.first()
    # Update elig
    self._elig_counts += next_timestep.observation['eligibility']
    
    # Update option_counts and option rewards
    for i in range(self.batch_size):
      option = action[i]
      if not is_first[i]:
        self._option_counts[i][option] += 1
        self._option_reward_sum[i][option] += next_rewards[i]
        #self._total_option_counts[i] += 1

  def _get_logits_from_count(self, observation):
    masked_eligibility = observation['mask'] * observation['eligibility']
    logits = np.full_like(observation['eligibility'], fill_value=-np.infty)
    for i, masked_elig in enumerate(masked_eligibility):
      if not observation['termination'][i]:
        options = masked_elig.nonzero()[0]

        # First add empirical mean reward
        reward_sums = self._option_reward_sum[i][options]
        reward_counts = self._option_counts[i][options]
        reward_means = reward_sums / reward_counts
        logits[i][options] = reward_means
        
        # + 2 for options that were never eligible
        never_eligible = self._elig_counts[i][options] == 1 # 1 means just got increased in current step.
        options_never_eligible = options[never_eligible]
        logits[i][options_never_eligible] += 3.0

        # + 2 for options that were never executed
        never_executed = self._option_counts[i][options] == 1
        options_never_executed = options[never_executed]
        logits[i][options_never_executed] += 3.0

        # XXX: debug
        """
        np_option_names = np.array(self.debug_option_name_list[i])
        if len(options_never_eligible) > 0:
          print('options_never_eligible=', np_option_names[options_never_eligible])
        if len(options_never_executed) > 0:          
          print('options_never_executed=', np_option_names[options_never_executed])"""
    return logits

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert observation['eligibility'].shape[1] == observation['mask'].shape[1]
    logits = self._get_logits_from_count(observation)
    assert self._action_spec.num_values == logits.shape[1], str(logits.shape)
    return logits * self._temperature

class GreedyActor(CountBasedActor):
  """An Actor that chooses random action."""
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temperature: float = 10.,
      verbose_level: int = 0,
  ):
    self.task_reset_flag = True
    super().__init__(environment_spec=environment_spec, temperature=temperature, verbose_level=verbose_level)

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    logits = np.full_like(observation['mask'], fill_value=-np.infty)
    completions = observation['completion'].copy().astype(np.int8)

    masked_eligibility = observation['mask'] * observation['eligibility']
    for i, (comp, masked_elig) in enumerate(zip(completions, masked_eligibility)):
      if not observation['termination'][i]:
        options = masked_elig.nonzero()[0]

        # First add empirical mean reward
        reward_sums = self._option_reward_sum[i][options]
        reward_counts = self._option_counts[i][options]
        reward_means = reward_sums / (reward_counts + 1e-5)
        logits[i][options] = reward_means
        
    return logits * self._temperature

class MetaGreedyActor(meta_agent.MetaAgent):
  """Meta agent for Greedy Actor"""
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      mode: str,
      logger: loggers.Logger = None,
      verbose_level: Optional[int] = 0,
  ):
    self._environment_spec = environment_spec
    self._mode = mode
    self._logger = logger
    self._verbose_level = verbose_level

    self._agent = GreedyActor(
        environment_spec=self._environment_spec,
        verbose_level=self._verbose_level
    )

  def instantiate_adapt_agent(self) -> agents.Agent:
    return self._agent

  def instantiate_test_actor(self) -> base.BaseActor:
    return self._agent

  def reset_agent(self, environment: dm_env.Environment):
    environment_spec = specs.make_environment_spec(environment)
    self._environment_spec = environment_spec
    self._agent = GreedyActor(
        environment_spec=self._environment_spec,
        verbose_level=self._verbose_level
    )

  def update(self):
    # Greedy has no meta-training. Do nothing.
    return

  def update_adaptation_progress(self, current_split: int, max_split: int):
    """Update the adaptation progress with adaptation / testing actors
    """
    pass # do nothing
