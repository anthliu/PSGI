"""Baselines Actor implementations."""

from typing import Dict, List
import numpy as np

import dm_env
import acme
from acme import specs, types

from psgi.agents import base
from psgi.graph.grprop import GRProp
from psgi.graph.option_grprop import CycleGRProp
from psgi.utils import graph_utils
from psgi.utils.graph_utils import SubtaskGraph

class GRPropActor(base.BaseActor):
  """An Actor that chooses random action."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temp: float = None,
      w_a: float = None,
      beta_a: float = None,
      ep_or: float = None,
      temp_or: float = None,
  ):
    self._observation_spec: types.NestedArray = environment_spec.observations
    self._action_spec: specs.DiscreteArray = environment_spec.actions
    self._grprop = GRProp(
        environment_spec=environment_spec,
        temp=temp,
        w_a=w_a, beta_a=beta_a,
        ep_or=ep_or, temp_or=temp_or
    )
    self._index_to_pool = None
    self._pool_to_index = None
    self._num_data = None

  @property
  def is_ready(self):
    return self._grprop.is_ready

  def observe_task(self, tasks: List[SubtaskGraph]):
    self._grprop.init_graph(graphs=tasks)
    self._index_to_pool = self._grprop._index_to_pool
    self._pool_to_index = np.stack([task.pool_to_index for task in tasks])
    self._num_data = np.stack([task.num_data for task in tasks])

  @property
  def confidence_score(self):
    return self._num_data

  def observe_first(self, timestep: dm_env.TimeStep):
    pass  # Do nothing.

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    pass  # Do nothing.

  def update(self):
    pass  # Do nothing.

  def _get_raw_logits_indexed_debug(self, indexed_obs: Dict[str, np.ndarray]) -> np.ndarray:
    # Run GRProp for test phase.
    logits_indexed = self._grprop.get_raw_logits(
        observation=indexed_obs
    )
    return logits_indexed

  def get_availability(self, observation: Dict[str, np.ndarray]) -> bool:
    if not self.is_ready:
      return np.full(shape=(observation['mask'].shape[0]), fill_value=False, dtype=np.bool)
    indexed_obs = graph_utils.transform_obs(
        observation=observation,
        index_to_pool=self._index_to_pool,
    )
    mask = indexed_obs['mask']
    eligibility = indexed_obs['eligibility']
    masked_eligibility = np.multiply(mask, eligibility)
    return masked_eligibility.sum(-1) > 0.5

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    observation['mask'] = observation['mask'] * (1 - observation['completion']) # This is assumed/required by grprop algorithm
    indexed_obs = graph_utils.transform_obs(
        observation=observation,
        index_to_pool=self._index_to_pool,
    )

    # Run GRProp for test phase.
    logits_indexed = self._grprop.get_raw_logits(
        observation=indexed_obs
    )

    # index_to_pool
    logits = graph_utils.map_index_arr_to_pool_arr(
        logits_indexed,
        pool_to_index=self._pool_to_index,
        default_val=0.
    )
    assert self._action_spec.num_values == logits.shape[1], str(logits.shape)
    return logits * observation['mask']

class CycleGRPropActor(GRPropActor):
  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temperature: float = 10.0,
      w_p: float = 0.6,
      verbose_level: int = 0,
  ):
    self._observation_spec: types.NestedArray = environment_spec.observations
    self._action_spec: specs.DiscreteArray = environment_spec.actions
    self._grprop = CycleGRProp(
        environment_spec=environment_spec,
        temperature=temperature,
        w_p=w_p,
    )
    self._verbose_level = verbose_level
    self._index_to_pool = None
    self._pool_to_index = None
    self._num_data = None

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    # Run GRProp for test phase.
    logits_feasible_option = self._grprop.get_raw_logits(
        observation=observation,
    )
    """if True: # XXX: debugging
      for ind, feasible_option_name in enumerate(self.feasible_option_name_list):
        logit = logits_feasible_option[ind]
        
        print('logits=')
        for option_index, name in enumerate(feasible_option_name):
          print(f'{name}: {logit[option_index]}')
    import ipdb; ipdb.set_trace()"""
    
    logits = graph_utils.map_index_arr_to_pool_arr(
        logits_feasible_option,
        pool_to_index=self._all_to_feasible,
        default_val=0. # give default point to infeasible option. 
        # Note: infeasible option should NOT be masked out, since it's possible that only infeasible options are
    )
    #self._debug_print_logits(logits, observation['mask'] * observation['eligibility'])
    
    assert self._action_spec.num_values == logits.shape[1], str(logits.shape)
    return logits * observation['mask']  

  def _debug_print_logits(self, logits, masked_eligibility):
    print('=== GRprop logits ===')
    for i, logit in enumerate(logits):
      mask = masked_eligibility[i]
      """
      goal_index = self.debug_option_name_list[i].index('(op_cook, pork, pan)')
      goal_index2 = self.debug_option_name_list[i].index('(op_cook, pork, pot)')
      if mask[goal_index] == 1:
        import ipdb; ipdb.set_trace()
      if mask[goal_index2] == 1:
        import ipdb; ipdb.set_trace()"""
      logit[logit == -np.infty] = 0.
      logit[mask==0] = 0.
      option_indices = logit.nonzero()[0]
      for index in option_indices:
        print(f"{self.debug_option_name_list[i][index]}: logit = {logit[index]}")

  def observe_task(self, tasks: List[SubtaskGraph]):
    self._grprop.init_graph(graphs=tasks)

    self._all_to_feasible = self._grprop._all_to_feasible
    self._feasible_to_all = self._grprop._feasible_to_all

    self.debug_feasible_option_name_list = [[node.name for node in graph.option_nodes] for graph in tasks] # for debugging
    self.debug_option_name_list = [graph.all_option_names for graph in tasks] # for debugging
    #self._index_to_pool = self._grprop._index_to_pool
    #self._pool_to_index = np.stack([task.pool_to_index for task in tasks])
    #self._num_data = np.stack([task.num_data for task in tasks])
