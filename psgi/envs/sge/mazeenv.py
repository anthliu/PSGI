from typing import Optional, Any, Dict

import os
import time
import numpy as np

from acme import types, specs

from psgi.envs import sge
from psgi.utils.graph_utils import SubtaskGraph
from psgi.envs.sge.utils import get_id_from_ind_multihot, TimeProfiler, AGENT


def _get_game_length(game_name, graph_param):
  if graph_param is None:
    return 70              # XXX generation mode

  if game_name == 'playground':
    if graph_param[:2] == 'D1':
      return 60
    elif graph_param[:2] == 'D2':
      return 65
    elif graph_param[:2] == 'D3':
      return 70
    elif graph_param[:2] == 'D4':
      return 70
  elif game_name == 'mining':
    return 70
  else:
    assert False


class IneligibleSubtaskError(Exception):
  pass

#################
# Note: There are few differences from msgi repo.
# 1. episode length is shorter in mining
# 2. In msgi, #blocks=0, always. In here, #blocks (and #waters) are non-zero.
# 3. In msgi, #objects == #relevant subtasks. In here, there may be extra objects (e.g., tree in mining)
class MazeEnv:  # single batch
  def __init__(self, *, game_name, graph_param: str,
               rank=0, gamma=0.99, seed=1, render=False, verbose_level=0,
               generate_graphs=False,
               ):
    '''
    Args:
      game_name: 'playground' or 'mining.'
      graph_param: Difficulty of subtask graph. e.g. 'train', 'eval' for mining,
        'D1_train', etc. for playground.
      generate_graphs:
    '''
    if game_name == 'playground':
      from .playground import Playground
      game_config = Playground()
      graph_folder = os.path.join('.', 'data', 'subtask_graph_play')
      filename = f'play_{graph_param}_{seed}'

    elif game_name == 'mining':
      from .mining import Mining
      game_config = Mining()
      graph_folder = os.path.join('.', 'data', 'subtask_graph_mining')
      filename = f'mining_{graph_param}_{seed}'

    else:
      raise NotImplementedError(game_name)

    self._game_name = game_name

    self.config = game_config
    self.subtask_pool = self.config.subtasks
    self._default_game_len = _get_game_length(game_name, graph_param)
    self._rank = rank
    self._verbose_level = verbose_level

    # graph & map
    if not generate_graphs:
      # just load all graph from pregenerated envs
      self.graph = sge.graph.SubtaskGraphCache(
          graph_folder, filename, self.config.nb_subtask_type)
    else:
      # generate on the fly
      self.graph = sge.graph.SubtaskGraphGenerator(
        game_config=game_config, env_name=game_name,
      )

    self.map = sge.Mazemap(rank=rank, game_name=game_name,
                           game_config=game_config, render=render)
    self.gamma = gamma
    self.count = 0

    # init
    self.step_reward = 0.0
    self.tpf = TimeProfiler('Environment/')

    # Store for observation specs.
    self.reset_task(task_index=None)  # dummy random task
    observation, _ = self.reset()
    self._observation_specs = {
        k: specs.Array(shape=v.shape, dtype=v.dtype, name=k) \
        for k, v in observation.items()
    }

  @property
  def max_task(self):
      return self.graph.max_task

  @property
  def environment_id(self) -> str:
    return self._game_name

  def reset_task(self, task_index: Optional[int] = None, **kwargs):
    """Reset the environment task/graph."""

    # Load the task from the suite or generate it on the fly.
    self.graph.reset_graph(graph_index=task_index, **kwargs)

    self.task_index = task_index
    self.num_subtasks = len(self.graph.subtask_id_list)
    self.subtask_reward = self.graph.subtask_reward
    self.subtask_id_list = self.graph.subtask_id_list
    self.game_length = self._default_game_len
    #self.game_length = int(np.random.uniform(0.8, 1.2) * self._default_game_len)

    # Reset map (96% of time)
    self.map.reset(subtask_id_list=self.subtask_id_list, reset_map=True)
    return self.task

  def set_task(self, task: dict):
    """Set the current environment task/graph to the new task."""
    self.graph = task['graph']
    self.game_length = task['game_length']

    # Setup task parameters.
    self.num_subtasks = len(self.graph.subtask_id_list)
    self.subtask_reward = self.graph.subtask_reward
    self.subtask_id_list = self.graph.subtask_id_list  # task-specific subtask ids

    # Setup map parameters.
    self.map.set_map(
        subtask_id_list=self.subtask_id_list,
        map_parameters=task['map_parameters']
    )

  def load_debugging_map(self, filename: str):
    data = np.load(filename, allow_pickle=True)
    [observation, omask, item_map, object_list, agent_x, agent_y] = data

    # msgi repo does not include (agent, block, water) while this repo includes these three.
    observation = np.pad(observation, ((3, 0), (0, 0), (0, 0)), 'constant')
    observation[AGENT][agent_x][agent_y] = True
    item_map = item_map + 3
    # msgi repo does not include (agent, block, water) while this repo includes these three.

    map_parameters = dict(
      init_object_list=object_list,
      init_item_map=item_map,
      init_obs=observation,
      init_pos_x=agent_x,
      init_pos_y=agent_y,
      distance=None,
    )
    self.map.set_map(
        subtask_id_list=self.subtask_id_list,
        map_parameters=map_parameters
    )

  def reset(self):
    # Reset subtask status (1.5%)
    self._dead = False
    self.executed_sub_ind = -1
    self.mask = np.ones(self.num_subtasks, dtype=np.uint8)
    self.mask_id = np.zeros(self.max_task, dtype=np.uint8)
    for ind, pool_idx in enumerate(self.graph.index_to_pool):
      self.mask_id[pool_idx] = 1
    self.completion = np.zeros(self.num_subtasks, dtype=np.int8)
    self.comp_id = np.zeros(self.max_task, dtype=np.int8)
    self._compute_elig()
    self.step_count = 0
    self.action_mask = False  # XXX (prev) action is not valid when env resets

    # Reset map (96% of time)
    self.map.reset(subtask_id_list=self.subtask_id_list, reset_map=False)

    # TODO: check
    #self.map.render(task=self.graph.graph_index, step=self.step_count, epi=self.count)
    self.count += 1

    # Initialize episodic statistics
    self.ret = 0.
    return self._get_state(np.array(False)), self._get_info()

  def step(self, action):
    if self.graph.graph_index is None:
      raise RuntimeError('Environment has never been reset()')
    sub_id = -1
    if self.is_done():
      raise ValueError(
          'Environment has already been terminated. need to be reset!')
    oid = self.map.act(action)
    if (action, oid) in self.config.subtask_param_to_id:  # if (action, item) is one of the subtasks
      sid = self.config.subtask_param_to_id[(action, oid)]
      if sid in self.subtask_id_list:  # if sub_id is in the subtask graph
        sub_id = sid
      else:
        #print('Warning! Executed a non-existing subtask')
        pass
    #
    try:
      reward = self._act_subtask(sub_id)
      self.step_count += 1
      self.ret += reward * (self.gamma ** self.step_count)

      return self._get_state(), reward, self.is_done(), self._get_info(reward, 1)
    except IneligibleSubtaskError:
      self._dead = True
      penalty = -1000   # TODO: Decide what happens on violation
      return self._get_state(), penalty, self.is_done(), self._get_info(0, 1)

  def is_done(self):
    if self._dead:
      return True
    time_over = self.step_count >= self.game_length
    game_over = (self.eligibility * self.mask).sum().item() == 0
    return time_over or game_over

  def discount_spec(self):
    return specs.Array(shape=(), dtype=np.float32, name="discount")

  @property
  def state_spec(self):
    return [
        {'dtype': self.map.get_obs().dtype, 'name': 'observation',
         'shape': self.map.get_obs().shape},
        {'dtype': self.mask_id.dtype, 'name': 'mask',
          'shape': self.mask_id.shape},
        {'dtype': self.comp_id.dtype, 'name': 'completion',
          'shape': self.comp_id.shape},
        {'dtype': self.elig_id.dtype, 'name': 'eligibility',
          'shape': self.elig_id.shape},
        {'dtype': int, 'name': 'step', 'shape': ()}
    ]

  @property
  def action_spec(self) -> types.NestedSpec:
    return specs.Array(shape=len(self.legal_actions), dtype=object)

  @property
  def legal_actions(self):
    return self.config.legal_actions

  # internal
  def _get_state(self, option_success=None):
    step = self.game_length - self.step_count + 1
    step_done = self.step_count >= self.game_length
    state_dict = {
        'observation': self.map.get_obs(),
        'mask': self.mask_id.astype(np.float32),
        'completion': self.comp_id.astype(np.float32),
        'eligibility': self.elig_id.astype(np.float32),
        'remaining_steps': np.array(step, dtype=np.int32),
        'action_mask': np.array(self.action_mask, dtype=np.float32),
        'termination': np.array(self.is_done(), dtype=np.bool),
        'step_done': np.array(step_done, dtype=np.float32),
        'option_success': np.array(False),
    }
    if option_success is not None:
      state_dict.update(
          {'option_success': option_success}
      )
    return state_dict

  def _get_info(self, reward=None, steps=0):
    return {
        # XXX this makes env slow.
        #'graph': self.graph,
        # For SMDP
        'step_count': steps,
        'discount': self.gamma**steps,
        'raw_reward': reward,
        # episodic
        'return': self.ret,
    }

  def _act_subtask(self, sub_id):
    self.executed_sub_ind = -1
    reward = self.step_reward
    if sub_id < 0:
      return reward
    sub_ind = self.graph.pool_to_index[sub_id]
    if self.eligibility[sub_ind] == 1 and self.mask[sub_ind] == 1:
      self.completion[sub_ind] = 1
      self.comp_id[sub_id] = 1
      reward += self.subtask_reward[sub_ind]
      self.executed_sub_ind = sub_ind
    else:
      raise IneligibleSubtaskError()
      #assert False, 'The selected subtask is not avilable!'
    self.mask[sub_ind] = 0
    self.mask_id[sub_id] = 0

    self._compute_elig()
    return reward

  def _compute_elig(self):
    self.eligibility = self.graph.get_elig(self.completion)
    self.elig_id = get_id_from_ind_multihot(
        self.eligibility, self.graph.index_to_pool, self.max_task)

  @property
  def task(self) -> dict:
    return dict(
        graph=self.graph,  # TODO: Replace this with SubtaskGraph
        map_parameters=self.map.parameters,
        game_length=self.game_length,
        seed=self.task_index,
    )

  @property
  def task_embedding(self) -> SubtaskGraph:
    # Light weight subtask graph class.
    return SubtaskGraph(
        numP=self.graph.numP,
        numA=self.graph.numA,
        index_to_pool=self.graph.index_to_pool,
        pool_to_index=self.graph.pool_to_index,
        subtask_reward=self.graph.subtask_reward,
        W_a=self.graph.W_a,
        W_o=self.graph.W_o,
        ORmat=self.graph.ORmat,
        ANDmat=self.graph.ANDmat,
        tind_by_layer=self.graph.tind_by_layer
    )

  @property
  def pool_to_index(self):
    return self.graph.pool_to_index

  @property
  def index_to_pool(self):
    return self.graph.index_to_pool


class MazeOptionEnv(MazeEnv):  # single batch. option. single episode
  # TODO: Do not use inheritance, but implement as wrapper (composition).
  # It does not make a sense (you are not overriding "env.step")
  def __init__(self, game_name: str, *, graph_param: str,
               gamma=0.99, rank=None, seed=1, render=False, verbose_level=0,
               generate_graphs=False):
    '''
    Args:
      game_name: 'mining' or 'playground'.
      graph_param: the difficulty of the environment. 'train', 'eval', or 'D1_train', etc.
    '''
    super().__init__(
      game_name=game_name,
      graph_param=graph_param,
      gamma=gamma,
      rank=rank,
      seed=seed,
      render=render,
      verbose_level=verbose_level,
      generate_graphs=generate_graphs,
    )

  def step(self, action: np.ndarray):
    assert self.graph.graph_index is not None, 'Environment need to be reset'
    assert not self.is_done(), 'Environment has already been terminated.'
    if action not in self.index_to_pool:
      assert False, f"action : {action} NOT in {self.index_to_pool}"
    assert action in self.subtask_id_list, 'Option %s does not exist!'%(action)
    sub_id = int(action)

    # TODO: check action and update action_mask.
    self.action_mask = True
    # teleport to target obj pos
    interact, oid = self.config.subtask_param_list[sub_id]
    nav_step, option_success = self.map.move_to_closest_obj(oid=oid, max_step=self.game_length - self.step_count - 1)
    option_step = nav_step + 1

    if option_success: # enough time.
      assert self.step_count + option_step <= self.game_length
      # interact with target obj in the map
      oid_ = self.map.act(interact)
      assert oid == oid_

      # process subtask
      reward = self._act_subtask(sub_id) # update comp, elig, mask.
      self.step_count += option_step
      self.ret += reward
    else: # not enough time. time over
      self.step_count = self.game_length
      reward = 0.
    #self.map.render(self.step_count)
    if self._verbose_level > 1:
      print(f'[{self.step_count}/{self.game_length}], pool_id={sub_id}  -> agent=({self.map.agent_x}, {self.map.agent_y}), #steps={option_step} reward={reward:.3f}')

    info = None
    return self._get_state(option_success), reward, self.is_done(), info

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_specs

  @property
  def legal_actions(self):
    return list(self.graph.index_to_pool)

  def action_spec(self) -> types.NestedSpec:
    # TODO: Decide continuous discrete space (may contain nonexistent subtasks) or not.
    return specs.DiscreteArray(
        num_values=self.max_task,
        dtype=np.int32
    )

  def reward_spec(self) -> types.NestedSpec:
    # TODO: Support batch reward shape. Currently batch shape will cause an error
    # in trfl.vtrace_from_importance_weights.
    return specs.Array(shape=(), dtype=np.float32, name='reward')

  @property
  def num_tasks(self):
    return self.graph.num_graph
