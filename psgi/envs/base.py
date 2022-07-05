from typing import Optional, List, Callable, Dict

import abc
import numpy as np

from acme import types, specs

from psgi.utils.graph_utils import SubtaskGraph
from psgi.envs.base_config import WobConfig  # only for typing


class BaseWoBEnv(abc.ABC):
  def __init__(
      self,
      rank: int,
      config_factory: List[Callable[[], WobConfig]],
      keep_pristine: bool,
      verbose_level: Optional[int] = 0,
  ):
    # Set environment configurations
    self.rank = rank

    # Set task-info. (For multi-task env, set this in reset_task())
    self._configs = config_factory
    self._keep_pristine = keep_pristine
    assert all([callable(config) for config in self._configs])

    # Sample at least one valid task config (default seed = 0).
    self.reset_task(task_index=0)

    # params
    self.env_done = False
    self.step_count = 0
    # TODO: Do we need action mask?
    self.action_mask = False

    self._verbose_level = verbose_level

    # Set observation specs.
    observation = self.reset()
    self._observation_specs = {
      k: specs.Array(shape=v.shape, dtype=v.dtype, name=k) \
      for k, v in observation.items()
    }

  def reset_task(self, task_index: Optional[int] = None):
    # TODO: Remove the assertion and support random task sampling when different
    # graph across the batch is supported. (See environment_loop.py L225)
    assert task_index is not None
    config_factory = self._configs[task_index % len(self._configs)]

    self.config: WobConfig = config_factory(seed=task_index,
                                            keep_pristine=self._keep_pristine)

    if 'v2' in self.config.environment_id:
      from psgi.envs import base_config_v2 as bc2
      self._subtask_pool_name_to_id = bc2.SUBTASK_POOL_NAME_TO_ID
      self._option_id_to_name = bc2.OPTION_ID_TO_NAME
      self._option_name_to_id = bc2.OPTION_NAME_TO_ID
    else:
      from psgi.envs import base_config as bc
      self._subtask_pool_name_to_id = bc.SUBTASK_POOL_NAME_TO_ID
      self._option_id_to_name = bc.OPTION_ID_TO_NAME
      self._option_name_to_id = bc.OPTION_NAME_TO_ID

    return self.config

  @property
  def num_total_subtasks(self):
    return len(self._subtask_pool_name_to_id)
  @property
  def action_dim(self):
    # TODO: This might differ from num_total_subtasks.
    return len(self._option_id_to_name)
  @property
  def num_graphs(self):
    return self.config.num_graphs
  @property
  def graph(self):
    return self.config.graph
  @property
  def options(self):
    return self.config.options
  @property
  def subtasks(self):
    return self.config.subtasks
  @property
  def subtask_reward(self):
    return self.config.subtask_reward
  @property
  def num_subtasks(self):
    return len(self.subtasks)
  @property
  def num_options(self):
    return len(self.options)
  @property
  def max_step(self):
    return self.config.max_step
  @property
  def task(self):
    return self.config
  @property
  def pool_to_index(self):
    return self.config._pool_to_index
  @property
  def index_to_pool(self):
    return self.config._index_to_pool

  ### Communicating with the agent (actor) --> should be id-based
  @property
  def task_embedding(self) -> SubtaskGraph:
    reward_vec = np.zeros((self.num_subtasks,), dtype=np.float)
    for i, subtask in enumerate(self.subtasks):
      reward_vec[i] = self.subtask_reward[subtask]

    # TODO: generate ground truth graph information that GRProp can interpret.
    return SubtaskGraph(
        numP=None,
        numA=None,
        index_to_pool=self.index_to_pool,
        pool_to_index=self.pool_to_index,
        subtask_reward=reward_vec,  # index-based
        W_a=None,
        W_o=None,
        ORmat=None,
        ANDmat=None,
        tind_by_layer=None
    )

  ### For connecting adaptation-test envs ~~ ###
  def set_task(self, task):
    """fixed the current task to the given task. used for aligning the task
      between the adaptation & test environments.
    """
    self.config = task

  def reset(self):
    # Reset episode-based internal states
    self.step_count = 0
    # TODO: Do we need action mask?
    self.action_mask = False  # XXX (prev) action is not valid when env resets
    self.done = False
    self.step_done = False
    self.mask = None
    self.completion = None
    self.eligibility = None

    # Reset & update batch_params
    self._update_batch_params(reset=True)
    assert not self.done, "done cannot be True here"

    observation = self._get_observation()
    return observation

  def _update_batch_params(self, action=None, step=None, reset=False):
    '''Reset & Update mask, completion, and eligibility.
    '''
    if reset:
      completion = {name: False for name in self.subtasks}
    else:
      completion = self._update_completion(action)
    self.eligibility = self._update_eligibility(completion)
    self.mask = self._update_mask(completion)
    self.completion = completion

    # 2. time_over
    if step is not None:
      self.step_count += step
    time_over = self.step_count >= self.max_step # Time exceeds episode limit
    agent_over = not any([m and e for m, e in zip(self.mask, self.eligibility)]) # No more action (option) left to execute
    action_over = self._check_environment_done(action) # Current action terminates environment

    self.done = time_over or agent_over or action_over
    self.step_done = time_over

    if self._verbose_level > 1 and self.rank == 0:
      print('='*40)
      print('[%d] Available options:'%(self.step_count))
      for subtask_name in self.eligibility:
        if self.eligibility[subtask_name] and self.mask[subtask_name]:
          print(subtask_name)

  def _get_observation(self):
    # Comprehensive version. slower.
    completion = self._dict_to_array(
        input=self.completion,
        dim=self.num_total_subtasks,
        mapping=self._subtask_pool_name_to_id
    )
    # Note: the dimension (action_dim) should be consistent with
    # WobConfig's pool_to_index (_construct_mappings).
    mask = self._dict_to_array(
        input=self.mask,
        dim=self.action_dim,
        # TODO: This had different number of subtasks from SUBTASK_POOL_NAME_TO_ID,
        # which results in differnt dimension (completion.shape != mask)
        mapping=self._option_name_to_id
    )
    eligibility = self._dict_to_array(
        input=self.eligibility,
        dim=self.action_dim,
        # TODO: This had different number of subtasks from SUBTASK_POOL_NAME_TO_ID,
        # which results in differnt dimension (completion.shape != eligibility)
        mapping=self._option_name_to_id
    )

    # Faster version. Use it if this is a bottleneck
    #completion = np.array([self.completion[name] if name in self.completion else 0 for name in SUBTASK_POOL_NAME_TO_ID], dtype=np.float32)
    #mask = np.array([self.mask[name] if name in self.mask else 0 for name in OPTION_NAME_TO_ID], dtype=np.float32)
    #eligibility = np.array([self.eligibility[name] if name in self.eligibility else 0 for name in OPTION_NAME_TO_ID], dtype=np.float32)

    observation = {
        'mask': mask,
        'completion': completion,
        'eligibility': eligibility,
        'option_success': np.asarray(True),
    }
    # Remaining episode step.
    remaining_steps = (self.max_step + 1 - self.step_count)
    observation.update(
        # TODO: Do we need action mask?
        action_mask=np.array(self.action_mask, dtype=np.float32),
        remaining_steps=np.array(remaining_steps, dtype=np.int32),
        termination=np.array(self.done, dtype=np.bool),
        step_done=np.array(self.step_done, dtype=np.float32),
    )
    return observation

  def step(self, action: np.ndarray, p=False):
    assert self.subtask_reward is not None, \
        'Subtask reward are not defined. Please reset the task.'
    action_names = self._unwrap_action(action)
    if self._verbose_level > 0 and self.rank == 0:
      print('[%d] Option name: %s'%(self.step_count, action_names))

    # TODO: Don't think this is needed any longer since we are masking out
    # the in-eligible actions. Also mismatch between the size of subtask pool
    # and the size of options cause an issue in action selection.
    # Set ineligible action to NO_OP
    #action = self._check_action_eligibility(action)
    self.action_mask = True

    # 1. Reward, step
    reward = np.array(self.subtask_reward[action_names]).astype(np.float32)
    step_count = 1  # XXX step: In web-nav, option == single primitive action

    # 2. Get next (comp, elig, mask, done)
    self._update_batch_params(action_names, step_count)

    # 3. Prepare outputs
    observation = self._get_observation()
    done = self.done
    return observation, reward, done, step_count

  # TODO: delete. not needed any longer?
  #def _check_action_eligibility(self, action):
  #  self.action_mask = True
  #  if action not in self.mask or not (self.mask[action] and self.eligibility[action]):
  #    # TODO(srsohn): Implement extra options (i.e., options without directly corresponding subtasks)
  #    self.action_mask = False
  #    return 'NO_OP'
  #  else:
  #    return action

  def _check_environment_done(self, action: str):
    """Check environment done."""
    return action in self.config.terminal_subtasks

  def _update_eligibility(self, completion: Dict[str, bool]):
    """Update eligiblity."""
    # 1. Visibility
    # 2. Whether the option will be successful.
    eligibility = self.graph.compute_eligibility(completion=completion)
    return eligibility

  def _update_mask(self, completion: Dict[str, bool]):
    """Update mask."""
    mask = {k: not v for k, v in completion.items()}

    # Check if the webpage sections (e.g. shipping address, payment) exist
    # in the graph, and mask out all the previous subtasks if the agent
    # proceeded to the next section.
    env_id = self.config.environment_id.capitalize()
    if 'v2' in env_id:
        subtasks = self.config.get_previous_subtasks(completion=completion)
        for subtask in subtasks:
          if subtask in self.subtasks:
            mask[subtask] = False
    else:
      if f'Click {env_id} GoPage4' in completion and completion[f'Click {env_id} GoPage4']:
        for subtask in self.subtasks:
          if subtask == f'Click {env_id} GoPage4': break
          mask[subtask] = False
      if f'Click {env_id} GoPage3' in completion and completion[f'Click {env_id} GoPage3']:
        for subtask in self.subtasks:
          if subtask == f'Click {env_id} GoPage3': break
          mask[subtask] = False
      if f'Click {env_id} GoPage2' in completion and completion[f'Click {env_id} GoPage2']:
        for subtask in self.subtasks:
          if subtask == f'Click {env_id} GoPage2': break
          mask[subtask] = False
      elif f'Click {env_id} GoPage1' in completion and completion[f'Click {env_id} GoPage1']:
        for subtask in self.subtasks:
          if subtask == f'Click {env_id} GoPage1': break
          mask[subtask] = False

    return mask

  def _update_completion(self, action: str):
    """Update completion."""
    # 1. Execution -> completion.
    # 2. Toggling/multiple choice
    next_comp = self.completion.copy()
    # If corresponding subtask exists, comp = True
    if action in next_comp:
      next_comp[action] = True

    # Process extra outcome
    if action in self.config.option_extra_outcome:
      option_outcome = self.config.option_extra_outcome[action]
      for subtask, val in option_outcome.items():
        next_comp[subtask] = val
    return next_comp

  def _dict_to_array(self, input: dict, dim: int, mapping: dict, dtype=np.float32):
    output = np.zeros((dim), dtype=dtype)
    for key, value in input.items():
      idx = mapping[key]
      output[idx] = value
    return output

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_specs

  def action_spec(self) -> types.NestedSpec:
    return specs.DiscreteArray(num_values=self.action_dim, dtype=np.int32)

  def reward_spec(self) -> types.NestedSpec:
    return specs.Array(shape=(), dtype=np.float32, name='reward')

  def _unwrap_action(self, action: np.ndarray) -> str:
    return self._option_id_to_name[int(action)]

