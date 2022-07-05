from typing import Optional
import numpy as np

import dm_env
from acme import types

from psgi import envs


class WoBWrapper(dm_env.Environment):
  def __init__(self, environment: envs.BaseWoBEnv):
    self._environment = environment
    self._reset_next_step = True

  @property
  def environment_id(self) -> str:
    return self._environment.environment_id

  def reset_task(self, task_index: Optional[int] = None):
    """Sample/reset a task for meta-learning and multi-task learning."""
    return self._environment.reset_task(task_index)

  def set_task(self, task):
    """Fixed the current task to the given task. Used for aligning the task
      between the adaptation & test environments.
    """
    self._environment.set_task(task)

  def reset(self) -> dm_env.TimeStep:
    # Reset environment.
    self._reset_next_step = False
    observation = self._environment.reset()
    return dm_env.TimeStep(dm_env.StepType.FIRST, 0.0, 1.0, observation)

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    # Reset if previous timestep was LAST.
    if self._reset_next_step:
      return self.reset()

    # Take an environment step.
    observation, reward, done, _ = self._environment.step(action)

    self._assert_shape(observation, reward, done)
    self._reset_next_step = done

    if done:
      return dm_env.termination(reward=reward, observation=observation)  # After this, it's always LAST
    return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self) -> types.NestedSpec:
    return self._environment.observation_spec()

  def action_spec(self) -> types.NestedSpec:
    return self._environment.action_spec()

  def reward_spec(self) -> types.NestedSpec:
    return self._environment.reward_spec()

  def _assert_shape(self, observation, reward, done):
    if 'mask' in observation:
      assert observation['mask'].ndim == 1
    assert observation['completion'].ndim == 1
    assert observation['eligibility'].ndim == 1
    assert reward is None or np.isscalar(reward) or reward.ndim == 0
    assert done is None or np.isscalar(done) or done.ndim == 0

  def evaluate_inferred_task(self, inferred_task):
    return self._environment.evaluate_inferred_task(inferred_task)
    
  @property
  def task(self):
    """Communicating between envs. Primitive."""
    return self._environment.task

  @property
  def num_tasks(self) -> int:
    """Total size of tasks/graphs."""
    return 1  # XXX single task for now

  @property
  def num_subtasks(self):
    """Size of the (task-specific) subtasks."""
    return self._environment.num_subtasks

  @property
  def num_options(self):
    """Size of the (task-specific) options."""
    return self._environment.num_options

  @property
  def task_embedding(self) -> dict:
    """Communicating with the agent (actor)."""
    return self._environment.task_embedding

  @property
  def pool_to_index(self):
    return self._environment.pool_to_index

  @property
  def index_to_pool(self):
    return self._environment.index_to_pool
  
  @property
  def parameters(self):
    return self._environment.parameters
  
  @property
  def feature(self):
    return self._environment.feature
