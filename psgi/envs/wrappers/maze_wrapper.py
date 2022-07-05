from typing import Optional
import numpy as np

import dm_env
from acme import types

from psgi import envs


class MazeWrapper(dm_env.Environment):
  # TODO: Unify with MazeOptionEnv (or making it gym-compatible)

  def __init__(self, environment: envs.MazeEnv):
    self._environment = environment
    self._reset_next_step = True

  @property
  def environment_id(self) -> str:
    return self._environment.environment_id

  def _assertions(self):
    num_indices = self.index_to_pool.shape[-1]
    num_pool = self.pool_to_index.shape[-1]
    assert np.all(self.pool_to_index < num_indices)
    unique_elements, counts = np.unique(self.pool_to_index, return_counts=True)
    if -1 in unique_elements:
      idx = int(np.argwhere(unique_elements == -1))
      assert np.all(counts[:idx] == 1) and np.all(counts[idx+1:] == 1), 'Error! index should be unique'
    else:
      assert num_indices == num_pool, "Error! The empty value in self.pool_to_index should be -1!"
      assert np.all(counts == 1), 'Error! self.pool_to_index should be unique'

    # index_to_pool
    assert np.all(self.index_to_pool < num_pool)
    assert np.all(self.index_to_pool >= 0)
    unique_elements, counts = np.unique(self.index_to_pool, return_counts=True)
    assert np.all(counts == 1), 'Error! self.index_to_pool should be unique'

  def reset_task(self, task_index: Optional[int] = None, **kwargs):
    """Sample/reset a task for meta-learning and multi-task learning."""
    self._assertions()
    return self._environment.reset_task(task_index=task_index, **kwargs)

  def load_debugging_map(self, filename: str):
    self._environment.load_debugging_map(filename)

  def set_task(self, task):
    """fixed the current task to the given task. used for aligning the task
      between the adaptation & test environments.
    """
    self._environment.set_task(task)
    self._assertions()

  def reset(self):
    # Reset environment.
    self._reset_next_step = False
    observation, _ = self._environment.reset()
    return dm_env.TimeStep(dm_env.StepType.FIRST, 0.0, 1.0, observation)

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    # Reset if previous timestep was LAST.
    if self._reset_next_step:
      return self.reset()

    # Take an environment step.
    observation, reward, done, _ = self._environment.step(action)
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

  def discount_spec(self):
    return self._environment.discount_spec()

  @property
  def task(self):
    """Communicating between envs. Primitive."""
    return self._environment.task

  @property
  def num_tasks(self) -> int:
    """Total size of tasks/graphs."""
    return self._environment.num_tasks

  @property
  def subtasks(self):
    # Return list of subtasks
    return self._environment.index_to_pool

  @property
  def num_subtasks(self):
    """Size of the (task-specific) subtasks."""
    return self._environment.num_subtasks

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
