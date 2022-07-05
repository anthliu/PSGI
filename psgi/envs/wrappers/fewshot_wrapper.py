from typing import Optional
import numpy as np

import dm_env
from acme import types, specs


class FewshotWrapper(dm_env.Environment):
  def __init__(
      self,
      environment: dm_env.Environment,
      max_steps: int):
    self._environment = environment
    self.max_steps = max_steps
    self.step_count = None

  def reset_task(self, task_index: Optional[int] = None):
    return self._environment.reset_task(task_index)

  def set_task(self, task):
    self._environment.set_task(task)
  
  def evaluate_inferred_task(self, inferred_task):
    return self._environment.evaluate_inferred_task(inferred_task)

  def reset(self) -> dm_env.TimeStep:
    # Reset maximum trial step counts.
    self.step_count = np.zeros((), dtype=np.int32)

    # Reset environment.
    timestep = self._environment.reset()

    # Compute remaining steps.
    trial_remaining_steps = (self.max_steps - self.step_count + 1)
    timestep.observation.update(
        trial_remaining_steps=np.array(trial_remaining_steps, dtype=np.int32),
    )
    return timestep

  def step(self, action: np.ndarray) -> dm_env.TimeStep:
    assert self.step_count < self.max_steps, \
        'Trial is done. Environment must be reset before running the next trial.'
    assert self.max_steps is not None and self.step_count is not None, \
        'Environment is has not been initialized. Please reset first.'

    # Take an environment step.
    timestep = self._environment.step(action)

    # Increment count only during the adaptation phase.
    self.step_count += 1

    # Compute remaining steps.
    trial_remaining_steps = (self.max_steps - self.step_count + 1)
    timestep.observation.update(
        trial_remaining_steps=np.array(trial_remaining_steps, dtype=np.int32),
    )
    return timestep

  def observation_spec(self) -> types.NestedSpec:
    observation_spec = self._environment.observation_spec()
    observation_spec.update({
      'trial_remaining_steps': specs.Array(
        shape=(),
        dtype=np.int32,
        name='trial_remaining_steps')
    })
    return observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self._environment.action_spec()

  def reward_spec(self) -> types.NestedSpec:
    return self._environment.reward_spec()

  @property
  def task(self):
    return self._environment.task

  @property
  def task_embedding(self) -> dict:
    """Communicating with the agent (actor)."""
    return self._environment.task_embedding

  @property
  def num_tasks(self) -> int:
    """Total size of tasks/graphs."""
    return self._environment.num_tasks

  @property
  def num_options(self):
    """Size of the (task-specific) subtasks."""
    return self._environment.num_options

  @property
  def num_subtasks(self):
    """Size of the (task-specific) subtasks."""
    return self._environment.num_subtasks

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
