"""Runs multiple environments serially to simulate parallel envs."""

from typing import Optional, Any, Callable, Sequence
import functools

from absl import logging

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

import dm_env
from acme import types

from psgi.utils import tf_utils

EnvConstructor = Callable[[], dm_env.Environment]


def wrap_as_batch_env(env: dm_env.Environment):
  if hasattr(env, 'batched') and env.batched:
    raise ValueError("The environment is already batched.")

  def _env_factory(env):
    return env

  batch_env = SerialEnvironments([functools.partial(_env_factory, env=env)])
  assert batch_env.batch_size == 1
  return batch_env


class SerialEnvironments(dm_env.Environment):
  """Batch together environments. This should give the same outputs as
  ParallelEnvironments do (i.e. batched observations), but the only difference
  is that no external process is made and that all environments are executed
  in serial. Used for debugging/testing purposes.
  """

  def __init__(self,
               env_constructors: Sequence[EnvConstructor],
               flatten: bool = False):
    """Batch together environments and simulate them in external processes.
    The environments can be different but must use the same action and
    observation specs.
    Args:
      env_constructors: List of callables that create environments.
      start_serially: Whether to start environments serially or in parallel.
      blocking: Whether to step environments one after another.
      flatten: Boolean, whether to use flatten action and time_steps during
        communication to reduce overhead.
    Raises:
      ValueError: If the action or observation specs don't match.
    """
    super(SerialEnvironments, self).__init__()
    if any([not callable(ctor) for ctor in env_constructors]):
      raise TypeError(
          'Found non-callable `env_constructors` in `SerialEnvironments` '
          '__init__ call. Did you accidentally pass in environment instances '
          'instead of constructors? Got: {}'.format(env_constructors))
    self._envs = [ctor() for ctor in env_constructors]
    self._num_envs = len(env_constructors)
    self.update_spec()
    self._flatten = flatten
  
  def update_spec(self):
    self._action_spec = self._envs[0].action_spec()
    self._observation_spec = self._envs[0].observation_spec()
    #self._time_step_spec = self._envs[0].time_step_spec()
    if any(env.action_spec() != self._action_spec for env in self._envs):
      raise ValueError('All environments must have the same action spec.')
    #if any(env.time_step_spec() != self._time_step_spec for env in self._envs):
    #  raise ValueError('All environments must have the same time_step_spec.')

  @property
  def batched(self) -> bool:
    return True

  @property
  def batch_size(self) -> int:
    return self._num_envs

  @property
  def envs(self):
    return self._envs

  @property
  def task(self) -> Sequence[Any]:
    return [env.task for env in self._envs]

  @property
  def num_subtasks(self) -> Sequence[int]:
    return [env.num_subtasks for env in self._envs]

  @property
  def num_options(self) -> Sequence[int]:
    return [env.num_options for env in self._envs]

  @property
  def task_embedding(self) -> Sequence[dict]:
    return [env.task_embedding for env in self._envs]

  @property
  def pool_to_index(self) -> Sequence[np.ndarray]:
    return [env.pool_to_index for env in self._envs]

  @property
  def index_to_pool(self) -> Sequence[np.ndarray]:
    return [env.index_to_pool for env in self._envs]
  
  @property
  def parameters(self) -> Sequence[list]: # list of (local) parameter
    return [env.parameters for env in self._envs]
  
  @property
  def features(self) -> Sequence[list]: # list of (local) parameter
    return [env.feature for env in self._envs]

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self._action_spec

  def reset_task(self, task_index: Optional[int] = None):
    tasks = []
    for env in self._envs:
      t = env.reset_task(task_index)
      tasks.append(t)
    self.update_spec()
    return tasks

  def set_task(self, tasks: Sequence[Any]) -> None:
    for task, env in zip(tasks, self._envs):
      env.set_task(task)
    self.update_spec()

  def reset(self):
    """Reset episode of all environments and combine the resulting observation.
    Returns:
      Time step with batch dimension.
    """
    time_steps = [env.reset() for env in self._envs]
    return self._stack_time_steps(time_steps)

  def evaluate_inferred_task(self, inferred_tasks: list):
    eval_results = []
    for env, inferred_task in zip(self._envs, inferred_tasks):
      eval_result = env.evaluate_inferred_task(inferred_task)
      eval_results.append(eval_result)
    return eval_results

  def step(self, actions: np.ndarray):
    """Forward a batch of actions to the wrapped environments.
    Args:
      actions: Batched action, possibly nested, to apply to the environment.
    Raises:
      ValueError: Invalid actions.
    Returns:
      Batch of observations, rewards, and done flags.
    """
    assert actions.ndim == 1 and isinstance(actions, np.ndarray)
    time_steps = [
        env.step(action)
        for env, action in zip(self._envs, self._unstack_actions(actions))]
    # When blocking is False we get promises that need to be called.
    return self._stack_time_steps(time_steps)

  def close(self) -> None:
    """Close all external process."""
    logging.info('Closing all processes.')
    for env in self._envs:
      env.close()
    logging.info('All processes closed.')

  def _stack_time_steps(self, time_steps):
    """Given a list of TimeStep, combine to one with a batch dimension."""
    if self._flatten:
      raise NotImplementedError("We don't have time_step_spec.")
      return tf_utils.fast_map_structure_flatten(
          lambda *arrays: np.stack(arrays), self._time_step_spec, *time_steps)
    else:
      return tf_utils.fast_map_structure(
          lambda *arrays: np.stack(arrays), *time_steps)

  def _unstack_actions(self, batched_actions):
    """Returns a list of actions from potentially nested batch of actions."""
    flattened_actions = tf.nest.flatten(batched_actions)
    if self._flatten:
      unstacked_actions = zip(*flattened_actions)
    else:
      unstacked_actions = [
          tf.nest.pack_sequence_as(batched_actions, actions)
          for actions in zip(*flattened_actions)
      ]
    return unstacked_actions
