import collections

from acme import types
from acme.adders import base
from acme.adders.reverb import Step
from acme.adders.reverb import utils
from acme.utils import tree_utils

import numpy as np
import random
import dm_env


class ReplayBuffer(base.Adder):
  """Queue based replay buffer."""

  def __init__(
      self,
      max_queue_size: int,
      batch_size: int
  ):
    self._buffer = collections.deque(maxlen=max_queue_size)
    self._batch_size = batch_size
    self._can_sample = False
    self._next_observation = None
    self._start_of_episode = False
    self._step = 0

  def __len__(self):
    return len(self._buffer)

  def reset(self):
    self._buffer.clear()
    self._next_observation = None
    self._step = 0

  def add_first(self, timestep: dm_env.TimeStep):
    # Clear the buffer for new experiences.
    self._next_observation = timestep.observation
    self._start_of_episode = True

  def add(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
      extras: types.NestedArray = (),
  ):
    assert self._next_observation is not None, \
        'add_first must be called before we call add.'

    # Add the timestep to the buffer.
    self._buffer.append(
        Step(
            observation=self._next_observation,
            action=action,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            start_of_episode=self._start_of_episode,  # dummy
            extras=extras
        ))

    # Record the next observation.
    self._next_observation = next_timestep.observation
    self._start_of_episode = False
    self._step += 1

  def sample(self):
    """Sample a batch of experiences."""
    samples = [self._buffer.popleft() for _ in range(self._batch_size)]
    return tree_utils.stack_sequence_fields(samples)
