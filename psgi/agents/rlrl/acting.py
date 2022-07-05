"""RL^2 actor implementation."""

from typing import Dict, Optional

from psgi.agents import base

from acme import adders
from acme import core
from acme import specs, types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import numpy as np
import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class FastRLActor(base.BaseActor):
  """A recurrent RL actor."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.RNNCore,
      queue: adders.Adder = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      verbose_level: Optional[int] = 0
  ):
    super().__init__(
        environment_spec=environment_spec,
        verbose_level=verbose_level,
    )
    # Store these for later use.
    self._queue = queue
    self._variable_client = variable_client
    self._network = network

    # TODO(b/152382420): Ideally we would call tf.function(network) instead but
    # this results in an error when using acme RNN snapshots.
    #self._policy = tf.function(network.__call__)
    self._policy = network

    self._state = None
    self._prev_state = None
    self._prev_logits = None

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> types.NestedArray:
    if self._state is None:
      assert observation['completion'].ndim == 2, \
        'Observation must be 2-dimensional.'
      batch_size = observation['completion'].shape[0]
      self._state = self._network.initial_state(batch_size)

    # Forward.
    (logits, _), new_state = self._policy(observation, self._state)

    self._prev_logits = logits
    self._prev_state = self._state
    self._state = new_state

    return logits.numpy()

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._queue is not None:
      self._queue.add_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if self._queue is None:
      return

    extras = {'logits': self._prev_logits, 'core_state': self._prev_state}
    extras = tf2_utils.to_numpy(extras)
    self._queue.add(action, next_timestep, extras)

  def update(self):
    if self._variable_client:
      self._variable_client.update()
