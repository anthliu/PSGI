"""Baselines Actor implementations."""

from typing import Dict
import numpy as np

import dm_env
import acme
from acme import specs, types
from acme.tf import utils as tf2_utils

from psgi.agents import base
from psgi.utils import tf_utils


class EvalWrapper(base.BaseActor):
  """Actor Base class
  """
  def __init__(
      self,
      actor: base.BaseActor,
      verbose_level: int = 0,
  ):
    self._actor = actor
    self._verbose_level = verbose_level

  def observe_task(self, tasks):
    self._actor.observe_task(tasks)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep=next_timestep)

  def update(self):
    self._actor.update()

  def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    masked_eligibility = np.multiply(observation['mask'], observation['eligibility'])

    # Choose from stochastic policy
    termination = observation['termination']
    logits = self._get_raw_logits(observation)

    # When the environment is done, replace with uniform policy. This action
    # will be ignored by the environment
    if np.any(termination):
      masked_eligibility[termination, :] = 1
      logits[termination, :] = 1.

    # Masking
    masked_logits = self._mask_logits(logits, masked_eligibility)

    # Add a random epsilon to break the ties.
    masked_logits = masked_logits + 1e-6 * np.random.random(masked_logits.shape)

    # Choose greedy action
    actions = np.argmax(masked_logits, axis=1)
    if self._verbose_level > 0:
      print(f'actions (pool)={actions[0].item()}')
      print('logits (pool)=', masked_logits[0])
    if self._verbose_level > 1:
      print('masked_eligibility (pool)= ', masked_eligibility[0])

    # Assertion
    action_masks = np.take_along_axis(arr=masked_eligibility, indices=np.expand_dims(actions, axis=-1), axis=1).squeeze(axis=1)
    assert np.all(action_masks == 1), "selected action is ineligible or masked-out"
    return actions

  def _get_raw_logits(self, observation: np.ndarray):
    return self._actor._get_raw_logits(observation)
