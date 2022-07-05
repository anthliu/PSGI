"""MSGI Actor implementation."""

from typing import Dict

import numpy as np

import dm_env
import acme
from acme import types

from psgi.agents import base
from psgi.utils import graph_utils


class MSGIActor(base.BaseActor):
  """An MSGI Actor is a wrapper of any actor to add the interaction with ILP."""

  # TODO: Consider making this is an "Agent" (do learning as well).

  def __init__(
      self,
      actor: acme.Actor,
      ilp: 'psgi.graph.ilp.ILP',
      verbose_level: int = 0,
    ):
    """
    Args:
      actor: an underlying adaptation policy (e.g. random or grprop).
      ilp: ILP module.
    """
    self._actor = actor
    self._ilp = ilp
    self._verbose_level = verbose_level

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

    # Add timestep to ILP.
    self._add_timestep_to_ilp(timestep=timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep)

    # Add timestep to ILP.
    self._add_timestep_to_ilp(timestep=next_timestep, action=action)

  def update(self):
    # Note: MSGI's learning behavior is implemented in MSGILearner.
    # We may want to combine implementation (acting + learning) as 'MSGIAgent',
    # directly implementing `acme.Agent`.
    self._actor.update()

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    return self._actor._get_raw_logits(observation)

  def _add_timestep_to_ilp(
      self,
      timestep: dm_env.TimeStep,
      action: types.NestedArray = None):
    # ILP expects these to be 2-dim numpy.
    if action is not None:
      action = np.expand_dims(action, axis=-1)
      rewards = np.expand_dims(timestep.reward, axis=-1)
    else:
      rewards = None  # ilp assumes reward = None when action = None
    indexed_obs = graph_utils.transform_obs(
        observation=timestep.observation,
        index_to_pool=self._ilp.index_to_pool,
    )

    # ignore dummy action after last timestep & failed option
    is_valid = (1 - timestep.first()) * indexed_obs['option_success']

    self._ilp.insert(
        is_valid=is_valid,
        completion=indexed_obs['completion'],
        eligibility=indexed_obs['eligibility'],
        action_id=action,
        rewards=rewards
    )
