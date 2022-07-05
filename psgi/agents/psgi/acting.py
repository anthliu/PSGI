"""PSGI Actor implementation."""

from typing import Dict, List

import numpy as np

import dm_env
import acme
from acme import types

from psgi.agents import base
from psgi.utils import graph_utils
from psgi.utils.graph_utils import SubtaskGraph

class PSGIActor(base.BaseActor):
  """An MSGI Actor is a wrapper of any actor to add the interaction with ILP."""

  # TODO: Consider making this is an "Agent" (do learning as well).

  def __init__(
      self,
      actor: acme.Actor,
      pilp: 'psgi.graph.ilp.PILP',
      verbose_level: int = 0,
    ):
    """
    Args:
      actor: an underlying adaptation policy (e.g. random or grprop).
      pilp: PILP module.
    """
    self._actor = actor
    self._pilp = pilp
    self._verbose_level = verbose_level

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

    # Add timestep to PILP.
    self._add_timestep_to_pilp(timestep=timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep)

    # Add timestep to PILP.
    self._add_timestep_to_pilp(timestep=next_timestep, action=action)

  def update(self):
    # Note: MSGI's learning behavior is implemented in MSGILearner.
    # We may want to combine implementation (acting + learning) as 'MSGIAgent',
    # directly implementing `acme.Agent`.
    self._actor.update()

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    return self._actor._get_raw_logits(observation)

  def _add_timestep_to_pilp(
      self,
      timestep: dm_env.TimeStep,
      action: types.NestedArray = None):
    # ILP expects these to be 2-dim numpy.
    if action is not None:
      action = np.expand_dims(action, axis=-1)
      rewards = np.expand_dims(timestep.reward, axis=-1)
    else:
      rewards = None  # ilp assumes reward = None when action = None
    indexed_obs = timestep.observation# XXX check this. Global action should be same as local action/option/subtasks

    # ignore dummy action after last timestep & failed option
    is_valid = (1 - timestep.first()) * indexed_obs['option_success']

    self._pilp.insert(
        is_valid=is_valid,
        completion=indexed_obs['completion'],
        eligibility=indexed_obs['eligibility'],
        parameter_embeddings=indexed_obs['parameter_embeddings'],
        action_id=action,
        rewards=rewards
    )

class MixedActor(base.BaseActor):
  def __init__(
      self,
      actor: acme.Actor,
      prior_actor: acme.Actor,
  ):
    """
    Args:
      actor: an underlying adaptation policy (e.g. random or grprop).
      ilp: ILP module.
    """
    self._actor = actor
    self._prior_actor = prior_actor
    self._prior_weights = 0.75
    self._prior_samping_prob = 1.0
    self._step_based_prob = 1.0

  def observe_task(self, tasks: List[SubtaskGraph]):
    self._actor.observe_task(tasks)

  def set_prior_weights(self, prior_weights):
    self._prior_weights = prior_weights

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep)

  def update_adaptation_progress(
      self,
      current_split: int,
      max_split: int,
  ):
    #self._step_based_prob = np.sqrt(1 - (current_split / max_split))
    self._prior_stop_split = 0.2
    self._step_based_prob = np.maximum(0., 1 - (1/self._prior_stop_split) * (current_split / max_split))

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert self._prior_actor is not None, 'Prior policy missing.'
    #prior_active = self._prior_actor.get_availability(observation) # [num_envs, 1]: False if all(masked_elig==0) for prior_grprop.
    if self._prior_actor.is_ready:
      #prior_confidence = self._prior_actor.confidence_score * np.power(self._prior_weights, 2)
      #conf_based_prob = 1.0 - np.clip(self._actor.confidence_score / prior_confidence, 0.0, 1.0)
      #self._prior_samping_prob = np.minimum(conf_based_prob, self._step_based_prob)
      self._prior_samping_prob = self._step_based_prob# TODO currently just use step based probability

      # Run GRProp for test phase.
      prob = np.random.uniform(size=self._prior_samping_prob.shape)
      #prior_mask = (prob < self._prior_samping_prob) * prior_active
      prior_mask = prob < self._prior_samping_prob
      prior_logits = self._prior_actor._get_raw_logits(observation)
      actor_logits = self._actor._get_raw_logits(observation)
      #logits = prior_logits * prior_mask + actor_logits * (1 - prior_mask)
      #logits = prior_logits + actor_logits  # XXX mix prior & current
      logits = prior_logits if prior_mask else actor_logits
      #logits = actor_logits
    else:
      logits = self._actor._get_raw_logits(observation)
    return logits

  @property
  def prior_samping_prob(self) -> np.ndarray:
    return self._prior_samping_prob
