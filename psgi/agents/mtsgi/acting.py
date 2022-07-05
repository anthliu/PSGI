"""MTSGI Actor implementation."""

from typing import Dict, List
import numpy as np

import dm_env
import acme
from acme import specs, types
from psgi.agents import base
from psgi.agents.msgi.acting import MSGIActor
from psgi.utils.graph_utils import SubtaskGraph

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
    self._step_based_prob = np.sqrt(1 - (current_split / max_split))

  def _get_raw_logits(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    assert self._prior_actor is not None, 'Prior policy missing.'
    prior_active = self._prior_actor.get_availability(observation) # [num_envs, 1]: False if all(masked_elig==0) for prior_grprop.
    if np.any(prior_active):
      prior_confidence = self._prior_actor.confidence_score * np.power(self._prior_weights, 2)
      conf_based_prob = 1.0 - np.clip(self._actor.confidence_score / prior_confidence, 0.0, 1.0)
      self._prior_samping_prob = np.minimum(conf_based_prob, self._step_based_prob)

      # Run GRProp for test phase.
      prob = np.random.uniform(size=self._prior_samping_prob.shape)
      prior_mask = (prob < self._prior_samping_prob) * prior_active
      prior_logits = self._prior_actor._get_raw_logits(observation)
      actor_logits = self._actor._get_raw_logits(observation)
      #logits = prior_logits * prior_mask[:, None] + actor_logits * (1 - prior_mask[:, None])
      logits = prior_logits + actor_logits  # XXX mix prior & current
    else:
      logits = self._actor._get_raw_logits(observation)
    return logits

  @property
  def prior_samping_prob(self) -> np.ndarray:
    return self._prior_samping_prob
