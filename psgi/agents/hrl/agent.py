"""Advantage actor-critic (A2C) agent implementation."""

from typing import Dict, Optional

from psgi import agents
from psgi.agents import queue
from psgi.agents import base
from psgi.agents import meta_agent
from psgi.agents.hrl import acting
from psgi.agents.hrl import learning
from psgi.utils import acme_utils
from psgi.utils import snt_utils

import acme
from acme import datasets
from acme import specs
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import networks
from acme.utils import counting
from acme.utils import loggers
from acme.adders import base as adder

import tree
import dm_env
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf


class HRL(meta_agent.MetaAgent):
  """Hierarchical RL agent for meta loop."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      mode: str,
      network: snt.RNNCore,
      n_step_horizon: int,
      learning_rate: float,
      discount: float,
      entropy_cost: float,
      baseline_cost: float,
      max_abs_reward: float,
      max_gradient_norm: float,
      decay: float = 0.99,
      epsilon: float = 1e-5,
      max_queue_size: int = 100000,
      logger: loggers.Logger = None,
      verbose_level: Optional[int] = 0,
  ):
    """
      Args:
        visualize: whether to visualize subtask graph inferred by ILP module.
        directory: directory for saving the subtask graph visualization.
        environment_id: environment name (e.g. toywob) for graph visualization.
    """
    self._environment_spec = environment_spec
    self._mode = mode
    self._verbose_level = verbose_level
    #
    self._discount = discount
    self._n_step_horizon = n_step_horizon
    self._learning_rate = learning_rate
    self._entropy_cost = entropy_cost
    self._baseline_cost = baseline_cost
    self._max_gradient_norm = max_gradient_norm
    self._max_abs_reward = max_abs_reward
    
    self._decay = decay
    self._epsilon = epsilon

    # Build network and replay buffer.
    self._network = network
    self._queue = queue.ReplayBuffer(
        max_queue_size=max_queue_size,
        batch_size=n_step_horizon
    )

    # Logger.
    self._logger = logger

  def instantiate_adapt_agent(self) -> agents.Agent:
    """Instantiate 'fast' agent for adaptation phase. This fast agent will
      interact with the environment and perform fast adaptation (learning).
    """
    self._adapt_agent = A2C(
        environment_spec=self._environment_spec,
        network=self._network,
        queue=self._queue,
        logger=self._logger,
        discount=self._discount,
        n_step_horizon=self._n_step_horizon,
        learning_rate=self._learning_rate,
        entropy_cost=self._entropy_cost,
        baseline_cost=self._baseline_cost,
        max_abs_reward=self._max_abs_reward,
        max_gradient_norm=self._max_gradient_norm,
        verbose_level=self._verbose_level
    )
    return self._adapt_agent

  def instantiate_test_actor(self) -> base.BaseActor:
    """Instantiate a new Actor for test phase. This actor will interact with
      the environment and evaluate the performance of the (adapted) fast agent.
    """
    # Return A2CActor for test phase.
    return acting.A2CActor(
            environment_spec=self._environment_spec,
            network=self._network,
            verbose_level=self._verbose_level,
        )

  def reset_agent(self, environment: dm_env.Environment):
    """Reset the 'fast' agent upon samping a new task."""
    # Re-initialize A2C network parameter.
    acme_utils.reinitialize_weights(self._network)

    # Reset optimizer.
    self._adapt_agent._learner._optimizer = snt.optimizers.RMSProp(
        learning_rate=self._learning_rate,
        decay=self._decay,
        epsilon=self._epsilon
    )

    # Reset replay buffer.
    self._queue.reset()

  def update(self):
    # HRL has no meta-training. Do nothing.
    return

  def update_adaptation_progress(self, current_split: int, max_split: int):
    """Update the adaptation progress with adaptation / testing actors
    """
    pass # do nothing


class A2C(agents.Agent):
  """A2C Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.RNNCore,
      queue: adder.Adder,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      discount: float = 0.99,
      n_step_horizon: int = 16,
      learning_rate: float = 1e-3,
      entropy_cost: float = 0.01,
      baseline_cost: float = 0.5,
      max_abs_reward: Optional[float] = None,
      max_gradient_norm: Optional[float] = None,
      verbose_level: Optional[int] = 0,
  ):
    num_actions = environment_spec.actions.num_values
    self._logger = logger or loggers.TerminalLogger('agent')

    extra_spec = {
        'core_state': network.initial_state(1),
        'logits': tf.ones(shape=(1, num_actions), dtype=tf.float32)
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    tf2_utils.create_variables(network, [environment_spec.observations])

    actor = acting.A2CActor(
        environment_spec=environment_spec,
        verbose_level=verbose_level,
        network=network,
        queue=queue
    )
    learner = learning.A2CLearner(
        environment_spec=environment_spec,
        network=network,
        dataset=queue,
        counter=counter,
        logger=logger,
        discount=discount,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        baseline_cost=baseline_cost,
        max_gradient_norm=max_gradient_norm,
        max_abs_reward=max_abs_reward,
    )

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=0,
        observations_per_step=n_step_horizon)
