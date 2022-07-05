"""RL^2 agent implementation."""

import os
from inspect import stack
import time
from typing import Dict, Optional

from psgi import agents
from psgi.agents import queue
from psgi.agents import base
from psgi.agents import meta_agent
from psgi.agents.rlrl import acting
from psgi.agents.rlrl import learning
from psgi.utils import snt_utils, acme_utils

import acme
from acme import specs
from acme import types
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tree_utils
from acme.adders import base as adder
from acme.adders.reverb import Step

import tree
import trfl
import dm_env
import numpy as np
import sonnet as snt

import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import schedules

tfd = tfp.distributions


class RLRL(meta_agent.MetaAgent):
  """RL^2 meta agent for meta loop."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      mode: str,
      network: snt.RNNCore,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      snapshot_dir: Optional[str] = None,
      n_step_horizon: int = 16,
      minibatch_size: int = 80,
      learning_rate: float = 2e-3,
      discount: float = 0.99,
      gae_lambda: float = 0.99,
      decay: float = 0.99,
      epsilon: float = 1e-5,
      entropy_cost: float = 0.,
      baseline_cost: float = 1.,
      max_abs_reward: Optional[float] = None,
      max_gradient_norm: Optional[float] = None,
      max_queue_size: int = 100000,
      verbose_level: Optional[int] = 0,
  ):
    # Internalize spec and replay buffer.
    self._environment_spec = environment_spec
    self._mode = mode
    self._verbose_level = verbose_level
    self._minibatch_size = minibatch_size
    self._queue = queue.ReplayBuffer(
        max_queue_size=max_queue_size,
        batch_size=n_step_horizon
    )

    # Internalize network.
    self._network = network

    # Setup optimizer and learning rate scheduler.
    self._learning_rate = tf.Variable(learning_rate, trainable=False)
    self._lr_scheduler = schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=8000,  # TODO make a flag
        decay_rate=0.96    # TODO make a flag
    )
    self._optimizer = snt.optimizers.RMSProp(
        learning_rate=self._learning_rate,
        decay=decay,
        epsilon=epsilon
    )
    #self._optimizer = snt.optimizers.Adam(
    #    learning_rate=self._learning_rate,
    #)

    # Hyperparameters.
    self._discount = discount
    self._gae_lambda = gae_lambda
    self._entropy_cost = entropy_cost
    self._baseline_cost = baseline_cost

    # Set up reward/gradient clipping.
    if max_abs_reward is None:
      max_abs_reward = np.inf
    if max_gradient_norm is None:
      max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
    self._max_abs_reward = tf.convert_to_tensor(max_abs_reward)
    self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

    if snapshot_dir is not None:
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'network': self._network},
          directory=snapshot_dir,
          time_delta_minutes=60.
      )

    # Logger.
    self._counter = counter or counting.Counter()
    self._logger = logger

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None
    self._pi_old = None

  def save(self, filename):
    snapshotter = tf2_savers.Snapshotter(
        objects_to_save={'network': self._network}, directory=filename
    )
    snapshotter.save()
  
  def load(self, filename):
    if isinstance(filename, list):
      filename = filename[-1]
    self._network = tf.saved_model.load(os.path.join(filename, 'network'))

  def instantiate_adapt_agent(self) -> agents.Agent:
    """Instantiate 'fast' agent for adaptation phase. This fast agent will
      interact with the environment and perform fast adaptation (learning).
    """
    self._adapt_agent = FastRLAgent(
        environment_spec=self._environment_spec,
        network=self._network,
        queue=self._queue,
        verbose_level=self._verbose_level
    )
    return self._adapt_agent

  def instantiate_test_actor(self) -> base.BaseActor:
    """Instantiate a new Actor for test phase. This actor will interact with
      the environment and evaluate the performance of the (adapted) fast agent.
    """
    # Return RLActor for test phase.
    self._test_actor = acting.FastRLActor(
        environment_spec=self._environment_spec,
        network=self._network,
        verbose_level=self._verbose_level
    )
    return self._test_actor

  def reset_agent(self, environment: dm_env.Environment):
    """Reset the 'fast' agent upon samping a new task."""
    # Reset RNN hidden state.
    self._adapt_agent._actor._state = None

    # Reset replay buffer.
    self._queue.reset()

  def update_adaptation_progress(self, current_split: int, max_split: int):
    """Update the adaptation progress with adaptation / testing actors
    """
    pass # do nothing

  @tf.function
  def _step(self, data: Step) -> Dict[str, tf.Tensor]:
    """Does an SGD step on a batch of sequences."""
    observations, actions, rewards, discounts, _, extra = data
    core_state = tree.map_structure(lambda s: s[0], extra['core_state'])

    actions = actions[:-1]  # [T-1]
    rewards = rewards[:-1]  # [T-1]
    discounts = discounts[:-1]  # [T-1]

    # Workaround for NO_OP actions
    # In some environments, passing NO_OP(-1) actions would lead to a crash.
    # These actions (at episode boundaries) should be ignored anyway,
    # so we replace NO_OP actions with a valid action index (0).
    actions = (tf.zeros_like(actions) * tf.cast(actions == -1, tf.int32) +
               actions * tf.cast(actions != -1, tf.int32))

    with tf.GradientTape() as tape:
      # Unroll current policy over observations.
      (logits, values), _ = snt.static_unroll(self._network, observations,
                                              core_state)

      # TODO: mask policy here as well.
      # Masked policy.
      #masked_eligibility = observations['mask'] * observations['eligibility']
      #out_logits = logits - logits.min(axis=-1, keepdims=True)
      #out_logits[masked_eligibility == 0] = -np.infty
      #out_logits -= out_logits.max(axis=-1, keepdims=True)
      pi = tfd.Categorical(logits=logits[:-1])

      # Optionally clip rewards.
      rewards = tf.clip_by_value(rewards,
                                 tf.cast(-self._max_abs_reward, rewards.dtype),
                                 tf.cast(self._max_abs_reward, rewards.dtype))
      values = tf.clip_by_value(values,
                                tf.cast(0.4, values.dtype),
                                tf.cast(0.4, values.dtype))
      # Compute returns (optionally, GAE)
      discounted_returns = trfl.generalized_lambda_returns(
          rewards=tf.cast(rewards, tf.float32),
          pcontinues=tf.cast(self._discount*discounts, tf.float32),
          values=tf.cast(values[:-1], tf.float32),
          bootstrap_value=tf.cast(values[-1], tf.float32),
          lambda_=self._gae_lambda
      )
      advantages = discounted_returns - values[:-1]

      # Compute actor & critic losses.
      critic_loss = tf.square(advantages)
      #policy_gradient_loss = trfl.policy_gradient(
      #    policies=pi,
      #    actions=actions,
      #    action_values=advantages
      #)
      # TODO: Remove later.
      action_values = advantages
      policy_vars = None
      policy_vars = list(policy_vars) if policy_vars else list()
      with tf1.name_scope(values=policy_vars + [actions, action_values], name="policy_gradient"):
        actions = tf1.stop_gradient(actions)
        action_values = tf1.stop_gradient(action_values)
        log_prob_actions = pi.log_prob(actions)
        # Prevent accidental broadcasting if possible at construction time.
        action_values.get_shape().assert_is_compatible_with(
            log_prob_actions.get_shape())
        policy_gradient_loss = -tf1.multiply(log_prob_actions, action_values)

      #entropy_loss = trfl.policy_entropy_loss(pi).loss
      # TODO: Remove later.
      entropy_info = trfl.policy_entropy_loss(pi)
      entropy_loss = entropy_info.loss


      loss = tf.reduce_mean(policy_gradient_loss +
                            self._baseline_cost * critic_loss +
                            self._entropy_cost * entropy_loss)

      # Compute gradients and optionally apply clipping.
      gradients = tape.gradient(loss, self._network.trainable_variables)
      gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
      self._optimizer.apply(gradients, self._network.trainable_variables)

    metrics = {
        'loss': loss,
        'critic_loss': tf.reduce_mean(critic_loss),
        'entropy_loss': tf.reduce_mean(entropy_loss),
        'policy_gradient_loss': tf.reduce_mean(policy_gradient_loss),
        'log_probs': tf.reduce_mean(log_prob_actions),
        'learning_rate': self._learning_rate,
        'advantages': tf.reduce_mean(advantages),
        'discounted_returns': tf.reduce_mean(discounted_returns),
        'entropy': tf.reduce_mean(pi.entropy())
    }

    return metrics, gradients, logits

  def update(self, last: bool = False):
    """Perform a meta-training update."""
    for i in range(self._minibatch_size):
      # Retrieve a batch of data from replay.
      data = self._queue.sample()
      data = tree_utils.fast_map_structure(lambda x: tf.convert_to_tensor(x), data)

      # Do a batch of SGD.
      results, gradients, logits = self._step(data=data)

      # Check gradients.
      #for g, v in zip(gradients, self._network.trainable_variables):
      #  name = v.name.replace('/', '-')
      #  results.update({name: tf.reduce_mean(g)})

      # Compute elapsed time.
      timestamp = time.time()
      elapsed_time = timestamp - self._timestamp if self._timestamp else 0
      self._timestamp = timestamp

      # Update our counts and record it.
      counts = self._counter.increment(steps=1, walltime=elapsed_time)
      results.update(counts)

      # Compute KL Divergence.
      pi = tfd.Categorical(logits=logits[:-1])
      if counts['steps'] > 1:
        kl_divergence = tf.reduce_mean(pi.kl_divergence(self._pi_old))
        results.update({'kl_divergence': kl_divergence})
      if counts['steps'] == 1:
        results.update({'kl_divergence': 0.0})
      self._pi_old = pi

      # Update learning rate.
      self._learning_rate.assign(self._lr_scheduler(step=counts['steps']))

      # Snapshot and attempt to write logs.
      if hasattr(self, '_snapshotter'):
        self._snapshotter.save()

      if self._logger:
        self._logger.write(results)


class FastRLAgent(agents.Agent):
  """fast RL Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.RNNCore,
      queue: adder.Adder,
      sample_size: int = 1000,
      verbose_level: Optional[int] = 0
  ):
    tf2_utils.create_variables(network, [environment_spec.observations])
    acme_utils.reinitialize_weights(network=network)

    # Actor and *empty* learner.
    actor = acting.FastRLActor(environment_spec, network, queue, verbose_level)
    learner = learning.FastRLLearner(environment_spec)

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=0,
        observations_per_step=sample_size)
