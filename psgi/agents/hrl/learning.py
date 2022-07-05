"""Learner for the advantage actor-critic (A2C) agent."""

import time
from typing import Dict, List, Mapping, Optional

import acme
from acme import specs
from acme.adders.reverb import Step
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tree_utils

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl

tfd = tfp.distributions


class A2CLearner(acme.Learner, tf2_savers.TFSaveable):
  """Learner for an n-step advantage actor-critic."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.RNNCore,
      dataset: tf.data.Dataset,
      learning_rate: float,
      discount: float = 0.99,
      decay: float = 0.99,
      epsilon: float = 1e-5,
      entropy_cost: float = 0.,
      baseline_cost: float = 1.,
      max_abs_reward: Optional[float] = None,
      max_gradient_norm: Optional[float] = None,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      snapshot_dir: Optional[str] = None,
  ):

    # Internalise, optimizer, and dataset.
    self._env_spec = environment_spec
    self._optimizer = snt.optimizers.RMSProp(
        learning_rate=learning_rate,
        decay=decay,
        epsilon=epsilon
    )

    self._network = network
    self._variables = network.variables
    # TODO(b/155086959): Fix type stubs and remove.
    #self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._dataset = dataset

    # Hyperparameters.
    self._discount = discount
    self._entropy_cost = entropy_cost
    self._baseline_cost = baseline_cost

    # Set up reward/gradient clipping.
    if max_abs_reward is None:
      max_abs_reward = np.inf
    if max_gradient_norm is None:
      max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
    self._max_abs_reward = tf.convert_to_tensor(max_abs_reward)
    self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=3.)

    if snapshot_dir is not None:
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'network': network}, time_delta_minutes=60.)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @property
  def state(self) -> Mapping[str, tf2_savers.Checkpointable]:
    """Returns the stateful objects for checkpointing."""
    return {
        'network': self._network,
        'optimizer': self._optimizer,
    }

  # XXX wrapping with tf.function causes a problem after we reset optimizer
  # because it tries to create new variable on a non-first call.
  #@tf.function
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

      pi = tfd.Categorical(logits=logits[:-1])

      # Optionally clip rewards.
      rewards = tf.clip_by_value(rewards,
                                 tf.cast(-self._max_abs_reward, rewards.dtype),
                                 tf.cast(self._max_abs_reward, rewards.dtype))
      values = tf.clip_by_value(values,
                                tf.cast(0.4, values.dtype),
                                tf.cast(0.4, values.dtype))
      # Compute actor & critic losses.
      discounted_returns = trfl.generalized_lambda_returns(
          rewards=tf.cast(rewards, tf.float32),
          pcontinues=tf.cast(self._discount*discounts, tf.float32),
          values=tf.cast(values[:-1], tf.float32),
          bootstrap_value=tf.cast(values[-1], tf.float32)
      )
      advantages = discounted_returns - values[:-1]

      #critic_loss = tf.square(advantages)
      policy_gradient_loss = trfl.policy_gradient(
          policies=pi,
          actions=actions,
          action_values=advantages
      )
      entropy_loss = trfl.policy_entropy_loss(pi).loss

      loss = tf.reduce_mean(policy_gradient_loss +
                            #self._baseline_cost * critic_loss +
                            self._entropy_cost * entropy_loss)

    # Compute gradients and optionally apply clipping.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    max_grad_norm = max([tf.norm(grad) if grad is not None else 0. for grad in gradients])
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    clipped_max_grad_norm = max([tf.norm(grad) if grad is not None else 0. for grad in gradients])
    self._optimizer.apply(gradients, self._network.trainable_variables)

    metrics = {
        'loss': loss,
        'Ent_loss': tf.reduce_mean(entropy_loss),
        'PG_loss': tf.reduce_mean(policy_gradient_loss),
        'mean_values': tf.reduce_mean(values),
        'mean_return': tf.reduce_mean(discounted_returns),
        #'max_grad_norm': max_grad_norm,
        #'clipped_grad_norm': clipped_max_grad_norm,
        #'Value_loss': tf.reduce_mean(critic_loss),
    }
    assert tf.math.is_finite(loss)

    return metrics

  def step(self):
    """Does a step of SGD and logs the results."""

    # Retrieve a batch of data from replay.
    data = self._dataset.sample()
    data = tree_utils.fast_map_structure(lambda x: tf.convert_to_tensor(x), data)

    # Do a batch of SGD.
    results = self._step(data=data)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    results.update(counts)

    # Snapshot and attempt to write logs.
    if hasattr(self, '_snapshotter'):
      self._snapshotter.save()
    self._logger.write(results)

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables)]
