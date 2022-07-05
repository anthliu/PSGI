from typing import Dict

import numpy as np
import tree
import sonnet as snt
import tensorflow as tf

import acme
from acme.utils import tree_utils
from acme.agents import agent


_zeros_initializer = snt.initializers.Zeros()
_ones_initializer = snt.initializers.Ones()
_he_xavier_initializer = snt.initializers.VarianceScaling(
    scale=2.0,
    mode='fan_in',
    distribution='truncated_normal'
)


def reinitialize_weights(network: snt.Module):
  # Re-initialize network weights.
  for v in network.trainable_variables:
    # Batch normalization.
    if 'bn' in v.name:
      if 'scale' in v.name:
        v.assign(_ones_initializer(shape=v.shape, dtype=v.dtype))
      else:  # offset
        v.assign(_zeros_initializer(shape=v.shape, dtype=v.dtype))
    else:
      if 'b:0' not in v.name:  # weights
        v.assign(_he_xavier_initializer(shape=v.shape, dtype=v.dtype))
      else:  # bias
        v.assign(_zeros_initializer(shape=v.shape, dtype=v.dtype))

def preprocess_observation(ob: Dict[str, np.ndarray]):
  # Type conversion to float32.
  ob = tree_utils.fast_map_structure(lambda x: tf.cast(x, tf.float32), ob)

  # Apply logarithm to remaining steps.
  def avoid_inf(x: tf.Tensor, epsilon: float = 1e-7):
    return tf.where(tf.math.is_inf(x), epsilon, x)
  ob['remaining_steps'] = avoid_inf(tf.math.log(ob['remaining_steps']))
  ob['trial_remaining_steps'] = avoid_inf(tf.math.log(ob['trial_remaining_steps']))

  # Pop spatial observation
  spatial_ob = ob.pop('observation', None)
  ob.pop('subtask_param_embeddings')
  ob.pop('option_param_embeddings')
  ob.pop('parameter_embeddings')

  # Add dimension
  ob['remaining_steps'] = tf.expand_dims(ob['remaining_steps'], axis=-1)
  ob['trial_remaining_steps'] = tf.expand_dims(ob['trial_remaining_steps'], axis=-1)
  ob['termination'] = tf.expand_dims(ob['termination'], axis=-1)
  ob['step_done'] = tf.expand_dims(ob['step_done'], axis=-1)
  if 'action_mask' in ob:
    ob['action_mask'] = tf.expand_dims(ob['action_mask'], axis=-1)
  ob['option_success'] = tf.expand_dims(ob['option_success'], axis=-1)
  flat_ob = tf.concat(tree.flatten(ob), axis=-1)
  return spatial_ob, flat_ob

def preprocess_att_observation(ob: Dict[str, np.ndarray]):
  # Type conversion to float32.
  ob = tree_utils.fast_map_structure(lambda x: tf.cast(x, tf.float32), ob)

  # Apply logarithm to remaining steps.
  def avoid_inf(x: tf.Tensor, epsilon: float = 1e-7):
    return tf.where(tf.math.is_inf(x), epsilon, x)
  ob['remaining_steps'] = avoid_inf(tf.math.log(ob['remaining_steps']))
  ob['trial_remaining_steps'] = avoid_inf(tf.math.log(ob['trial_remaining_steps']))

  # Pop spatial observation
  spatial_ob = ob.pop('observation', None)
  subtask_ob = ob.pop('subtask_param_embeddings')
  option_ob = ob.pop('option_param_embeddings')
  completion = ob.pop('completion')
  eligibility = ob.pop('eligibility')
  _ = ob.pop('parameter_embeddings')

  # Add dimension
  ob['remaining_steps'] = tf.expand_dims(ob['remaining_steps'], axis=-1)
  ob['trial_remaining_steps'] = tf.expand_dims(ob['trial_remaining_steps'], axis=-1)
  ob['termination'] = tf.expand_dims(ob['termination'], axis=-1)
  ob['step_done'] = tf.expand_dims(ob['step_done'], axis=-1)
  if 'task_success' in ob:
    ob['task_success'] = tf.expand_dims(ob['task_success'], axis=-1)
  if 'action_mask' in ob:
    ob['action_mask'] = tf.expand_dims(ob['action_mask'], axis=-1)
  ob['option_success'] = tf.expand_dims(ob['option_success'], axis=-1)
  flat_ob = tf.concat(tree.flatten(ob), axis=-1)
  return spatial_ob, subtask_ob, option_ob, completion, eligibility, flat_ob
