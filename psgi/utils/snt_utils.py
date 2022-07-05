from typing import Optional, Text

from psgi.utils import acme_utils

from acme import types, specs
from acme.tf import networks

import sonnet as snt
import numpy as np
import tensorflow as tf


class RecurrentAttentionNN(snt.RNNCore):
  def __init__(
      self,
      name: Optional[Text] = None
  ):
    super().__init__(name=name)
    # TODO: make a flags for hidden layer dims.
    self.embed_dim = 32
    self.attn_heads = 3

    self.subtask_embed = snt.Linear(self.embed_dim * 2, name='subtask_embed')
    self.option_embed = snt.Linear(self.embed_dim * 2, name='option_embed')
    self.option_act_embed = snt.Linear(self.embed_dim, name='option_act_embed')

    self.query_w = snt.Linear(self.attn_heads * self.embed_dim, name='query_w')
    self.value_w = snt.Linear(self.embed_dim, name='value_w')
    self.key_w = snt.Linear(self.embed_dim, name='key_w')

    self.flat = snt.nets.MLP([64, 64], activation=tf.nn.leaky_relu, name="mlp_1")
    self.rnn = snt.DeepRNN([
        snt.nets.MLP([50, 50], activate_final=True, activation=tf.nn.leaky_relu, name="mlp_2"),
        snt.GRU(512, name="gru"),
        snt.nets.MLP([self.embed_dim], activate_final=True, activation=tf.nn.leaky_relu, name='mlp_3')
    ])

    self.value_score_w = snt.Linear(1, name='value_score_w')

  @tf.function
  def __call__(self, inputs, prev_state):
    _, subtask_ob, option_ob, completion, eligibility, flat_ob = acme_utils.preprocess_att_observation(inputs)
    # reshape observations
    completion = tf.one_hot(tf.cast(completion > 0.5, tf.int32), 2)
    eligibility = tf.one_hot(tf.cast(eligibility > 0.5, tf.int32), 2)
    subtask_ob = tf.reshape(subtask_ob, tf.constant([-1, subtask_ob.shape[1], subtask_ob.shape[2] * subtask_ob.shape[3]]))
    option_ob = tf.reshape(option_ob, tf.constant([-1, option_ob.shape[1], option_ob.shape[2] * option_ob.shape[3]]))

    # Attention over subtasks
    subtask_feat = tf.nn.leaky_relu(self.subtask_embed(subtask_ob))
    subtask_feat = tf.reshape(subtask_feat, tf.constant([-1, subtask_feat.shape[1], self.embed_dim, 2]))
    # subtask_feat : N x L x embed x 2
    # completion : N x L x 2
    # subtasks : N x L x embed
    subtasks = tf.einsum('nlei,nli->nle', subtask_feat, completion)

    # Attention over options
    option_feat = tf.nn.leaky_relu(self.option_embed(option_ob))
    option_feat = tf.reshape(option_feat, tf.constant([-1, option_feat.shape[1], self.embed_dim, 2]))
    # option_feat : N x L x embed x 2
    # eligibility : N x L x 2
    # options : N x L x embed
    options = tf.einsum('nlei,nli->nle', option_feat, eligibility)

    subtask_option_feat = tf.concat((subtasks, options), axis=1)

    # query = subtask_parameter_embeddings
    # keys = subtask_parameter_embeddings
    # values = completion
    query = self.query_w(tf.nn.leaky_relu(prev_state[0]))
    query = tf.reshape(query, tf.constant([-1, self.attn_heads, self.embed_dim]))
    values = self.value_w(subtask_option_feat)
    keys = self.key_w(subtask_option_feat)

    # query : N x heads x embed
    # values : N x L x embed
    # keys : N x L x embed
    attn = tf.nn.softmax(tf.einsum('nhe,nle->nhl', query, keys))
    attn = tf.einsum('nhl,nle->nhe', attn, values)
    attn = tf.reshape(attn, tf.constant([-1, self.attn_heads * self.embed_dim]))

    # Process flat observations.
    feat = self.flat(flat_ob)
    feat = tf.nn.leaky_relu(feat)

    total_feat = tf.concat((feat, attn), 1)
    outputs, new_state = self.rnn(total_feat, prev_state)
    value_score = tf.squeeze(self.value_score_w(outputs), -1)

    # attention over options
    option_act_feat = self.option_act_embed(option_ob)

    outputs = tf.expand_dims(outputs, 2)
    option_logits = tf.squeeze(tf.matmul(option_act_feat, outputs), 2)

    #assert not tf.math.reduce_any(tf.math.is_nan(option_logits))

    return (option_logits, value_score), new_state

  def initial_state(self, batch_size):
    return self.rnn.initial_state(batch_size)


class RecurrentNN(snt.RNNCore):
  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      name: Optional[Text] = None
  ):
    super().__init__(name=name)
    # TODO: make a flags for hidden layer dims.
    self.flat = snt.nets.MLP([64, 64], name="mlp_1")
    self.rnn = snt.DeepRNN([
        snt.nets.MLP([50, 50], activate_final=True, name="mlp_2"),
        snt.GRU(512, name="gru"),
        networks.PolicyValueHead(action_spec.num_values)
    ])

  @tf.function
  def __call__(self, inputs, prev_state):
    _, flat_ob = acme_utils.preprocess_observation(inputs)

    # Process flat observations.
    feat = self.flat(flat_ob)
    feat = tf.nn.relu(feat)
    outputs, new_state = self.rnn(feat, prev_state)
    return outputs, new_state

  def initial_state(self, batch_size):
    return self.rnn.initial_state(batch_size)


class CombinedNN(snt.RNNCore):
  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      name: Optional[Text] = None
  ):
    super().__init__(name=name)

    # Spatial
    self.conv1 = snt.Conv2D(16, 1, 1, data_format="NHWC", name="conv_1")
    self.conv2 = snt.Conv2D(32, 3, 1, data_format="NHWC", name="conv_2")
    self.conv3 = snt.Conv2D(64, 3, 1, data_format="NHWC", name="conv_3")
    self.conv4 = snt.Conv2D(32, 3, 1, data_format="NHWC", name="conv_4")
    self.flatten = snt.Flatten()

    self.fc1 = snt.Linear(256, name="fc_1")

    # Flat
    self.flat = snt.nets.MLP([64, 64], name="mlp_1")
    self.rnn = snt.DeepRNN([
        snt.nets.MLP([50, 50], activate_final=True, name="mlp_2"),
        snt.GRU(512, name="gru"),
       networks.PolicyValueHead(action_spec.num_values)
    ])

  @tf.function
  def __call__(self, inputs, prev_state):
    spatial_ob, flat_ob = acme_utils.preprocess_observation(inputs)

    # TODO: use gpu and switch data_format NHWC --> NCHW
    spatial_ob = tf.transpose(spatial_ob, perm=[0, 2, 3, 1])
    spatial_output = self.conv1(spatial_ob)
    spatial_output = tf.nn.relu(spatial_output)

    spatial_output = self.conv2(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)

    spatial_output = self.conv3(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)

    spatial_output = self.conv4(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)
    spatial_output = self.flatten(spatial_output)

    spatial_output = self.fc1(spatial_output)
    spatial_output = tf.nn.relu(spatial_output)

    # Process flat observations.
    flat_output = self.flat(flat_ob)
    flat_output = tf.nn.relu(flat_output)

    feat = tf.concat([spatial_output, flat_output], axis=-1)
    outputs, new_state = self.rnn(feat, prev_state)
    return outputs, new_state

  def initial_state(self, batch_size):
    return self.rnn.initial_state(batch_size)
