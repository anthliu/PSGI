# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def masked_softmax(mat, mask, dim=2, epsilon=1e-6):
  nb, nr, nc = mat.shape
  masked_mat = mat * mask
  masked_min = tf.tile(tf.reduce_min(masked_mat, axis=dim, keepdims=True), [1, 1, nc])
  masked_nonneg_mat = (masked_mat - masked_min) * mask
  max_mat = tf.tile(tf.reduce_max(masked_nonneg_mat, axis=dim, keepdims=True), [1, 1, nc])
  exps = tf.exp(masked_nonneg_mat - max_mat)
  masked_exps = exps * mask
  masked_sums = tf.reduce_sum(masked_exps, axis=dim, keepdims=True) + epsilon
  prob = masked_exps / masked_sums
  tf.debugging.Assert(tf.reduce_all(prob >= 0), [prob])
  return prob

@tf.function
def categorical_sampling(probs: np.ndarray):
  return tfd.Categorical(probs=probs).sample()

def fast_map_structure_flatten(func, structure, *flat_structure, **kwargs):
  expand_composites = kwargs.get('expand_composites', False)
  entries = zip(*flat_structure)
  return tf.nest.pack_sequence_as(
      structure, [func(*x) for x in entries],
      expand_composites=expand_composites)

def fast_map_structure(func, *structure, **kwargs):
  expand_composites = kwargs.get('expand_composites', False)
  flat_structure = [
      tf.nest.flatten(s, expand_composites=expand_composites) for s in structure
  ]
  entries = zip(*flat_structure)

  return tf.nest.pack_sequence_as(
      structure[0], [func(*x) for x in entries],
      expand_composites=expand_composites)
