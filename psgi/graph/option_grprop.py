"""Graph Reward Propagation (GRProp) policy implementation."""

from psgi.envs.predicate_node import OptionNode
from typing import Optional, List, Dict, Sequence
from collections import OrderedDict

from copy import deepcopy
import numpy as np
from acme import specs

from psgi.utils import tf_utils
from psgi.envs.logic_graph import LogicOp
from psgi.envs.predicate_graph import LiteralLogicGraph

import tensorflow as tf

class CycleGRProp:

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      temperature: float,
      w_p: float):
    # Set additional attributes.
    self._action_spec = environment_spec.actions
    self.max_task = self._action_spec.num_values
    self.max_depth = 10
    # Hyperparameter
    self.temperature = temperature
    self.weight_p = w_p
    self._graph_initialized = False

  @property
  def is_ready(self):
    return self._graph_initialized

  def init_graph(self, graphs: Sequence[LiteralLogicGraph]):
    """Initialize Wa_tensor, Wo_tensor, subtask_reward."""
    assert isinstance(graphs, list), "Loading graph from a file is not supported."

    def to_multi_hot(index_tensor, max_dim):
      out = (np.expand_dims(index_tensor, axis=1) == np.arange(max_dim).reshape(1, max_dim)).sum(axis=0, keepdims=True)
      return out

    self.debug_graphs = graphs
    self.debug_option_name_list = [graph.all_option_names for graph in graphs] # for debugging
    self.debug_feasible_option_name_list = [[node.name for node in graph.option_nodes] for graph in graphs] # for debugging
    self.debug_subtask_name_list = [[node.name for node in graph.subtask_nodes] for graph in graphs] # for debugging
    self._graph_initialized = True
    self.matrices = []
    self.subtask_reward_list = []
    self.batch_size = len(graphs)
    
    self._all_to_feasible_list = []
    self._feasible_to_all_list = []
    max_num_option = 0
    for graph in graphs:
      self._all_to_feasible_list.append(graph.all_to_feasible)
      self._feasible_to_all_list.append(graph.feasible_to_all)
      kmap_set = graph.to_kmap()
      subtask_reward = graph.subtask_reward
      option_nodes = graph.option_nodes
      subtask_nodes = graph.subtask_nodes

      subtask_name_to_index = graph.subtask_name_to_index
      assert len(subtask_nodes) == len(subtask_reward)
      self.subtask_reward_list.append(subtask_reward)
      
      num_option = len(option_nodes)
      num_subtask = len(subtask_name_to_index)
      if num_option > max_num_option:
        max_num_option = num_option

      # ANDmat / ORmat from kmap
      and_node_index_list_by_option = []
      and_vec_to_ind = OrderedDict()
      and_node_index_bias = 0
      for option_ind, node in enumerate(option_nodes):
        kmap = kmap_set[option_ind]
        if kmap is None:
          and_node_index_list_by_option.append([])
        else:
          # kmap: np.array( [numA, num_subtask] ) \in {-1, 0, 1}
          and_node_indices = []
          for and_vec in kmap:
            and_tuple = tuple(and_vec.tolist())
            if and_tuple not in and_vec_to_ind:
              and_vec_to_ind[and_tuple] = and_node_index_bias
              and_node_index = and_node_index_bias
              and_node_index_bias += 1
            else: # handle duplicates
              and_node_index = and_vec_to_ind[and_tuple]
            and_node_indices.append(and_node_index)
          and_node_index_list_by_option.append(and_node_indices)

      # ORmat
      num_and = len(and_vec_to_ind)
      ORmat = np.zeros((num_option, num_and)) # num_option x num_AND
      for index, and_node_index_list in enumerate(and_node_index_list_by_option):
        if len(and_node_index_list) > 0: # if no precondition, just leave it as an all-zero row in ORmat
          index_tensor = np.array(and_node_index_list)
          ORmat[index] = to_multi_hot(index_tensor=index_tensor, max_dim=num_and)

      # ANDmat
      ANDmat = np.stack(list(and_vec_to_ind.keys())) # num_AND x num_subtask
      ANDmat[ANDmat<0] = - 10
      denom_AND = np.maximum(np.clip(ANDmat, 0, 1).sum(axis=1, keepdims=True), 1) # num_AND x 1

      # effect
      children_by_option = [node.effect._children for node in option_nodes]
      # children_by_option = [LogicOp[(craft, wood, bracelet)], LogicOp[(craft, wood, sword)]]
      effect_dict_list = []
      effect_vec_list = []
      for effect_op_list in children_by_option:
        effect_vec = np.zeros((num_subtask))
        effect_dict = dict()
        for effect_op in effect_op_list:
          if effect_op._op_type == LogicOp.LEAF:
            value = 1
            weight = 1
            name = str(effect_op)
          else:
            value = 0
            weight = 0 # when propagating, we don't propogate negative effect since we assume GRProp will not execute such option. But this is inaccurate..
            name = str(effect_op._children[0])

          index = subtask_name_to_index[name]
          effect_dict[index] = value
          effect_vec[index] = weight
        effect_dict_list.append(effect_dict)
        effect_vec_list.append(effect_vec)
      effect_mat = np.stack(effect_vec_list).T
      matrix = {
        'ANDmat': ANDmat,
        'ORmat': ORmat,
        'effect_dict_list': effect_dict_list,
        'effect_mat': effect_mat,
        'denom_AND': denom_AND,
      }
      self.matrices.append(matrix)
    self.max_num_option = max_num_option
    self._all_to_feasible = np.stack(self._all_to_feasible_list)
    self._feasible_to_all = np.stack(self._feasible_to_all_list)

  def compute_option_score(self, comp, feasible_elig, subtask_reward, option_reward, matrix):
    """Computes the FD reward."""
    assert comp.ndim == 2
    base_progress, base_elig = self._compute_soft_progress(self.max_depth, comp, matrix)

    option_score = []
    effect_dict_list = matrix['effect_dict_list']
    assert len(effect_dict_list) == len(feasible_elig), "Shape error!" # feasible_elig
    for option_index, effect_dict in enumerate(effect_dict_list):
      if feasible_elig[option_index] == 1:
        #
        new_comp = np.copy(comp)
        for index, value in effect_dict.items():
          new_comp[index] = value
        progress, elig = self._compute_soft_progress(self.max_depth, new_comp, matrix)
        score = np.dot((progress - base_progress), subtask_reward) + np.dot((elig - base_elig), option_reward)
        option_score.append(score)
      else:
        option_score.append(0.) # Anyway, this option will never be executed since it's ineligible
    return option_score

  def _compute_soft_progress(self, num_iter, completion, matrix):
    w_p = self.weight_p
    ANDmat, ORmat = tf.constant(matrix['ANDmat'], dtype=tf.float32), tf.constant(matrix['ORmat'], dtype=tf.float32)
    denom_AND = tf.constant(matrix['denom_AND'], dtype=tf.float32)
    effect_mat = tf.constant(matrix['effect_mat'], dtype=tf.float32)   # num_subtask x num_option
    num_and, num_subtask = ANDmat.shape
    num_option, num_and_ = ORmat.shape
    assert num_and == num_and_, "shape of ANDmat and ORmat does not match"

    completion = tf.constant(completion, dtype=tf.float32)
    progress = completion # num_subtask
    #self.debug_progress(progress=progress, iter = -1)
    # SR: (correct up to here)
    for iter in range(num_iter):
      # Subtask -> option
      ANDout = tf.clip_by_value(tf.linalg.matmul(ANDmat, progress) / denom_AND, 0, 1) # num_and x 1
      soft_elig = tf.math.reduce_max(ORmat * tf.expand_dims(tf.transpose(ANDout), 1), axis=-1)
      # soft_elig: [1 x num_option]

      # Option -> subtask. propagate the maximum eligibility to the subtask progress
      progress_from_elig = tf.transpose(tf.math.reduce_max(effect_mat * tf.expand_dims(soft_elig, 1), axis=-1))
      # progress_from_elig in range(0, 1) since soft_elig in range(0, 1)
      if np.abs(progress_from_elig).sum() == 0:
        break
      progress = tf.maximum(progress, progress_from_elig * w_p) # num_subtask x 1
      #self.debug_progress(progress=progress, iter = iter)
    return tf.squeeze(progress).numpy(), tf.squeeze(soft_elig).numpy()

  def debug_progress(self, progress, iter):
    print(f'iteration = {iter}')
    for subtask_index in progress.nonzero()[0]:
      print(f"{self.debug_subtask_name_list[0][subtask_index]}: {progress[subtask_index]}")
      
  def get_raw_logits(
    self, 
    observation: Dict[str, np.ndarray], 
    subtask_reward_list: Optional[List] = None, 
    option_reward_list: Optional[List] = None, 
    temperature: Optional[float] = None):
    if temperature is None:
      temperature = self.temperature

    assert self._graph_initialized, \
      'Graph not initialized. "init_graph" must be called at least once before taking action.'
    completion_batch = observation['completion']
    eligibility_batch = observation['eligibility']
    if subtask_reward_list is None: # eval
      subtask_reward_list = self.subtask_reward_list
    if option_reward_list is None:
      option_reward_list = np.zeros_like(self._feasible_to_all)
    logits = np.zeros((self.batch_size, self.max_num_option))
    for i in range(self.batch_size):
      is_last_step = observation['termination'][i]
      feasible_to_all = self._feasible_to_all_list[i]
      matrix = self.matrices[i]
      subtask_reward = np.array(subtask_reward_list[i])
      option_reward = np.array(option_reward_list[i])
      
      comp = np.expand_dims(completion_batch[i], axis=-1)
      elig_feasible = np.take_along_axis(eligibility_batch[i], feasible_to_all, axis=-1)
      if is_last_step:
        logits[i] = elig_feasible
      elif np.all(subtask_reward == 0): # random
        logits[i] = elig_feasible
        print(f'Warning!! batch#{i} has not observed any rewarding subtask. Using random instead of GRProp!')
      else:
        #option_score = self.compute_option_score(comp, elig_feasible, subtask_reward, matrix)
        option_score = self.compute_option_score(comp, elig_feasible, subtask_reward, option_reward, matrix)
        logits[i] = temperature * np.array(option_score)
    return logits
