import abc
from copy import deepcopy
import contextlib

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import os, sys
import numpy as np
from psgi.utils.predicate_utils import Predicate, Symbol
from psgi.envs.logic_graph import LogicOp, SubtaskLogicGraph, op_to_dict, _extract_leaf_nodes
from psgi.utils.graph_utils import SubtaskGraph, dotdict, GraphVisualizer, compute_mapping
from psgi.envs.predicate_node import _parse_name, PNode, OptionNode

class LiteralLogicGraph(SubtaskLogicGraph):
  """
  """
  _nodes: Dict[str, PNode]

  def __init__(self, name: str):
    self._name = name
    self._is_literal = True
    self._nodes = OrderedDict() # str -> PNode
    self._verbose = False
    self.freeze_flag = False
    self.all_option_names = []

  def __repr__(self):
    return f"LiteralLogicGraph[{self.name}, {len(self.nodes)} nodes]"

  # ==== Construction ====
  def _add_node(self, name: str, node: PNode):
    assert not self.freeze_flag, "Error! The graph is frozen, and cannot be changed"
    if name not in self._nodes:
      self._nodes[name] = node
    else:
      pass #SR: allow this for the partially parameterized subtasks
      #assert param not in self._nodes, f"Error while adding subtasks! duplicated node param: {param}"
      
  def add_subtasks(self, subtask_names: Sequence[str]):
    for name in subtask_names: # param = ('pickup', 'apple')
      node = PNode(name=name, node_type='subtask')
      assert not node.param.has_symbol(), "Error: literal graph cannot have any predicate subtask"
      self._add_node(name, node)

  def add_option(self, name: str, precondition: PNode, effect: PNode, is_feasible: bool):
    # create an option node
    assert name not in self._nodes, f"Error while adding options! duplicated node name: {name}"
    self.all_option_names.append(name)
    if is_feasible: # Only maintain feasible option as a node
      node = OptionNode(name=name, precondition=precondition.as_op(), effect=effect.as_op())
      assert not node.param.has_symbol(), "Error: literal graph cannot have any predicate option"
      self._add_node(name, node)
  
  def finalize_graph(self): # shared
    # Sort nodes
    names = self._nodes.keys()
    new_dict = OrderedDict()
    for name in sorted(names):
      new_dict[name] = self._nodes[name]
    self._nodes = new_dict

    # fillout parent-children
    for node in self.option_nodes:
      option_name = node.name

      # Effect
      subtask_names = _extract_leaf_nodes(node.effect)
      for subtask_name in subtask_names:
        # Add subtask as a parent of option
        node.add_parent(subtask_name)
        # Add option as a child of subtask
        self[subtask_name].add_child(option_name)
      
      # precondition
      subtask_names = _extract_leaf_nodes(node.precondition)
      for subtask_name in subtask_names:
        # Add subtask as a child of option
        node.add_child(subtask_name)
        # Add option as a parent of subtask
        self[subtask_name].add_parent(option_name)
      
    if self._is_literal:
      self.all_option_names = sorted(self.all_option_names)
      feasible_option_names = [node.name for node in self.option_nodes]
      feasible_to_all, all_to_feasible = compute_mapping(subset_list=feasible_option_names, superset_list = self.all_option_names)
      feasible_mask = [1 if name in feasible_option_names else 0 for name in self.all_option_names]
      self.feasible_mask = np.array(feasible_mask)
      self.feasible_to_all = feasible_to_all
      self.all_to_feasible = all_to_feasible
    #
    self.freeze_flag = True

  def defrost(self):
    self.freeze_flag = False
    # HACK
  # ==== Computation ====
  def compute_eligibility(self, completion: dict):
    elig = OrderedDict()
    for v in self.option_nodes:
      elig[v.name] = v._precondition.evaluate(completion)
    return elig
    
  def compute_effect(self, option: str, completion = None):
    assert option in self, f"Error. Option {option} does not exist"
    effect_op = self._nodes[option]._effect # LogicOp
    return op_to_dict(effect_op)

  #===== Transformation

  def to_neural_net(self): #literal
    #
    kmap_set = self.to_kmap() # 18 options x [1x42]
    # kmap_set is aligned with 'subtask_name'

    subtask_name_from_option = [node.name[3:] for node in self.option_nodes] # len == 10
    subtask_name = [node.name for node in self.subtask_nodes] # len == 18

    # 1. for each option in "subtask_name_from_option"
    mapped_kmap_set, mapping_from_subtask_to_option = self._match_subtask_to_option(kmap_set, subtask_name_from_option, subtask_name)
    # mapped_kmap_set now is aligned with 'subtask_name_from_option'
    subtask_ind_by_layer = self._locate_node(mapped_kmap_set)
    self.tind_by_layer = subtask_ind_by_layer
    matrices = self._kmap_to_mat(mapped_kmap_set, self.tind_by_layer)

    mapping_from_option_to_subtask = np.full((len(subtask_name)), -1, dtype=np.int)
    for ind, val in enumerate(mapping_from_subtask_to_option):
      mapping_from_option_to_subtask[val] = ind
    nn_graph = SubtaskGraph(
      env_id=self._name,
      task_index=0,
      num_data=0,
      index_to_pool=mapping_from_subtask_to_option,
      pool_to_index=mapping_from_option_to_subtask) # TODO: do we need this?
    nn_graph.fill_edges(**matrices)

    mapped_subtask_reward = np.take_along_axis(self.subtask_reward, mapping_from_subtask_to_option, axis=-1)
    nn_graph.subtask_reward = mapped_subtask_reward
    return nn_graph
  
  def _match_subtask_to_option(self, kmap_set, subtask_name_from_option, subtask_name): # literal - to_neural_net()
    assert len(kmap_set) == len(subtask_name_from_option)
    mapping_from_subtask_to_option = []
    for name in subtask_name_from_option:
      ind = subtask_name.index(name)
      mapping_from_subtask_to_option.append(ind)
    mapping_from_subtask_to_option = np.array([mapping_from_subtask_to_option])
    mapped_kmap_set = []
    for kmap in kmap_set:
      if kmap is not None: # kmap: [1x42]
        new_kmap = np.take_along_axis(kmap, mapping_from_subtask_to_option, axis=-1)
        mapped_kmap_set.append(new_kmap)
      else:
        mapped_kmap_set.append(None)
    return mapped_kmap_set, mapping_from_subtask_to_option.squeeze()
  
  def _locate_node(self, kmap_set): # literal - to_neural_net()
    num_nodes = len(kmap_set)
    max_num_layer = num_nodes
    first_layer = [ind for ind, kmap in enumerate(kmap_set) if kmap is None] # no precond --> first layer
    allocated = set(first_layer)
    subtask_ind_by_layer = [first_layer]
    for _ in range(max_num_layer):
      curr_layer = []
      for index, kmap in enumerate(kmap_set): # for kmap of precondition of each option
        if index not in allocated: # not yet allocated
          is_covered = True
          for and_node_vec in kmap:
            children_subtasks = and_node_vec.nonzero()[0]
            if not set(children_subtasks).issubset(allocated):
              is_covered = False
          if is_covered:
            curr_layer.append(index)
      assert len(curr_layer) > 0, f'Error! cycle exists and cannot determine the layer of nodes: {[index for index in range(num_nodes) if index not in allocated]}'
        
      subtask_ind_by_layer.append(curr_layer)
      allocated = allocated.union(curr_layer)
      if len(allocated) == len(kmap_set):
        break
    return subtask_ind_by_layer

  def _kmap_to_mat(self, kmap_set: np.ndarray, tind_by_layer):
    W_a, W_o, cand_tind = [], [], []
    num_prev_or = 0
    numA_all = 0
    num_subtasks = len(kmap_set)

    #1. fillout W_a/W_o
    for layer_ind in range(1, len(tind_by_layer)):
      num_prev_or = num_prev_or + len(tind_by_layer[layer_ind - 1])
      num_cur_or = len(tind_by_layer[layer_ind])
      W_a_layer, W_a_layer_padded = [], []
      cand_tind += tind_by_layer[layer_ind - 1]
      OR_table = [None] * num_subtasks
      numA = 0

      # fill out 'W_a_layer' and 'OR_table'
      for ind in tind_by_layer[layer_ind]:
        Kmap = kmap_set[ind]
        if Kmap is None:
          print('ind=', ind)
          print('tind_by_layer', tind_by_layer[layer_ind] )
          assert False, "Kmap should not be None"

        if len(Kmap) > 0:  # len(Kmap)==0 if no positive sample exists
          OR_table[ind] = []
          for j in range(Kmap.shape[0]):
            ANDnode = Kmap[j, cand_tind].astype(np.float)

            # see if duplicated
            duplicated_flag = False
            for row in range(numA):
              if np.all(np.equal(W_a_layer[row],ANDnode)):
                duplicated_flag = True
                and_node_index = row
                break

            if duplicated_flag == False:
              W_a_layer.append(ANDnode)
              cand_tind_tensor = np.array(cand_tind).astype(np.int32)
              assert cand_tind_tensor.shape[0] == ANDnode.shape[0]

              padded_ANDnode = np.zeros((num_subtasks))
              np.put_along_axis(padded_ANDnode, cand_tind_tensor, ANDnode, axis=0)
              #padded_ANDnode = np.zeros( (num_subtasks) ).scatter_(0, cand_tind_tensor, ANDnode)
              W_a_layer_padded.append(padded_ANDnode)
              OR_table[ind].append(numA) #add the last one
              numA = numA + 1
            else:
              OR_table[ind].append(and_node_index) #add the AND node

      if numA > 0:
        numA_all = numA_all + numA
        W_a_tensor = np.stack(W_a_layer_padded, axis=0)
        W_a.append(W_a_tensor)

      # fill out 'W_o_layer' from 'OR_table'
      W_o_layer = np.zeros((num_subtasks, numA))
      for ind in tind_by_layer[layer_ind]:
        OR_table_row = OR_table[ind]
        for j in range(len(OR_table_row)):
          and_node_index = OR_table_row[j]
          W_o_layer[ind][and_node_index] = 1

      W_o.append(W_o_layer)

    if self._verbose:
      print('W_a')
      for i in range(len(W_a)): print(W_a[i])
      print('W_o')
      for i in range(len(W_o)): print(W_o[i])

    #2. fillout ANDmat/ORmat
    assert len(W_a) > 0
    ANDmat = np.concatenate(W_a, axis=0)
    ORmat = np.concatenate(W_o, axis=1)

    if numA_all == 0 or num_subtasks == 0:
      print('num_subtasks=', num_subtasks)
      print('numA_all=', numA_all)
      print('kmap_set=', kmap_set)
      print('tind_by_layer=', tind_by_layer)
      # TODO: add helpful assert message
      assert False

    if self._verbose:
      print('Inference result:')
      print('ANDmat=', ANDmat)
      print('ORmat=', ORmat)
    matrices = dict(
      W_a=W_a,
      W_o=W_o,
      ANDmat=ANDmat,
      ORmat=ORmat,
      tind_by_layer=tind_by_layer,
    )
    return matrices

  def to_kmap(self): # shared
    # Assume each precondition is in SoP form
    feature_and_subtask_name_to_index = self.feature_and_subtask_name_to_index
    kmap_by_option = []
    for option_node in self.option_nodes: # For each option,
      pcond = option_node._precondition
      if pcond._op_type == LogicOp.TRUE:
        and_nodes = [] # no precondition
      elif pcond._op_type == LogicOp.OR:
        and_nodes = pcond._children
      elif pcond._op_type == LogicOp.AND:
        and_nodes = [pcond]
      elif pcond._op_type in (LogicOp.LEAF, LogicOp.NOT):
        and_nodes = [LogicOp(LogicOp.AND, children=[option_node])]

      kmap = []
      # 1. Iterate over each AND node
      for and_node in and_nodes:
        assert and_node._op_type in (LogicOp.AND, LogicOp.LEAF, LogicOp.NOT), "ERROR: precondition not in DNF form!"
        if and_node._op_type != LogicOp.AND: # wrap it with AND ode
          and_node = LogicOp(LogicOp.AND, children=[and_node])
        and_vec = self.and_node_to_vec(and_node, feature_and_subtask_name_to_index)
        kmap.append(and_vec)
      kmap_mat = np.stack(kmap) if len(kmap) > 0 else None
      kmap_by_option.append(kmap_mat)
    return kmap_by_option
  
  def and_node_to_vec(self, and_node, feature_and_subtask_name_to_index): # shared - by to_kmap()
    and_vec = np.zeros(shape=(len(feature_and_subtask_name_to_index)), dtype=np.int8)
    for child_op in and_node._children:
      if child_op._op_type == LogicOp.NOT:
        subtask_name = str(child_op._children[0])
        if subtask_name not in feature_and_subtask_name_to_index:
          import ipdb; ipdb.set_trace() 
        index = feature_and_subtask_name_to_index[subtask_name]
        and_vec[index] = -1
      elif child_op._op_type == LogicOp.LEAF:
        subtask_name = str(child_op)
        if subtask_name not in feature_and_subtask_name_to_index:
          print(f'ERROR!! {subtask_name} does not exist')
          import ipdb; ipdb.set_trace()
        index = feature_and_subtask_name_to_index[subtask_name]
        and_vec[index] = 1
      else:
        assert False, "ERROR: precondition not in DNF form!"
    return and_vec

  def add_reward(self, name: str, reward: float): # shared
    if name in self._nodes: # XXX: temporary
      self._nodes[name].set_reward(reward)

  def __getitem__(self, name) -> PNode: # shared
    return self._nodes[name]

  def __contains__(self, name): # shared
    return name in self._nodes

  @property
  def nodes(self) -> Sequence[PNode]: # shared
    return tuple(self._nodes.values())
  @property
  def subtask_nodes(self) -> Sequence[PNode]: # shared
    return tuple([node for node in self._nodes.values() if node.node_type == 'subtask'])
  @property
  def option_nodes(self) -> Sequence[PNode]: # shared
    return tuple([node for node in self._nodes.values() if node.node_type == 'option'])
  @property
  def option_name_to_index(self) -> Dict[str, int]: # shared
    assert self.freeze_flag, "Error: graph should be finalized before being used!"
    return {node.name: index for index, node in enumerate(self.option_nodes)}
  @property
  def subtask_name_to_index(self) -> Dict[str, int]: # shared
    assert self.freeze_flag, "Error: graph should be finalized before being used!"
    return {node.name: index for index, node in enumerate(self.subtask_nodes)}
  @property
  def feature_and_subtask_name_to_index(self) -> Dict[str, int]:
    assert self.freeze_flag, "Error: graph should be finalized before being used!"
    return {node.name: index for index, node in enumerate(list(self.subtask_nodes))}
  @property
  def subtask_reward(self):
    return [node.reward for node in self.subtask_nodes]

  # Utils
  def print_graph(self): # shared
    print('Feature:')
    for node in self.feature_nodes:
      print('- ', node.param)
    print('Subtasks:')
    for node in self.subtask_nodes:
      print('- ', node.param)
    print('Options:')
    for node in self.option_nodes:
      print(f"[ind] {node.param }")
      print(f" - precondition: {node.precondition}")
      print(f" - effect: {node.effect}")

  def visualize_graph(self) -> 'graphviz.Digraph': # shared
    from psgi.utils import graph_utils
    return graph_utils.GraphVisualizer().visualize_logicgraph(self)

class PredicateLogicGraph(LiteralLogicGraph):
  def __init__(self, name: str):
    super().__init__(name)
    self._is_literal = False
    self.literal_subtask_reward = None
  
  def __repr__(self):
    return f"PredicateLogicGraph[{self.name}, {len(self.nodes)} nodes]"
  
  @property
  def feature_nodes(self) -> Sequence[PNode]: # predicate
    return tuple([node for node in self._nodes.values() if node.node_type == 'feature'])
  @property
  def feature_and_subtask_name_to_index(self) -> Dict[str, int]: # predicate
    assert self.freeze_flag, "Error: graph should be finalized before being used!"
    return {node.name: index for index, node in enumerate(list(self.feature_nodes)+list(self.subtask_nodes))}

  # ==== Construction ====
  def add_features(self, features: Sequence[str]): #predicate
    for name in features: # name = ('pickup', Symbol(x), Symbol(y))
      node = PNode(name=name, node_type='feature')
      self._add_node(name, node)
  
  def add_subtasks(self, subtask_names: Sequence[str]):
    for name in subtask_names: # name = ('pickup', Symbol(x), Symbol(y))
      node = PNode(name=name, node_type='subtask')
      self._add_node(name, node)

  def add_option(self, name: str, precondition: PNode, effect: PNode):
    # create an option node
    assert name not in self._nodes, f"Error while adding options! duplicated node name: {name}"
    node = OptionNode(name=name, precondition=precondition.as_op(), effect=effect.as_op())
    self._add_node(name, node)
  
  # ==== Computation ====
  def compute_eligibility(self, completion: dict):
    raise NotImplementedError
  
  def compute_effect(self, option: str, completion = None):
    raise NotImplementedError
  # ==== Computation ====

  # ==== Conversion
  def to_neural_net(self):
    raise NotImplementedError
  
  def kmap_to_logic_op(self, kmap: np.ndarray, names: list, symbols: dict): # predicate - called by initialize_from_kmap
    def and_vec_to_logic_op(and_node_vec, names, symbols):
      subtask_inds = and_node_vec.nonzero()[0]
      and_op = LogicOp(LogicOp.TRUE)
      for subtask_ind in subtask_inds:
        name = names[subtask_ind]
        if and_node_vec[subtask_ind] > 0:
          and_op = and_op & self._nodes[name]
        else:
          and_op = and_op & (~self._nodes[name])
      return and_op
    if isinstance(kmap, bool):
      if kmap == False:
        return LogicOp(LogicOp.FALSE)
      elif kmap == True:
        return LogicOp(LogicOp.TRUE)
      else:
        raise ValueError
    elif kmap.ndim == 1: # effect kmap
      and_node_vec = kmap
      return and_vec_to_logic_op(and_node_vec, names, symbols)
    else:
      assert 2 == kmap.ndim, "Error: shape of kmap is wrong!"
      assert len(names) == kmap.shape[1], "Error: shape of names and kmap are inconsistent!"
      or_op = LogicOp(LogicOp.FALSE)
      for and_node_vec in kmap:
        and_op = and_vec_to_logic_op(and_node_vec, names, symbols)
        or_op = or_op | and_op
      return or_op
  
  def parse_inferred_name(self, name: str, symbols: dict): # predicate - called by initialize_from_kmap
    # name: "(pickup, A)" or "fisplace(A)"
    # symbols: {'A': Symbol(A), 'B': Symbol(B)}
    # Output: Predicate(['pickup', Symbol(A)]) or Predicate(['fisplace', Symbol(A)])
    is_feature = False
    if name[0] == 'f': # remove prefix 'f' for features
      is_feature = True
      name = name[1:]
    param = _parse_name(name)
    param = [symbols[par] if par in symbols else par for par in param]
    return Predicate(param), is_feature
  
  def unroll_graph(self, param_pool, feature) -> LiteralLogicGraph: # predicate
    g = LiteralLogicGraph(self._name)
    # Function for substituting op
    def substitute_in_op(op: LogicOp, assignment):
      if isinstance(op, PNode):
        op.param.substitute(assignment)
        op.update_name()
      elif op._children != op:
        for child in op._children:
          substitute_in_op(child, assignment)

    # Unroll subtask
    subtask_names = []
    for pred_node in self.subtask_nodes:
      predicate = pred_node.get_predicate_with_param(param_pool)
      prop_params = predicate.ground() # [Predicate('pickup', 'apple'), ...]
      prop_names = [prop.pretty() for prop in prop_params]
      subtask_names.extend(prop_names)
    g.add_subtasks(subtask_names=subtask_names)

    # Add subtask reward
    if self.literal_subtask_reward is not None:
      for node in g.subtask_nodes:
        g.add_reward(node.name, self.literal_subtask_reward[node.name])
    
    # Unroll options
    option_name_pool = []
    for pred_node in self.option_nodes:
      predicate = pred_node.get_predicate_with_param(param_pool)
      prop_params = predicate.ground() # [Predicate('pickup', 'apple'), ...]
      assignments = predicate.ground_assignments()
      #
      precond = pred_node.precondition
      effect = pred_node.effect
      for prop_param, assignment in zip(prop_params, assignments):
        option_name = prop_param.pretty()
        option_name_pool.append(option_name)
        prop_precond = deepcopy(precond)
        prop_effect = deepcopy(effect)
        substitute_in_op(prop_precond, assignment)
        substitute_in_op(prop_effect, assignment)
        # assign value to feature and perform reduction
        assigned_precond = prop_precond.assign(feature)
        is_feasible = not assigned_precond._op_type == LogicOp.FALSE
        g.add_option(option_name, assigned_precond, prop_effect, is_feasible)
    g.finalize_graph()
    return g, option_name_pool
    
  def update_subtask_reward(
      self,
      literal_subtask_names,
      literal_subtask_reward
  ):
    self.literal_subtask_reward = OrderedDict() # keep for unrolling later
    for name, reward in zip(literal_subtask_names, literal_subtask_reward):
      self.literal_subtask_reward[name] = reward
  
  def initialize_from_kmap(
      self,
      pcond_kmap_set,
      effect_kmap_set,
      subtask_and_feature_names,
      option_names,
      literal_subtask_names,
      literal_subtask_reward,
  ):
    assert len(option_names) == len(pcond_kmap_set), "Error: The 'option_names' is inconsistent with the shape of 'pcond_kmap_set'"
    assert len(subtask_and_feature_names) == max([kmap.shape[1] for kmap in pcond_kmap_set if isinstance(kmap, np.ndarray)]), "Error: The 'subtask_and_feature_names' is inconsistent with the shape of 'pcond_kmap_set'"
    assert effect_kmap_set.shape[0] == len(option_names), "Error: The 'option_names' is inconsistent with the shape of 'effect_kmap_set'"
    assert effect_kmap_set.shape[1] == len(subtask_and_feature_names), "Error: The 'subtask_and_feature_names' is inconsistent with the shape of 'effect_kmap_set'"
    assert len(literal_subtask_reward) == len(literal_subtask_names), "Error: literal_subtask_reward and literal_subtask_names have different dimension"
    
    symbols = {
      'A': Symbol(set(), 'A'),
      'B': Symbol(set(), 'B'),
    }
    subtasks, features = [], []
    for name in subtask_and_feature_names:
      if 'f_' in name:
        features.append(name)
      else:
        subtasks.append(name)
    self.add_subtasks(subtask_names=subtasks)
    self.add_features(features)

    for precond_mat, effect_mat, option_name in zip(pcond_kmap_set, effect_kmap_set, option_names):
      precond_op = self.kmap_to_logic_op(precond_mat, subtask_and_feature_names, symbols)
      effect_op = self.kmap_to_logic_op(effect_mat, subtask_and_feature_names, symbols)
      self.add_option(
        name=option_name,
        precondition=precond_op,
        effect=effect_op,
      )
    self.literal_subtask_reward = OrderedDict() # keep for unrolling later
    for name, reward in zip(literal_subtask_names, literal_subtask_reward):
      self.literal_subtask_reward[name] = reward
    self.finalize_graph()
  
