from typing import Optional, List

import os
import numpy as np
import tensorflow as tf

from acme import specs
from acme.utils import paths
from acme import specs, types


def _sample_int_layer_wise(nbatch, high, low):
  assert(high.ndim == 1 and low.ndim == 1)
  ndim = len(high)
  out_list = []
  for d in range(ndim):
    out_list.append( np.random.randint(low[d], high[d]+1, (nbatch,1 ) ) )
  return np.concatenate(out_list, axis=1)

def _sample_layer_wise(nbatch, high, low):
    assert len(high.shape) == 1 and len(low.shape) ==1
    nsample = len(high)
    base = np.random.rand( nbatch, nsample )
    return base * (high - low) + low

def compute_mapping(subset_list, superset_list):
  superset = set(superset_list)
  subset = set(subset_list)
  assert subset.issubset(superset), "error: subset relation is broken"

  sub_to_super = np.zeros(len(subset), dtype=np.int)
  super_to_sub = np.full(shape=(len(superset)), fill_value=-1, dtype=np.int)
  for sub_index, item in enumerate(subset_list):
    super_index = superset_list.index(item)
    sub_to_super[sub_index] = super_index
    super_to_sub[super_index] = sub_index
  return sub_to_super, super_to_sub


def get_index_from_pool(pool_ids: np.ndarray, pool_to_index: np.ndarray):
  # (Batched) operation of index = pool_to_index[pool_ids]
  if pool_ids.ndim == 1 and pool_to_index.ndim == 1:
    pool_index = pool_to_index[pool_ids]
  elif pool_ids.ndim == 2 and pool_to_index.ndim == 2:
    assert pool_ids.shape[0] == pool_to_index.shape[0], 'Error: batch dimension is different!'
    pool_index = np.take_along_axis(arr=pool_to_index, indices=pool_ids, axis=1)
  else:
    assert False, 'Error: the shape of "pool_ids" and "pool_to_index" should be both either a) 1-dimensional or b) 2-dimensional!'
  return pool_index

def get_pool_from_index(indices: np.ndarray, index_to_pool: np.ndarray):
  # (Batched) operation of index = index_to_pool[pool_ids]
  if indices.ndim == 1 and index_to_pool.ndim == 1:
    pool_ids = index_to_pool[indices]
  elif indices.ndim == 2 and index_to_pool.ndim == 2:
    assert indices.shape[0] == index_to_pool.shape[0], 'Error: batch dimension is different!'
    pool_ids = np.take_along_axis(arr=index_to_pool, indices=indices, axis=1)
  else:
    assert False, 'Error: the shape of "indices" and "index_to_pool" should be both either a) 1-dimensional or b) 2-dimensional!'
  return pool_ids

def map_index_arr_to_pool_arr(arr_by_index: np.ndarray, pool_to_index: np.ndarray, default_val: int = 0):
  """
    Maps from "array indexed by 'index'" to "array indexed by 'pool-id'".
    E.g., arr_by_index = [1, 2, 3, 4], pool_to_index = [0, -1, -1, 3, -1, 2, 1] ==> arr_by_pool = [1, 0, 0, 4, 0, 3, 2]
    assert no duplication except -1 in pool_to_index
    max(pool_to_index) + 1 == len(arr_by_index) == 13 (for playground)
  """
  assert not isinstance(arr_by_index, tf.Tensor)
  if isinstance(arr_by_index, np.ndarray) and arr_by_index.ndim == 2:
    batch_size = arr_by_index.shape[0]
    if pool_to_index.ndim == 1:
      pool_to_index = np.expand_dims(pool_to_index, 0)
    extended_arr = np.append(arr_by_index, np.full((batch_size, 1), default_val).astype(arr_by_index.dtype), axis=-1)
    arr_by_pool = np.take_along_axis(extended_arr, pool_to_index, axis=-1)
  else:
    extended_arr = np.append(arr_by_index, default_val)
    arr_by_pool = extended_arr[pool_to_index]
  return arr_by_pool

def batched_mapping_expand(arr: np.ndarray, mapping: np.ndarray, default_val: int = 0):
  return map_index_arr_to_pool_arr(arr_by_index=arr, pool_to_index=mapping, default_val=default_val)

def map_pool_arr_to_index_arr(arr_by_pool: np.ndarray, index_to_pool: np.ndarray):
  # assert no duplication except -1 in index_to_pool
  # max(index_to_pool) + 1 == len(arr_by_pool) == 16 (for playground)
  if arr_by_pool.ndim == 2:
    if index_to_pool.ndim == 1:
      index_to_pool = np.expand_dims(index_to_pool, 0)
    arr_by_index = np.take_along_axis(arr_by_pool, index_to_pool, axis=-1)
  elif arr_by_pool.ndim == 3:
    while index_to_pool.ndim < 3:
      index_to_pool = np.expand_dims(index_to_pool, 0)
    arr_by_index = np.take_along_axis(arr_by_pool, index_to_pool, axis=-1)
  elif arr_by_pool.ndim == 1:
    #assert arr_by_index.ndim == 1
    arr_by_index = arr_by_pool[index_to_pool]
  else:
    assert False, f"arr_by_pool.ndim should be either 1 or 2 but got {arr_by_pool.ndim}"
  return arr_by_index

def transform_obs(observation: types.NestedArray, index_to_pool: np.ndarray):
  indexed_obs = {}
  for key, val in observation.items():
    if key in ['mask', 'completion', 'eligibility']:
      indexed_obs[key] = map_pool_arr_to_index_arr(arr_by_pool=val, index_to_pool=index_to_pool)
    else:
      indexed_obs[key] = val
  return indexed_obs

def to_multi_hot(index_tensor, max_dim):
  # number-to-onehot or numbers-to-multihot
  if len(index_tensor.shape) == 1:
    out = (np.expand_dims(index_tensor, axis=1) == \
           np.arange(max_dim).reshape(1, max_dim))
  else:
    out = (index_tensor == np.arange(max_dim).reshape(1, max_dim))
  return out

def sample_subtasks(
    rng: np.random.RandomState,
    pool: List[str],
    minimum_size: int,
    maximum_size: Optional[int] = None,
    replace: bool = False
) -> List[str]:
  if maximum_size is not None:
    assert maximum_size <= len(pool), 'Invalid maximum_size.'
  maximum_size = maximum_size or len(pool)
  random_size = rng.randint(minimum_size, maximum_size + 1)
  sampled_subtasks = rng.choice(pool, size=random_size, replace=replace)
  return list(sampled_subtasks)

def add_sampled_nodes(
    graph: 'logic_graph.SubtaskLogicGraph',
    rng: np.random.RandomState,
    pool: List[str],
    minimum_size: int = 1,
    maximum_size: Optional[int] = None
):
  valid_nodes = list(graph.nodes)

  # Sample distractors.
  distractors = sample_subtasks(
      rng=rng,
      pool=pool,
      minimum_size=minimum_size,
      maximum_size=maximum_size
  )

  distractors_added = []
  for distractor in distractors:
    if distractor not in graph:
      distractor_at = np.random.choice(valid_nodes)
      graph[distractor] = distractor_at
      distractors_added.append(distractor)
  return graph, distractors_added


from dataclasses import dataclass

@dataclass
class SubtaskGraph:
  env_id: str
  task_index: int
  num_data: np.ndarray
  numP: np.ndarray
  numA: np.ndarray
  index_to_pool: np.ndarray
  pool_to_index: np.ndarray
  subtask_reward: np.ndarray
  W_a: np.ndarray
  W_o: np.ndarray
  ORmat: np.ndarray
  ANDmat: np.ndarray
  tind_by_layer: list
  #kmap: list

  def __init__(
      self,
      env_id: Optional[str] = None,
      task_index: Optional[int] = None,
      num_data: Optional[int] = 0,
      numP: Optional[np.ndarray] = None,
      numA: Optional[np.ndarray] = None,
      index_to_pool: Optional[np.ndarray] = None,
      pool_to_index: Optional[np.ndarray] = None,
      subtask_reward: Optional[np.ndarray] = None,
      W_a: Optional[np.ndarray] = None,
      W_o: Optional[np.ndarray] = None,
      ORmat: Optional[np.ndarray] = None,
      ANDmat: Optional[np.ndarray] = None,
      tind_by_layer: Optional[list] = None,
  ):
    self.env_id = env_id
    self.task_index = task_index
    self.num_data = num_data
    self.numP = numP
    self.numA = numA
    self.W_a = W_a
    self.W_o = W_o
    self.index_to_pool = index_to_pool
    self.pool_to_index = pool_to_index
    self.ORmat = ORmat
    self.ANDmat = ANDmat
    self.subtask_reward = subtask_reward
    self.tind_by_layer = tind_by_layer

  """def initialize_from_kmap(
      self,
      kmap,
  ):
    self.ANDmat = np.zeros(shape=(0))
    self.ORmat = np.zeros(shape=(0))
    self.W_a = []
    self.W_o = []
    self.numP, self.numA = [], []
    self.kmap = kmap"""
  
  def fill_edges(
      self,
      ANDmat: Optional[np.ndarray] = None,
      ORmat: Optional[np.ndarray] = None,
      W_a: Optional[np.ndarray] = None,
      W_o: Optional[np.ndarray] = None,
      tind_by_layer: Optional[list] = None,
  ):
    self.ANDmat = ANDmat
    self.ORmat = ORmat
    self.W_a = W_a
    self.W_o = W_o
    self.tind_by_layer = tind_by_layer
    self.numP, self.numA = [], []
    for tind_list in tind_by_layer:
      self.numP.append(len(tind_list))
    for wa_row in self.W_a:
      self.numA.append(wa_row.shape[0])

class dotdict(dict):
  """dot.notation access to dictionary attributes"""
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

class OptionSubtaskGraphVisualizer:
  def __init__(self):
    pass

  def render_and_save(self, g: 'graphviz.Digraph', path: str):
    g.render(filename=path)
    print('Saved graph @', path)
    return self

  def make_digraph(self) -> 'graphviz.Digraph':
    from graphviz import Digraph
    dot = Digraph(comment='subtask graph', format='pdf')
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(nodesep="0.2", ranksep="0.3")
    dot.node_attr.update(fontsize="14", fontname='Arial')
    return dot

  SUBTASK_NODE_STYLE = dict(shape='oval', height="0.2", width="0.2", margin="0")
  FEATURE_NODE_STYLE = dict(shape='oval', height="0.2", width="0.2", margin="0", rank="min")
  OPTION_NODE_STYLE = dict(shape='rect', height="0.2", width="0.2", margin="0.03")
  OPERATOR_NODE_STYLE = dict(shape='rect', style='filled',
                             height="0.15", width="0.15", margin="0.03")

  def visualize(self, cond_kmap_set, effect_mat,
      subtask_label: List[str], option_label: List[str]) -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_kmap_set: A sequence of eligibility CNF notations.
        cond_kmap_set[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    dot = self.make_digraph()
    omitted_option_count = 0
    for ind, label in enumerate(option_label):
      subtask_indices = effect_mat[ind].nonzero()[0]
      Kmap_tensor = cond_kmap_set[ind]
      if isinstance(Kmap_tensor, np.ndarray) or len(subtask_indices)> 0:
        dot.node('OPTION'+str(ind), label, **self.OPTION_NODE_STYLE) # visualize option node that has either effect or precondition
      else:
        omitted_option_count += 1
    print(f'#omitted options in the visualization={omitted_option_count}')

    for ind, label in enumerate(option_label):
      Kmap_tensor = cond_kmap_set[ind]
      if isinstance(Kmap_tensor, np.ndarray):
        numA, feat_dim = Kmap_tensor.shape
        for aind in range(numA):
          anode_name = 'AND'+str(ind)+'_'+str(aind)
          dot.node(anode_name, "&", **self.OPERATOR_NODE_STYLE)
          # OR-AND
          dot.edge(anode_name, 'OPTION'+str(ind))
          sub_indices = Kmap_tensor[aind, :].nonzero()[0]
          for sub_ind in sub_indices:
            if subtask_label[sub_ind].startswith('f'):
              style = self.FEATURE_NODE_STYLE
            else:
              style = self.SUBTASK_NODE_STYLE
            dot.node('SUBTASK'+str(sub_ind), subtask_label[sub_ind], **style)
            sub_ind = sub_ind.item()
            target = 'SUBTASK'+str(sub_ind)

            if Kmap_tensor[aind, sub_ind] > 0: #this looks wrong but it is correct since we used '>' instead of '<='.
              # AND-OR
              dot.edge(target, anode_name)
            else:
              dot.edge(target, anode_name, style="dashed")

    for option_ind, option_label in enumerate(option_label):
      subtask_indices = effect_mat[option_ind].nonzero()[0]
      for subtask_ind in subtask_indices:
        from_node = 'OPTION'+str(option_ind)
        to_node = 'SUBTASK'+str(subtask_ind)
        if subtask_label[subtask_ind].startswith('f'):
          style = self.FEATURE_NODE_STYLE
        else:
          style = self.SUBTASK_NODE_STYLE
        dot.node('SUBTASK'+str(subtask_ind), subtask_label[subtask_ind], **style)
        if effect_mat[option_ind, subtask_ind] > 0.5:
          dot.edge(from_node, to_node)
        else:
          dot.edge(from_node, to_node, style="dashed")
    return dot

  def _count_children(self, sub_indices, target_indices):
    count = 0
    for sub_ind in sub_indices:
      sub_ind = sub_ind.item()
      if sub_ind < 39 and sub_ind in target_indices:
        count += 1
    return count

class GraphVisualizer:
  def __init__(self):
    pass

  def set_num_subtasks(self, num_subtasks: int):
    self._num_subtasks = num_subtasks

  def render_and_save(self, g: 'graphviz.Digraph', path: str):
    g.render(filename=path)
    print('Saved graph @', path)
    return self

  def make_digraph(self) -> 'graphviz.Digraph':
    from graphviz import Digraph
    dot = Digraph(comment='subtask graph', format='pdf')
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(nodesep="0.2", ranksep="0.3")
    dot.node_attr.update(fontsize="14", fontname='Arial')
    return dot

  SUBTASK_NODE_STYLE = dict()
  OPERATOR_NODE_STYLE = dict(shape='rect', style='filled',
                             height="0.15", width="0.15", margin="0.03")

  def visualize_logicgraph(self, g: 'psgi.envs.logic_graph.SubtaskLogicGraph'
                           ) -> 'graphviz.Digraph':
    import psgi.envs.logic_graph
    LogicOp = psgi.envs.logic_graph.LogicOp

    dot = self.make_digraph()
    def _visit_node(node: LogicOp, to: str, has_negation=False):
      # TODO: This access private properties of LogicOp too much.
      # definitely we should move this to logic_graph?
      if node._op_type == LogicOp.TRUE:
        #v_true = f'true_{id(node)}'
        #dot.edge(v_true, to, style='filled')
        pass
      elif node._op_type == LogicOp.FALSE:
        v_false = f'_false_'
        dot.edge(v_false, to, style='filled', shape='rect')
      elif node._op_type == LogicOp.LEAF:
        leaf = node._children[0]
        dot.edge(leaf.name, to, style=has_negation and 'dashed' or '')
      elif node._op_type == LogicOp.NOT:
        op: LogicOp = node._children[0]
        _visit_node(op, to=to, has_negation=not has_negation)
      elif node._op_type == LogicOp.AND:
        v_and = f'and_{to}_{id(node)}'
        dot.node(v_and, "&", **self.OPERATOR_NODE_STYLE)
        dot.edge(v_and, to, style=has_negation and 'dashed' or '')
        for child in node._children:
          _visit_node(child, to=v_and)
        pass
      elif node._op_type == LogicOp.OR:
        v_or = f'or_{to}_{id(node)}'
        dot.node(v_or, "|", **self.OPERATOR_NODE_STYLE)
        dot.edge(v_or, to, style=has_negation and 'dashed' or '')
        for child in node._children:
          _visit_node(child, to=v_or)
      else:
        assert False, str(node._op_type)

    for name, node in g._nodes.items():
      assert name == node.name
      dot.node(node.name)
      _visit_node(node.precondition, to=name)
    return dot

  def visualize(self, cond_kmap_set, subtask_layer,
                subtask_label: List[str]) -> 'graphviz.Digraph':
    """Visualize the subtask graph given its eligibility logic expression.

    Args:
      cond_kmap_set: A sequence of eligibility CNF notations.
        cond_kmap_set[i] = list of clauses, each of which represents
        a vector c where c[j] consists of either {-1, 0, 1}.
        For example, eligibility[i] = c1 OR c2 where
        e.g. c1 = [0, 0, 1, 0, -1]:  (#2) AND (NOT #4)
             c2 = [0, 1, 0, -1]   :  (#1) AND (NOT #3)
    """
    dot = self.make_digraph()

    #cond_kmap_set
    for ind in range(self._num_subtasks):
      if subtask_layer[ind] > -2:
        label = subtask_label[ind]
        dot.node('OR'+str(ind), label, shape='rect', height="0.2", width="0.2", margin="0")
        Kmap_tensor = cond_kmap_set[ind]
        if Kmap_tensor is None:
          continue
        numA, feat_dim = Kmap_tensor.shape
        for aind in range(numA):
          anode_name = 'AND'+str(ind)+'_'+str(aind)
          dot.node(anode_name, "&", shape='rect', style='filled',
                   height="0.15", width="0.15", margin="0.03")
          # OR-AND
          dot.edge(anode_name, 'OR'+str(ind))
          sub_indices = Kmap_tensor[aind, :].nonzero()[0]
          for sub_ind in sub_indices:
            sub_ind = sub_ind.item()
            target = 'OR'+str(sub_ind)

            if Kmap_tensor[aind, sub_ind] > 0: #this looks wrong but it is correct since we used '>' instead of '<='.
              # AND-OR
              dot.edge(target, anode_name)
            else:
              dot.edge(target, anode_name, style="dashed")
    return dot

  def _count_children(self, sub_indices, target_indices):
    count = 0
    for sub_ind in sub_indices:
      sub_ind = sub_ind.item()
      if sub_ind < 39 and sub_ind in target_indices:
        count += 1
    return count

'''
def batch_bin_encode_string(bin_tensor):
  assert bin_tensor.dtype == np.bool
  feat_dim = bin_tensor.shape[-1]
  if feat_dim > 63:
    dim = bin_tensor.ndim
    if dim == 2:
      output = [''.join(str(int(x)) for x in v) for v in bin_tensor]
      return output

    elif dim == 1:
      output = np.array2string(bin_tensor)
      return output
    else:
      raise ValueError("dim = %s" % dim)
  else:
    return batch_bin_encode_64(bin_tensor)
'''

def batch_bin_encode(bin_tensor):
  if bin_tensor.ndim==2:
    return [hash(row.tobytes()) for row in bin_tensor]
  elif bin_tensor.ndim==1:
    return hash(bin_tensor.tobytes())
"""
def batch_bin_encode(bin_tensor):
  assert bin_tensor.dtype == np.bool
  feat_dim = bin_tensor.shape[-1]
  if feat_dim > 63:
    unit = 60
    num_iter = (feat_dim-1) // unit + 1
    if bin_tensor.ndim == 2 and bin_tensor.shape[0] > 1:
      code_batch=[]
      # parse by 50 dim
      bias = 0
      for i in range(num_iter):
        ed = min(feat_dim, bias + unit)
        code_column = batch_bin_encode_64(bin_tensor[:, bias:ed]) # np.arr(batch_size x unit) -> [int] * batch_size
        code_batch.append(code_column)
        bias = ed
      return np.stack(code_batch).T

    elif bin_tensor.ndim == 1 or (bin_tensor.ndim == 2 and bin_tensor.shape[0] == 1):
      if bin_tensor.ndim == 2:
        bin_vec = bin_tensor.squeeze()
      else:
        bin_vec = bin_tensor
      code_batch=[]
      # parse by 50 dim
      bias = 0
      for i in range(num_iter):
        ed = min(feat_dim, bias + unit)
        code = batch_bin_encode_64(bin_vec[bias:ed]) # np.arr(batch_size x unit) -> [int] * batch_size
        code_batch.append(int(code))
        bias = ed
      if bin_tensor.ndim == 2:
        return [tuple(code_batch)]
      else:
        return tuple(code_batch)
    else:
      raise ValueError(f"should be {bin_tensor.ndim} <= 2")
  else:
    return batch_bin_encode_64(bin_tensor)


def batch_bin_encode_64(bin_tensor):
  # bin_tensor: Nbatch x dim
  assert isinstance(bin_tensor, np.ndarray)
  assert bin_tensor.shape[-1] < 64, "Error, cannot handle large bin vector."
  assert bin_tensor.dtype == np.bool           # XXX graph generation has bug (ternary inputs)
  return bin_tensor.dot(
      (1 << np.arange(bin_tensor.shape[-1]))
  ).astype(np.int64)
"""
