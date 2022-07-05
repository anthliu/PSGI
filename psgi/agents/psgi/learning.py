from collections import OrderedDict
from copy import deepcopy
from typing import List, Dict
import acme
import numpy as np

from psgi.envs.predicate_graph import PredicateLogicGraph
from psgi.utils.graph_utils import OptionSubtaskGraphVisualizer

class PSGILearner(acme.Learner):
  """Meta-Learner for PSGI Agent within a single trial.

  Typically this is a fast-learning through ILP."""

  # TODO: Consider merging this with PSGIActor into PSGIAgent.

  def __init__(
      self,
      environment_id: str,
      mode: str,
      pilp: 'psgi.graph.pilp.PILP',
      grprop_actor: 'psgi.graph.option_grprop.CycleGRProp',
      ):
    self._environment_id = environment_id
    self._mode = mode
    self._grprop_actor = grprop_actor
    self._pilp = pilp
    self._skip_flag = False
    self._prev_task_embedding = None
    self.step_count = 0
    self._prior_instantiated = False
    
  @property
  def skip_test(self) -> bool:
    return self._skip_flag
    
  @property
  def inferred_task(self):
    return self._prev_task_embedding

  def reset(self, environment): # called when sampling a new task
    self._parameters = environment.parameters
    pass
    #self._features = environment.features

  def step(self):
    """Perform fast-learning & update posterior. Called by (adapt)agent.update()
    For MSGI, it's subtask graph inference and update GRProp policy (test actor).
    """
    # Fast learning
    predicate_graphs = self._pilp.infer_task()

    # Posterior update
    self.step_count += 1
    self._prev_task_embedding = predicate_graphs

    if not self._skip_flag: # execution
      unrolled_graphs = []
      for predicate_graph, param_pool in zip(predicate_graphs, self._parameters):
        feature_dict = self._build_feature_dict(self._pilp._feature_set.parameter_embedding, self._pilp._feature_set.feature_labels, param_pool)
        graph, option_name_pool = predicate_graph.unroll_graph(param_pool, feature_dict)
        # after this, there should be no feature remaining in 'graph'
        unrolled_graphs.append(graph)
      self._grprop_actor.observe_task(unrolled_graphs)
      self.unrolled_graphs = unrolled_graphs
    else:
      print("skipping!")

  def run(self):
    raise NotImplementedError("This is not used in a distributed setting.")

  def get_variables(self, names: List[str]) -> List[acme.types.NestedArray]:
    raise NotImplementedError

  def _build_feature_dict(self, feature_mat, feature_labels, parameters):
    assert feature_mat.shape[0] == len(parameters)
    assert feature_mat.shape[1] == len(feature_labels)
    feature_dict = OrderedDict()
    for j, param in enumerate(parameters):
      for k, feature_label in enumerate(feature_labels):
        feature_node = f'({feature_label}, {param})'
        feature_dict[feature_node] = feature_mat[j, k] > 0.5
    return feature_dict

  def _is_graph_same(self, currs, prevs):
    if prevs is None:
      return False
    for curr, prev in zip(currs, prevs):
      is_same = _is_same(curr.__dict__, prev.__dict__, ['num_data'])
      if not is_same:
        return False
    return True

def _is_same(data1, data2, ignore_key=[]):
  if isinstance(data1, list) or isinstance(data1, tuple):
    if len(data1) != len(data2):
      return False
    for dat1, dat2 in zip(data1, data2):
      if not _is_same(dat1, dat2, ignore_key):
        return False
  elif isinstance(data1, dict):
    if data1.keys() != data2.keys():
      return False
    for key, val in data1.items():
      if key not in ignore_key and not _is_same(val, data2[key], ignore_key):
        return False
  elif isinstance(data1, np.ndarray):
    return np.allclose(data1, data2, rtol=1e-3)
  else:
    return data1 == data2
  return True
