from typing import List
import acme
import numpy as np

class MSGILearner(acme.Learner):
  """Meta-Learner for MSGI Agent within a single trial.

  Typically this is a fast-learning through ILP."""

  # TODO: Consider merging this with MSGIActor into MSGIAgent.

  def __init__(
      self,
      ilp: 'psgi.graph.ilp.ILP',
      grprop: 'psgi.graph.grprop.GRPropActor'):
    self._grprop = grprop
    self._ilp = ilp
    self._skip_flag = False
    self._prev_task_embedding = None

  @property
  def skip_test(self) -> bool:
    return self._skip_flag

  def step(self):
    """Perform fast-learning & update posterior. Called by (adapt)agent.update()
    For MSGI, it's subtask graph inference and update GRProp policy (test actor).
    """
    # Fast learning
    task_embedding = self._ilp.infer_task()

    # Posterior update
    #self._skip_flag = self._is_graph_same(task_embedding, self._prev_task_embedding)
    self._skip_flag = False  # XXX do not skip test
    if not self._skip_flag:
      self._grprop.observe_task(task_embedding)
      self._prev_task_embedding = task_embedding
    else:
      print("skipping!")

  def run(self):
    raise NotImplementedError("This is not used in a distributed setting.")

  def get_variables(self, names: List[str]) -> List[acme.types.NestedArray]:
    raise NotImplementedError

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
