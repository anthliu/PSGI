from typing import List
import acme
import numpy as np
from psgi.agents import msgi
from psgi.agents.mtsgi import acting

class MTSGILearner(msgi.learning.MSGILearner):
  """Fast-Learner for MSGI Agent within a single trial.

  Typically this is a fast-learning through ILP."""

  # TODO: Consider merging this with MTSGIActor into MTSGIAgent.

  def __init__(
      self,
      ilp: 'psgi.graph.ilp.ILP',
      grprop: 'psgi.graph.grprop.GRProp',
      prior_grprop: 'psgi.graph.grprop.GRProp'):
    super().__init__(
      ilp=ilp,
      grprop=grprop,
    )
    self._prior_grprop = prior_grprop

  '''
  def step(self):
    """Perform fast-learning. For MSGI, it's subtask graph inference and update GRProp policy (test actor).
    And update posterior.
    Called by adapt_agent.update()
    """
    # Fast learning
    task_embedding = self._ilp.infer_task()

    # Posterior update
    self._grprop.observe_task(task_embedding)
    '''

  def run(self):
    raise NotImplementedError("This is not used in a distributed setting.")

  def get_variables(self, names: List[str]) -> List[acme.types.NestedArray]:
    raise NotImplementedError
