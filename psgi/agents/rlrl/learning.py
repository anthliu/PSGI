"""Learner for the RL^2 agent."""

from typing import List, Mapping

import acme
from acme import specs
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers

import numpy as np


class FastRLLearner(acme.Learner, tf2_savers.TFSaveable):
  """(Empty) Learner for an Fast RL agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):
    # Internalise, optimizer, and dataset.
    self._env_spec = environment_spec

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @property
  def state(self) -> Mapping[str, tf2_savers.Checkpointable]:
    """Returns the stateful objects for checkpointing."""
    return {}

  def step(self):
    """Does a step of SGD and logs the results."""
    pass  # No update for fast-agent

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables)]
