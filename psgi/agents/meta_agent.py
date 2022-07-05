import abc
from typing import List, Optional

import dm_env
import numpy as np
import acme
import acme.agents
from acme import specs, types
from acme.agents.agent import Agent
from acme.utils import loggers


class MetaAgent(acme.VariableSource):
  """Meta-Agent Base class.

  see also: MetaEnvironmentLoop.
  """
  __metaclass__ = abc.ABC

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      mode: str,
      num_adapt_steps: int,
      num_trial_splits: int,
      environment_id: Optional[str] = None,
      logger: loggers.Logger = None):
    self._mode = mode
    self._environment_spec = environment_spec
    self._update_period = num_adapt_steps // num_trial_splits
    self._environment_id = environment_id
    self._logger = logger

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return self._learner.get_variables(names)

  @abc.abstractmethod
  def instantiate_adapt_agent(self) -> Agent:
    """Instantiate 'fast' agent for adaptation phase. This fast agent will
      interact with the environment and perform fast learning & posterior update.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def instantiate_test_actor(self) -> acme.Actor:
    """Instantiate the actor for test phase. This actor will interact with
      the environment and evaluate the performance of the (adapted) fast agent.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def reset_agent(self, environment: dm_env.Environment):
    """Reset 'fast' agent and prior upon samping a new task."""
    raise NotImplementedError

  @abc.abstractmethod
  def update(self):
    """Perform meta-training (slow-learning), and update prior
    """
    return

  @abc.abstractmethod
  def update_adaptation_progress(self, current_split, max_split):
    """Update the adaptation progress with adaptation / testing actors
    """
    raise NotImplementedError
