"""Implements MSGI."""

from typing import Optional

import dm_env
from acme import specs
from acme.utils import loggers

from psgi import agents
from psgi.agents import meta_agent
from psgi.agents.msgi import acting, learning
from psgi.agents.base import RandomActor, CountBasedActor, UCBActor
from psgi.agents.grprop import GRPropActor
from psgi.agents.eval_actor import EvalWrapper

from psgi.graph.ilp import ILP


class MSGI(meta_agent.MetaAgent):
  """MSGI with a random/count-based adaptation policy."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      mode: str,
      num_adapt_steps: int,
      num_trial_splits: int,
      logger: loggers.Logger = None,
      # ILP
      branch_neccessary_first: bool = True,
      # MSGI
      exploration: str = 'ucb',
      ucb_temp: float = 10.0,
      # GRProp
      temp: float = None,
      w_a: float = None,
      beta_a: float = None,
      ep_or: float = None,
      temp_or: float = None,
      visualize: bool = False,
      directory: str = 'visualize',
      environment_id: Optional[str] = None,
      verbose_level: Optional[int] = 0,
      ):
    """
      Args:
        visualize: whether to visualize subtask graph inferred by ILP module.
        directory: directory for saving the subtask graph visualization.
        environment_id: environment name (e.g. toywob) for graph visualization.
    """
    # Internalize spec and MSGI modules.
    self._mode = mode
    self._update_period = num_adapt_steps // num_trial_splits
    self._environment_spec = environment_spec
    self._verbose_level = verbose_level

    # Define sub-modules shared between adapt-agent and test-actor
    self._ilp = ILP(
        environment_spec=environment_spec,
        num_adapt_steps=num_adapt_steps,
        branch_neccessary_first=branch_neccessary_first,
        visualize=visualize,
        directory=directory,
        environment_id=environment_id
    )
    self._grprop = GRPropActor(
        environment_spec=environment_spec,
        temp=temp,
        w_a=w_a, beta_a=beta_a,
        ep_or=ep_or, temp_or=temp_or
    )

    # Adaptation strategy.
    self._explore_strategy = exploration
    self._ucb_temp = ucb_temp

  @property
  def skip_test(self) -> bool:
    return self._learner.skip_test

  def instantiate_adapt_agent(self) -> acting.MSGIActor:
    """Instantiate 'fast' agent for adaptation phase. This fast agent will
      interact with the environment and perform fast learning & posterior update.
    """
    if self._explore_strategy == 'random':
      self._actor = RandomActor(self._environment_spec, verbose_level=self._verbose_level)
    elif self._explore_strategy == 'count':
      self._actor = CountBasedActor(self._environment_spec, verbose_level=self._verbose_level)
    elif self._explore_strategy == 'ucb':
      self._actor = UCBActor(
          environment_spec=self._environment_spec,
          temperature=self._ucb_temp,
          verbose_level=self._verbose_level
      )
    else:
      raise NotImplementedError

    # Create MSGI Actor for adaptation phase.
    adaptation_actor = acting.MSGIActor(
        actor=self._actor,
        ilp=self._ilp,
        verbose_level=self._verbose_level)

    # Create MSGI Learner. Perform fast-learning and posterior update
    learner = learning.MSGILearner(
        ilp=self._ilp,
        grprop=self._grprop)

    return agents.Agent(
        actor=adaptation_actor,
        learner=learner,
        min_observations=0,
        observations_per_step=self._update_period,
    )

  def instantiate_test_actor(self) -> GRPropActor:
    """Instantiate the actor for test phase. This actor will interact with
      the environment and evaluate the performance of the (adapted) fast agent.
    """
    # Return GRProp for test phase.
    return EvalWrapper(self._grprop, verbose_level=self._verbose_level)

  def reset_agent(self, environment: dm_env.Environment):
    """Reset 'fast' agent and prior upon samping a new task."""
    self._ilp.reset(environment)

    # reset adaptation actor.
    if self._explore_strategy == 'count':
      self._actor.observe_task(environment.task_embedding)
    elif self._explore_strategy == 'ucb':
      self._actor.observe_task(environment.task_embedding)

  def update(self):
    """Perform meta-training (slow-learning), and update prior
    """
    return

    # See also: EnvironmentMetaLoop's _meta_agent.update().
    # TODO: MetaAgent.update() could be called during adaptation.
    # In this case, we can have a property like `_should_meta_update`.

    # TODO: For MSGI-Adapt, etc. where meta-training happens,
    # the updates made here (e.g. parameter update and ILP policy) should be
    # reflected to the fast 'Agent'. Therefore, we need a reference to it.

  def update_adaptation_progress(self, current_split: int, max_split: int):
    """Update the adaptation progress with adaptation / testing actors
    """
    pass # do nothing
