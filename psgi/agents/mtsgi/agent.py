"""Implements MTSGI."""

from typing import Optional

import dm_env
import numpy as np
from acme import specs, types
from acme.utils import loggers

from psgi import agents
from psgi.agents.psgi import acting, learning
from psgi.agents.base import RandomActor, UCBActor, MTUCBActor
from psgi.agents.grprop import GRPropActor
from psgi.agents.eval_actor import EvalWrapper
from psgi.utils import graph_utils, tf_utils

#from psgi.graph.bayesian_ilp import BayesILP
from psgi.graph.expand_ilp import ExpandILP
from psgi.graph.ilp import ILP


class MTSGI(agents.MSGI):
  """mtsgi with an adaptation policy."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      mode: str,
      num_adapt_steps: int,
      num_trial_splits: int,
      logger: loggers.Logger = None,
      # ILP
      branch_neccessary_first: bool = True,
      # psgi
      exploration: str = 'ucb',
      ucb_temp: float = 1.0,
      prior_sample_mode: str = 'coverage',
      posterior_mode: str = 'policy',
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
    self._update_period = num_adapt_steps // num_trial_splits
    self._environment_spec = environment_spec
    self._mode = mode
    self._verbose_level = verbose_level

    # Define sub-modules shared between adapt-agent and test-actor
    self._grprop = GRPropActor(
        environment_spec=environment_spec,
        temp=temp,
        w_a=w_a, beta_a=beta_a,
        ep_or=ep_or, temp_or=temp_or
    )

    # Multi-task learning
    self._prior_sample_mode = prior_sample_mode
    self._posterior_mode = posterior_mode
    #
    self.prior_pool = []
    self._current_prior = None
    #
    if self._posterior_mode == 'policy':
      ilp_class = ILP
      self._prior_grprop = GRPropActor(
          environment_spec=environment_spec,
          temp=temp,
          w_a=w_a, beta_a=beta_a,
          ep_or=ep_or, temp_or=temp_or
      )
    elif self._posterior_mode == 'ilp':
      ilp_class = ExpandILP
      self._prior_grprop = None
    else:
      raise ValueError(self._posterior_mode)
    self._ilp = ilp_class(
        environment_spec=environment_spec,
        num_adapt_steps=num_adapt_steps,
        branch_neccessary_first=branch_neccessary_first,
        visualize=visualize,
        directory=directory,
        environment_id=environment_id
    )
    self._learner = None
    self._test_actor = None

    # Adaptation strategy.
    self._explore_strategy = exploration
    self._ucb_temp = ucb_temp

    # For debugging mode
    self._count = 0

  @property
  def skip_test(self) -> bool:
    return self._learner.skip_test

  def update_adaptation_progress(self, current_split: int, max_split: int):
    """Update the adaptation progress with adaptation / testing actors
    """
    if isinstance(self._test_actor, acting.MixedActor):
      self._test_actor.update_adaptation_progress(current_split, max_split)

  def instantiate_adapt_agent(self) -> agents.Agent:
    """Instantiate 'fast' agent for adaptation phase. This fast agent will
      interact with the environment and perform fast learning & posterior update.
    """
    if self._explore_strategy == 'random':
      self._actor = RandomActor(self._environment_spec, verbose_level=self._verbose_level)
    elif self._explore_strategy == 'ucb':
      self._actor = UCBActor(
          environment_spec=self._environment_spec,
          temperature=self._ucb_temp,
          verbose_level=self._verbose_level
      )
    elif self._explore_strategy == 'mtucb':  # ONLY for meta-eval
      self._actor = MTUCBActor(
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

    # Create MSGI Learner.
    learner = learning.psgiLearner(
        ilp=self._ilp,
        grprop=self._grprop,
        prior_grprop=self._prior_grprop)
    self._learner = learner

    return agents.Agent(
        actor=adaptation_actor,
        learner=learner,
        min_observations=0,
        observations_per_step=self._update_period,
    )

  def instantiate_test_actor(self) -> acting.MixedActor:
    """Instantiate the actor for test phase. This actor will interact with
      the environment and evaluate the performance of the (adapted) fast agent.
    """
    # Return GRProp for test phase.
    if self._posterior_mode == 'policy':
      self._test_actor = acting.MixedActor(
          actor=self._grprop,
          prior_actor=self._prior_grprop,
      )
    else:
      self._test_actor = self._grprop
    return EvalWrapper(self._test_actor, verbose_level=self._verbose_level)

  def reset_agent(self, environment: dm_env.Environment):
    """Reset 'fast' agent and prior upon samping a new task."""
    # 1. Reset fast agent
    self._ilp.reset(environment)

    # 2. Sample prior
    if len(self.prior_pool) > 0 and self._prior_sample_mode is not None:
      self._current_prior, weights = self._sample_prior(self.prior_pool, self._prior_sample_mode, environment)
      if self._posterior_mode == 'policy':
        prior_graphs = [prior['graph'] for prior in self._current_prior]
        self._prior_grprop.observe_task(prior_graphs)
        self._test_actor.set_prior_weights(weights)
      elif self._posterior_mode == 'ilp': # ilp mixing
        self._ilp.update_prior(self._current_prior)
      else:
        raise ValueError(self._posterior_mode)

    # Reset adaptation actor.
    if self._explore_strategy == 'ucb':
      self._actor.observe_task(environment.task_embedding)
    elif self._explore_strategy == 'mtucb':
      # TODO: move this to ilp
      for p in self._current_prior:
        p['graph'].reward_count = p['reward_count']
      self._actor.observe_task([p['graph'] for p in self._current_prior])

  def _sample_prior(self, buffer, mode, environment):
    batch_size = environment.batch_size
    if mode == 'uniform':
      indices = np.random.randint(len(buffer), size=batch_size)
      weights = [0.5] * len(buffer)
    elif mode == 'coverage':
      # Assumes that tasks are same across parallel envs
      indices, weights = [], []
      current_pools = environment.index_to_pool
      for current_pool in current_pools:
        num_current = len(current_pool)
        coverage = []
        for data in buffer:
          prior_pool = data['index_to_pool']
          subtask_reward = data['graph'].subtask_reward

          num_prior = len(prior_pool)
          num_common = np.in1d(current_pool, prior_pool, assume_unique=True).sum()
          bias = (np.max(subtask_reward) > 0.0).astype(np.float)  # temporary bonus
          coverage.append(bias + (num_common/num_current + num_common/num_prior * 0.01))
        weights.append(np.max(coverage))
        idx = np.argmax(coverage + 1e-5 * np.random.random(len(coverage)))
        indices.append(idx)
    elif mode == 'debug':
      # assumes that we train / eval on the same dataset
      graph_index = self._count * batch_size
      self._count += 1
      indices = graph_index + np.arange(batch_size)
      print('indices=', indices)
      weights = [1.0] * len(buffer)
    return [buffer[idx] for idx in indices], weights

  def update(self, last=False):
    """Perform meta-training (slow-learning), and update prior
    """
    if last:
      prior_pool = []
      ilp = self._ilp
      for traj, kmap, graph in zip(ilp.trajectory, ilp.kmaps, ilp.graphs):
        prior_dict = traj
        prior_dict['kmap'] = kmap
        prior_dict['graph'] = graph
        prior_dict['env_id'] = ilp.environment_id
        prior_dict['index_to_pool'] = graph.index_to_pool
        prior_dict['pool_to_index'] = graph.pool_to_index
        prior_pool.append(prior_dict)
      self.prior_pool.extend(prior_pool)
    # See also: EnvironmentMetaLoop's _meta_agent.update().
    # TODO: MetaAgent.update() could be called during adaptation.
    # In this case, we can have a property like `_should_meta_update`.

    # TODO: For psgi-Adapt, etc. where meta-training happens,
    # the updates made here (e.g. parameter update and ILP policy) should be
    # reflected to the fast 'Agent'. Therefore, we need a reference to it.

  def save(self, filename):
    """Save prior & meta-learned modules."""
    np.save(filename, self.prior_pool)

  def load(self, filename):
    """Load prior & meta-learned modules."""
    if isinstance(filename, list):  # Load from multiple files.
      self.prior_pool = []
      for f in filename:
        self.prior_pool.extend(np.load(f, allow_pickle=True))
    else:
      self.prior_pool = np.load(filename, allow_pickle=True)
