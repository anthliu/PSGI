"""Implements PSGI."""

from typing import Optional

import dm_env
import numpy as np
from acme import specs
from acme.utils import loggers

from psgi import agents
from psgi.agents import meta_agent
from psgi.agents.psgi import acting, learning
from psgi.agents.psgi.adapt_acting import MTCountGRPropActor, CountGRPropActor, SubtaskCountActor
from psgi.agents.base import RandomActor, CountBasedActor, UCBActor
from psgi.envs.predicate_graph import PredicateLogicGraph

from psgi.agents.grprop import CycleGRPropActor
from psgi.agents.eval_actor import EvalWrapper

from psgi.graph.pilp import PILP

REWARD_TRANSFER = False
USE_EMBEDDINGS = True
if USE_EMBEDDINGS:
  from torchtext.vocab import GloVe
  from sklearn.cluster import KMeans
  from sklearn.exceptions import ConvergenceWarning

  GLOVE_DIM = 50
  MAX_PRED_DIM = 4
  EMBEDDING_GLOVE = GloVe(name='6B', dim=GLOVE_DIM)
  SPECIAL_CASES = {
    'aluminumfoil': 'aluminum',
    'butterknife': 'knife',
    'coffeemachine': 'brewer',
    'creditcard': 'card',
    'diningtable': 'table',
    'dishsponge': 'sponge',
    'garbagebag': 'bag',
    'garbagecan': 'garbage',
    'handfull': 'full',
    'lightswitch': 'switch',
    'papertowelroll': 'towel',
    'peppershaker': 'pepper',
    'saltshaker': 'salt',
    'shelvingunit': 'shelf',
    'sidetable': 'table',
    'sinkbasin': 'sink',
    'soapbottle': 'soap',
    'spraybottle': 'bottle',
    'stoveburner': 'stove',
    'stoveknob': 'knob',
    'winebottle': 'bottle',
  }
  def _clean_name(s):
    name = s.split('_')[-1].lower()
    return SPECIAL_CASES.get(name, name)# return special or name if not in special cases

  EMBED_FUNC = lambda s: EMBEDDING_GLOVE.vectors[EMBEDDING_GLOVE.stoi[_clean_name(s)]].numpy()

class PSGI(meta_agent.MetaAgent):
  """PSGI with a random/count-based adaptation policy."""

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
      ucb_temp: float = 1.0,
      # GRProp
      grprop_temp: float = 10.0,
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
    self._environment_id = environment_id
    self._visualize = visualize
    self._directory = directory

    # Define sub-modules shared between adapt-agent and test-actor
    self._pilp = PILP(
        environment_spec=environment_spec,
        num_adapt_steps=num_adapt_steps,
        branch_neccessary_first=branch_neccessary_first,
        visualize=visualize,
        directory=directory,
        environment_id=environment_id
    )
    self._grprop_actor = CycleGRPropActor(
        environment_spec=environment_spec,
    )
    self._grprop = self._grprop_actor._grprop

    # Adaptation strategy.
    self._explore_strategy = exploration
    self._ucb_temp = ucb_temp

    # Multi-task learning
    self.prior_pool = []
    self._test_actor = None

    self._prior_grprop_actor = CycleGRPropActor(
        environment_spec=environment_spec,
    )
    self._prior_grprop = self._prior_grprop_actor._grprop


  @property
  def skip_test(self) -> bool:
    return self._learner.skip_test

  def instantiate_adapt_agent(self) -> acting.PSGIActor:
    """Instantiate 'fast' agent for adaptation phase. This fast agent will
      interact with the environment and perform fast learning & posterior update.
    """
    if self._explore_strategy == 'random':
      self._actor = RandomActor(self._environment_spec, verbose_level=self._verbose_level)
    elif self._explore_strategy == 'count':
      self._actor = SubtaskCountActor(
          self._environment_spec, 
          temperature=self._ucb_temp,
          verbose_level=self._verbose_level)
    elif self._explore_strategy == 'ucb':
      self._actor = UCBActor(
          environment_spec=self._environment_spec,
          temperature=self._ucb_temp,
          verbose_level=self._verbose_level)
    elif self._explore_strategy == 'grprop':
      if self._mode == 'meta_eval':
        self._actor = MTCountGRPropActor(
            prior_grprop=self._prior_grprop_actor._grprop,
            environment_spec=self._environment_spec,
            temperature=self._ucb_temp,
            verbose_level=self._verbose_level,
            grprop=self._grprop)
      else:
        self._actor = CountGRPropActor(
            environment_spec=self._environment_spec,
            temperature=self._ucb_temp,
            verbose_level=self._verbose_level,
            grprop=self._grprop)
    else:
      raise NotImplementedError

    # Create MSGI Actor for adaptation phase.
    adaptation_actor = acting.PSGIActor(
        actor=self._actor,
        pilp=self._pilp,
        verbose_level=self._verbose_level)

    # Create MSGI Learner. Perform fast-learning and posterior update
    self._learner = learning.PSGILearner(
        environment_id=self._environment_id,
        mode=self._mode,
        pilp=self._pilp,
        grprop_actor=self._grprop_actor)

    return SGIAgent(
        actor=adaptation_actor,
        learner=self._learner,
        min_observations=0,
        observations_per_step=self._update_period,
    )

  def instantiate_test_actor(self) -> CycleGRPropActor:
    """Instantiate the actor for test phase. This actor will interact with
      the environment and evaluate the performance of the (adapted) fast agent.
    """
    if self._mode == 'meta_eval':
      self._test_actor = acting.MixedActor(
          actor=self._grprop_actor,
          prior_actor=self._prior_grprop_actor,
      )
    else:
      self._test_actor = self._grprop_actor
    # Return GRProp for test phase.
    #return EvalWrapper(self._test_actor, verbose_level=self._verbose_level)
    return self._test_actor
  
  def reset_agent(self, environment: dm_env.Environment):
    """Reset 'fast' agent and prior upon samping a new task."""
    self.batch_size = environment.batch_size
    self._pilp.reset(environment)
    self._learner.reset(environment) # update parameter-set/-embedding for new task
    if hasattr(self._actor, 'observe_task'):
      self._actor.observe_task(environment.task_embedding) # reset adaptation actor.

    # Sample prior and feed to adapt-/test-actor
    if len(self.prior_pool) > 0:
      best_prior_index = np.argmax([prior['reward_sum'].sum() for prior in self.prior_pool]) # choose best prior
      prior_unrolled_graphs = self._update_unroll_prior(self.prior_pool[best_prior_index], environment) # fit prior to current task

      for prior_graph in prior_unrolled_graphs:
        prior_graph.finalize_graph()

      self._prior_grprop_actor.observe_task(prior_unrolled_graphs) # feed prior to prior_grprop in adapt-/test-actor

  def update(self, last=False):
    """Perform meta-training (slow-learning), and update prior
    """
    if last:
      for kmap, graph, effect_mat, comp_count, reward_sum in zip(self._pilp._kmap_sets, self._pilp._graphs, self._pilp._effect_mats, self._pilp.comp_count, self._pilp.reward_sum):
        prior_dict = {
            'kmap': kmap,
            'graph': graph,
            'effect_mat': effect_mat,
            'feature_set': self._pilp._feature_set,# TODO: feature set different across batches?
            'option_param': self._pilp.option_param,
            'subtask_param': self._pilp.subtask_param,
            'option_label': self._pilp._option_label,
            'subtask_label': self._pilp._subtask_label,
            'literal_subtask_label': self._pilp._literal_subtask_label,
            'comp_count': comp_count,
            'reward_sum': reward_sum
        }
        self.prior_pool.append(prior_dict)
    return

    # See also: EnvironmentMetaLoop's _meta_agent.update().
    # TODO: MetaAgent.update() could be called during adaptation.
    # In this case, we can have a property like `_should_meta_update`.

    # TODO: For MSGI-Adapt, etc. where meta-training happens,
    # the updates made here (e.g. parameter update and ILP policy) should be
    # reflected to the fast 'Agent'. Therefore, we need a reference to it.
  
  def _update_unroll_prior(self, prior, environment):
    _parameters = environment.parameters
    prior_unrolled_graphs = []
    for i in range(self.batch_size):
      prior_predicate_graph = PredicateLogicGraph('Prior graph')
      assert all(prior_op == op for prior_op, op in zip(prior['option_label'], self._pilp._option_label)), 'Predicate options in prior must be the samei as current task'

      # find parameter remapping
      prior_parameters = [None] * prior['feature_set'].nparams
      for param, j in prior['feature_set'].param_to_idx.items():
        prior_parameters[j] = param
      prior_param_embed = np.stack([EMBED_FUNC(param) for param in prior_parameters])
      param_embed = np.stack([EMBED_FUNC(param) for param in _parameters[i]])

      nearest = param_embed.dot(prior_param_embed.T).argmax(axis=1)
      param_to_prior = {_parameters[i][j]: prior_parameters[k] for j, k in enumerate(nearest)}

      # remap all predicate graph things
      updated_kmap, updated_subtask_labels = self._remap_subtask_arr(
          arrs=prior['kmap'],
          prior_subtask_label=prior['subtask_label'], 
          subtask_label=self._pilp._subtask_label, 
          prior_parameters=prior_parameters,
          parameters=_parameters[i],
          param_to_prior=param_to_prior
      )
      (updated_effect_mat,), _ = self._remap_subtask_arr(
          arrs=(prior['effect_mat'],),
          prior_subtask_label=prior['subtask_label'], 
          subtask_label=self._pilp._subtask_label, 
          prior_parameters=prior_parameters,
          parameters=_parameters[i],
          param_to_prior=param_to_prior
      )
      (prior_reward_count, prior_comp_count), updated_literal_subtask_labels = self._remap_subtask_arr(
          arrs=(prior['reward_sum'], prior['comp_count']),
          prior_subtask_label=prior['literal_subtask_label'], 
          subtask_label=self._pilp._literal_subtask_label, 
          prior_parameters=prior_parameters,
          parameters=_parameters[i],
          param_to_prior=param_to_prior
      )

      if REWARD_TRANSFER:
        updated_reward_count = (prior_reward_count / np.maximum(prior_comp_count, 1))
        updated_comp_count = (prior_comp_count >= 1).astype(np.int32)
        updated_reward_count += self._pilp.reward_sum[i]
        updated_comp_count += self._pilp.comp_count[i]
      else:
        updated_reward_count = self._pilp.reward_sum[i]
        updated_comp_count = self._pilp.comp_count[i]

      updated_reward = self._pilp._infer_reward(updated_comp_count, updated_reward_count,)# TODO call this a proper way

      # translate features
      #updated_feature_mat = np.zeros((len(_parameters[i]), prior['feature_set'].nfeatures), dtype=np.float32)
      updated_feature_mat = prior['feature_set'].parameter_embedding[nearest]
      assert nearest.shape[0] == len(_parameters[i])
      assert updated_feature_mat.shape[1] == prior['feature_set'].nfeatures

      prior_feature_dict = self._learner._build_feature_dict(updated_feature_mat, prior['feature_set'].feature_labels, _parameters[i])

      # fill edges in prior predicate graph
      prior_predicate_graph.initialize_from_kmap(
        updated_kmap, updated_effect_mat, updated_subtask_labels,
        self._pilp._option_label, updated_literal_subtask_labels, updated_reward)

      graph, option_name_pool = prior_predicate_graph.unroll_graph(_parameters[i], prior_feature_dict)
      prior_unrolled_graphs.append(graph)
      """
      if i == 0 and self._visualize:
        if self._visualize and i == 0:
          dot = self._graph_visualizer.visualize(updated_kmap, updated_effect_mat, subtask_label=updated_subtask_labels, option_label=self._pilp._option_label)
          filepath = f"{self._visualize_directory}/prior_graph_step{self.step_count}"
          self._graph_visualizer.render_and_save(dot, path=filepath)"""

    return prior_unrolled_graphs

  def save(self, filename):
    """Save prior & meta-learned modules."""
    np.save(filename, self.prior_pool)
    return 

  def load(self, filename):
    """Load prior & meta-learned modules."""
    if isinstance(filename, list):  # Load from multiple files.
      for f in filename:
        self.prior_pool.extend(np.load(f, allow_pickle=True))
    else:
      self.prior_pool = np.load(filename, allow_pickle=True)
    return 

  def update_adaptation_progress(self, current_split: int, max_split: int):
    """Update the adaptation progress with adaptation / testing actors
    """
    if isinstance(self._test_actor, acting.MixedActor):
      self._test_actor.update_adaptation_progress(current_split, max_split)

  def _remap_subtask_arr(self, arrs, prior_subtask_label, subtask_label, prior_parameters, parameters, param_to_prior):
    # create prior subtask to subtask mapping
    prior_subtask_to_idx = {psl: i for i, psl in enumerate(prior_subtask_label)}
    subtask_to_prior = np.full(len(subtask_label), -1, dtype=np.int32)
    for i, subtask in enumerate(subtask_label):
      name, *args = subtask.split(',')
      if name.startswith('(f_'):
        continue
      new_args = []
      for arg in args:
        new_arg = arg
        for param, prior_param in param_to_prior.items():
          new_arg = new_arg.replace(param, prior_param)# TODO inefficient and clunky
        new_args.append(new_arg)
      new_subtask = ','.join([name] + new_args)
      assert new_subtask in prior_subtask_to_idx
      subtask_to_prior[i] = prior_subtask_to_idx[new_subtask]

    # create array mapping vars
    subtask_feature_mask = np.array([not subtask.startswith('(f_') for subtask in subtask_label], dtype=np.bool)
    prior_subtask_feature_mask = np.array([not subtask.startswith('(f_') for subtask in prior_subtask_label], dtype=np.bool)
    new_arr_length = subtask_feature_mask.sum() + (~prior_subtask_feature_mask).sum()
    new_arr_feature_mask = np.zeros(new_arr_length, dtype=np.bool)
    new_arr_feature_mask[:subtask_feature_mask.sum()] = True
    remapped_arrs = []
    for arr in arrs:
      if isinstance(arr, bool):
        remapped_arr = arr
      elif len(arr.shape) == 1:
        assert arr.shape[0] == len(prior_subtask_label)
        remapped_arr = np.zeros(new_arr_length, dtype=arr.dtype)
        remapped_arr[subtask_feature_mask] = arr[subtask_to_prior[subtask_feature_mask]]
      elif len(arr.shape) == 2:
        assert arr.shape[1] == len(prior_subtask_label)
        remapped_arr = np.zeros((arr.shape[0], new_arr_length), dtype=arr.dtype)
        remapped_arr[:, new_arr_feature_mask] = arr[:, subtask_to_prior[subtask_feature_mask]]
        remapped_arr[:, ~new_arr_feature_mask] = arr[:, ~prior_subtask_feature_mask]
      else:
        raise NotImplementedError
      remapped_arrs.append(remapped_arr)

    # create new subtask labels
    new_subtask_labels = [subtask_label[i] for i in range(len(subtask_label)) if subtask_feature_mask[i]]
    new_subtask_labels.extend([prior_subtask_label[i] for i in range(len(prior_subtask_label)) if not prior_subtask_feature_mask[i]])

    return remapped_arrs, new_subtask_labels

class MSGI_plus(PSGI):
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
      ucb_temp: float = 1.0,
      # GRProp
      grprop_temp: float = 10.0,
      visualize: bool = False,
      directory: str = 'visualize',
      environment_id: Optional[str] = None,
      verbose_level: Optional[int] = 0,
      ):
    super().__init__(
      environment_spec=environment_spec,
      mode=mode,
      num_adapt_steps=num_adapt_steps,
      num_trial_splits=num_trial_splits,
      logger=logger,
      branch_neccessary_first=branch_neccessary_first,
      exploration=exploration,
      ucb_temp=ucb_temp,
      grprop_temp=grprop_temp,
      visualize=visualize,
      directory=directory,
      environment_id=environment_id,
      verbose_level=verbose_level,
    )
    self._pilp.switch_to_msgi()
  
  def instantiate_test_actor(self) -> CycleGRPropActor:
    """Instantiate the actor for test phase. This actor will interact with
      the environment and evaluate the performance of the (adapted) fast agent.
    """
    self._test_actor = self._grprop_actor
    # Return GRProp for test phase.
    #return EvalWrapper(self._test_actor, verbose_level=self._verbose_level)
    return self._test_actor

  def save(self, filename):
    raise NotImplementedError
  
  def load(self, filename):
    raise NotImplementedError

class SGIAgent(agents.Agent):
  """Common Agent class for all SGI-based agents that infers subtask graph.
  """
  def __init__(
      self,
      actor,
      learner,
      min_observations,
      observations_per_step,
  ):
    self._previous_learner_step = 0
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=min_observations,
        observations_per_step=observations_per_step)
  
  # Additional APIs for SGI
  @property
  def inferred_task(self):
    return self._learner.inferred_task
  
  @property
  def is_learner_updated(self):
    return self._learner.step_count != self._previous_learner_step

  def sync_learner(self):
    self._previous_learner_step = self._learner.step_count

