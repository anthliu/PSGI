'''Configuration script for Target environment.'''
import abc
import warnings
from typing import Dict
from collections import OrderedDict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs.logic_graph import op_to_dict
from psgi.envs.predicate_graph import _parse_name

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


class BasePredicateConfig:
  environment_id = 'base'

  def __init__(self, seed: int, graph_param: str, keep_pristine: bool = False, feature_mode: str = 'gt', visualize: bool = False):
    self.graph_param = graph_param
    self._visualize = visualize
    self.seed = seed if not keep_pristine else None
    self.predicate_graph = self._construct_predicate_graph() # shared for train/test
    self.predicate_graph.finalize_graph()
    assert feature_mode in ['gt', 'singletons', 'cluster-s']
    self.feature_mode = feature_mode
    if self.seed is not None:
      np.random.seed(self.seed)

    # randomly sample parameters for current task
    self.param_set = self._sample_param_set(graph_param)

    # Construct feature
    feature_positives = self._construct_feature_positives()
    feature_func_names, feature, feature_mat = self._construct_feature(
      feature_nodes=self.predicate_graph.feature_nodes,
      feature_positives=feature_positives,
      param_pool=self.param_set)
    self.feature_func_names = feature_func_names
    self.feature = feature
    self.feature_mat = feature_mat

    # Unroll predicate into proposition & assign feature. after this, precondition is defined only in terms of completion
    _graph, _option_name_pool = self.predicate_graph.unroll_graph(param_pool=self.param_set, feature=feature)
    self.graph = _graph
    self.option_name_pool = _option_name_pool
    self.initial_completion = self._construct_initial_completion(self.graph) # initial completion

    self.kmap_by_option = self.graph.to_kmap()

    # Set subtask and option ordering
    self._subtasks = list(sorted([node.name for node in self.graph.subtask_nodes]))
    self._options = list(sorted([name for name in self.option_name_pool]))
    
    self.parameter_name_to_global_index = {name: index for index, name in enumerate(self.parameters)}
    self.parameter_name_to_index = {name: index for index, name in enumerate(self.param_set)}
    self._subtask_param_embeddings, self._option_param_embeddings = self._construct_param_embeddings()

    # Set subtask reward
    self.terminal_subtasks = []
    self.success_subtasks = []
    self.repeatable_subtasks = []
    self._set_reward_and_termination(self.param_set)
    
    if keep_pristine:
      assert True # TODO: add assertions

    # Define subtask rewards.
    self.subtask_reward = OrderedDict()
    for subtask in self.graph.subtask_nodes:
      self.subtask_reward[subtask.name] = subtask.reward

    # Visualize graph
    if self._visualize:
      self.visualize_GT_graph()
  
  @property
  def subtasks(self):
    return self._subtasks
  @property
  def subtask_name_to_index(self) -> Dict[str, int]:
    return {name:index for index,name in enumerate(self.subtasks)}
  @property
  def options(self): # All the options. Eligibility name
    return self._options
  @property
  def option_name_to_index(self): # All the options. Eligibility name
    return {name:index for index,name in enumerate(self.options)}
  @property # For special purpose only
  def feasible_options(self): # only feasible options (i.e., name of option nodes)
    return [node.name for node in self.graph.option_nodes]

  @property
  def subtask_param_embeddings(self):
    # ndarray : #Subtasks x #parameters x #embedding dim
    return self._subtask_param_embeddings
  @property
  def option_param_embeddings(self):
    # ndarray : #Options x #parameters x #embedding dim
    return self._option_param_embeddings
  
  def _add_reward(self, name, reward, terminal, success, repeat=False):
    if name not in self.graph:
      return False
    self.graph.add_reward(name, reward)
    if terminal:
      self.terminal_subtasks.append(name)
    if success:
      self.success_subtasks.append(name)
    if repeat:
      self.repeatable_subtasks.append(name)
    return True

  def compute_effect_mat(self, graph) -> np.ndarray:
    num_option = len(graph.option_nodes)
    num_subtask = len(graph.subtask_nodes)
    num_feature = len(graph.feature_nodes) if hasattr(graph, 'feature_nodes') else 0
    effect_mat = np.zeros((num_option, num_subtask), dtype=np.int32)
    for i, option_node in enumerate(graph.option_nodes):
      effect_dict = op_to_dict(option_node.effect)
      for subtask_name, value in effect_dict.items():
        index = graph.subtask_name_to_index[subtask_name]
        effect_mat[i, index] = 1 if value else -1

    padded_effect_mat = np.pad(effect_mat, pad_width=((0, 0), (num_feature,0)))
    return padded_effect_mat
  
  @abc.abstractmethod
  def parameters(self):
    """Return parameters
    """

  @abc.abstractmethod
  def _construct_predicate_graph(self):
    """Implement predicate precondition&effect
    """

  @abc.abstractmethod
  def _sample_param_set(self, seed, graph_param):
    """Implement (random) parameter sampling
    """
  
  @abc.abstractmethod
  def _set_reward_and_termination(self, params):
    """Implement reward & termination options
    """
      
  @abc.abstractmethod
  def _construct_initial_completion(self, unrolled_graph):
    """Implement initial completion
    """
  
  @abc.abstractmethod
  def _perturb_subtasks(self, rng):
    """Implement graph perturbation
    """

  @abc.abstractmethod
  def _construct_feature_positives(self):
    """Implement a list of parameters whose feature outcome is True
    """

  def visualize_GT_graph(self):
    _graph_visualizer = graph_utils.OptionSubtaskGraphVisualizer()
    """
    # Visualize unrolled graph
    kmap_by_option = self.kmap_by_option
    _effect_mat = self.compute_effect_mat(self.graph)
    dot = _graph_visualizer.visualize(kmap_by_option, _effect_mat, subtask_label=self.subtasks, option_label=self.feasible_options)
    filepath = f"visualize/GT_graph_{self.environment_id}"
    _graph_visualizer.render_and_save(dot, path=filepath)
    """

    # Visualize predicate graph
    kmap_by_option = self.predicate_graph.to_kmap()
    _effect_mat = self.compute_effect_mat(self.predicate_graph)
    feature_label = [node.name for node in self.predicate_graph.feature_nodes]
    subtask_label = [node.name for node in self.predicate_graph.subtask_nodes]
    option_label = [node.name for node in self.predicate_graph.option_nodes]
    dot = _graph_visualizer.visualize(kmap_by_option, _effect_mat, subtask_label=feature_label+subtask_label, option_label=option_label)
    filepath = f"visualize/GT_predicate_graph_{self.environment_id}"
    _graph_visualizer.render_and_save(dot, path=filepath)


  def _construct_feature(self, feature_nodes, feature_positives, param_pool):
    # Initialize
    feature = OrderedDict()
    feature_func_names = set()
    for node in feature_nodes:
      predicate = node.get_predicate_with_param(param_pool)
      prop_params = predicate.ground()
      for prop_param in prop_params:
        name = prop_param.pretty()
        feature[name] = False
        feature_func_names.add(prop_param[0])
    feature_func_names = list(sorted(feature_func_names))

    ### Fill out feature
    for feat_name, param_set in feature_positives.items():
      for x in param_set:
        feat_func_name = f"({feat_name}, {x})"
        if feat_func_name in feature:
          feature[feat_func_name] = True

    # Assertion.
    for feature_func_name in feature_func_names:
      if not any([val for feature_name, val in feature.items() if feature_func_name in feature_name]):
        print(f'Warning: The feature {feature_func_name}() is always False')
        #assert False
    #
    if self.feature_mode == 'gt':
      feature_mat = np.zeros( (len(self.param_set), len(feature_func_names)) )
      for row, par in enumerate(self.param_set):
        for col, feat in enumerate(feature_func_names):
          feature_mat[row, col] = feature[f"({feat}, {par})"]
      feature_labels = feature_func_names
    elif self.feature_mode == 'singletons':
      feature_mat = np.eye(len(self.param_set))
      feature_labels = [f'f_is_{par}' for par in self.param_set]
    elif self.feature_mode == 'cluster-s':
      X = np.stack([EMBED_FUNC(p) for p in self.param_set])
      candidate_features = []
      seen = set()
      for num_clusters in range(2, len(self.param_set)+1):
        with warnings.catch_warnings():
          warnings.filterwarnings("ignore", category=ConvergenceWarning)
          clusters = KMeans(n_clusters=num_clusters, random_state=1).fit_predict(X)
        for cls in range(num_clusters):
          cand = cls == clusters
          if cand.mean() > 0.5:
            cand = ~cand# prefer smaller feature
          if cand.sum() <= 0:
            continue# skip everything-features
          if np.any(cand):
            cand_hash = graph_utils.batch_bin_encode(cand)
            cand_not_hash = graph_utils.batch_bin_encode(~cand)
            if (cand_hash not in seen) and (cand_not_hash not in seen):
              seen.add(cand_hash)
              candidate_features.append(cand)
      # add in singletons in case of same embedding features but different behavior
      for singleton in range(len(self.param_set)):
        cand = np.zeros(len(self.param_set), dtype=np.bool)
        cand[singleton] = True
        cand_hash = graph_utils.batch_bin_encode(cand)
        cand_not_hash = graph_utils.batch_bin_encode(~cand)
        if (cand_hash not in seen) and (cand_not_hash not in seen):
          seen.add(cand_hash)
          candidate_features.append(cand)
      
      feature_labels = ['f_is_' + '|'.join(self.param_set[p] for p in np.flatnonzero(cand)) for cand in candidate_features]
      feature_mat = np.stack(candidate_features).astype(int).T
    else:
      raise NotImplementedError
    return feature_labels, feature, feature_mat

  def _construct_param_embeddings(self):
    subtask_param_embed = np.zeros((len(self.subtasks), MAX_PRED_DIM, GLOVE_DIM), dtype=np.float32)
    for i, name in enumerate(self.subtasks):
      params = _parse_name(name)
      assert 0 < len(params) <= MAX_PRED_DIM
      for j, param in enumerate(params):
        subtask_param_embed[i, j] = EMBED_FUNC(param)
    subtask_param_embed /= np.abs(subtask_param_embed).max()

    option_param_embed = np.zeros((len(self.options), MAX_PRED_DIM, GLOVE_DIM), dtype=np.float32)
    for i, name in enumerate(self.options):
      params = _parse_name(name)
      assert 0 < len(params) <= MAX_PRED_DIM
      for j, param in enumerate(params):
        option_param_embed[i, j] = EMBED_FUNC(param)
    option_param_embed /= np.abs(option_param_embed).max()

    return subtask_param_embed, option_param_embed

  def intersect(self,listA, listB):
    return [elem for elem in listA if elem in listB]

  def difference(self,listA, listB):
    return [elem for elem in listA if elem not in listB]

  def _random_sample(self, param_set, min_p, max_p, must_set):
    num_p = np.random.randint(min_p, max_p+1)
    must_params = self.intersect(must_set, param_set)

    if num_p > len(must_params):
      num_remain = num_p - len(must_params)
      params_except_must = self.difference(param_set, must_set)
      rest_params = np.random.permutation(params_except_must)[:num_remain].tolist()
      return must_params + rest_params
    else:
      return must_params
