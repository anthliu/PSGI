import os
import abc
from typing import Optional, Dict, List, Union, Sequence, Any

import numpy as np
import scipy.special
from .maze_config import MazeConfig
from psgi.utils import graph_utils


__PATH__ = os.path.abspath(os.path.dirname(__file__))


class SubtaskGraph(abc.ABC):
    """Abstract interface for SubtaskGraphCache and SubtaskGraphGenerator.

    This is a sufficient statistics for the behavior of MazeEnv, defining
    subtask nodes and their dependencies, etc."""
    pass

    '''
    (Note: pool and subtask id are exchangable)

    index_to_pool: ndarray[subtask index:int -> subtask id:int]
    pool_to_index: ndarray[subtask id:int -> subtask index:int] (missing: -1)
    get_elig(completion) -> eligibility
    subtask_id_list: List[subtask index: int]
    rew_mag: List[subtask index: int]
    '''

    @abc.abstractmethod
    def reset_graph(self, graph_index: Optional[int]):
        raise NotImplementedError

    @abc.abstractmethod
    def get_elig(self, completion):
        raise NotImplementedError


class SubtaskGraphCache(SubtaskGraph):
    """
    Loads a SubtaskGraph dataset for MazeEnv.

    This is a different representation from graph_utils.SubtaskGraph that
    is used in wob environments and maze envs.  Attributes/properties
    are not identical.

    Required properties (see env.task_embedding):
    - num_graph (XXX)
    - subtask_id_list
    - subtask_reward
    - index_to_pool (ndarray)
    - pool_to_index (ndarray)
    - graph_index
    - get_elig()
    - numP
    - numA
    """

    def __init__(self, folder, filename, max_task=0):
        # 1. init/load graph
        self.filename = filename

        # subtasks / edges (ANDmat&ORmat, allow) / subtask reward
        self._load_graph(folder)
        self.max_task = max_task
        self.graph_index = -1
        # NOTE: attributes won't be set until set_graph_index is called

    def _load_graph(self, folder):
        fname = os.path.join(__PATH__, folder, self.filename+'.npy')
        self.graph_list = np.load(fname, allow_pickle=True)

    @property
    def num_graph(self):
        return len(self.graph_list)

    def reset_graph(self, graph_index=None):
        self.set_graph_index(graph_index=graph_index)

    def set_graph_index(self, graph_index):
        """Load a specific graph instance from the graph_list."""

        if graph_index is None:
          graph_index = np.random.permutation(self.num_graph)[0]
        else:
          graph_index = graph_index % self.num_graph

        # Load data from file. Choose one from the full suite.
        self.graph_index = graph_index
        graph: dict   # ORmat, rmag, W_o, W_a, ANDmat, trind
        graph = self.graph_list[graph_index]

        self.num_level = len(graph['W_a'])
        self.ANDmat = graph['ANDmat'].astype(np.float)
        self.ORmat = graph['ORmat'].astype(np.float)
        self.W_a = graph['W_a']
        self.W_o = graph['W_o']
        self.nb_subtask = self.ORmat.shape[0]

        self.numP = [self.W_a[0].shape[1]]
        self.numA = []
        self.num_or = self.numP[0]
        self.num_and = 0
        for lv in range(self.num_level):
            self.numP.append(self.W_o[lv].shape[0])
            self.numA.append(self.W_a[lv].shape[0])
            self.num_or = self.num_or + self.numP[lv + 1]
            self.num_and = self.num_and + self.numA[lv]

        self.b_AND = np.not_equal(self.ANDmat, 0).sum(1).astype(np.float)
        self.b_OR = np.ones(self.nb_subtask)
        self.b_OR[:self.numP[0]] = 0

        self.subtask_reward = np.array(graph['rmag'])
        self.subtask_id_list = graph['trind'].tolist()

        self._index_to_pool = np.zeros( (self.nb_subtask), dtype=np.int32)
        self._pool_to_index = np.full( (self.max_task), fill_value=-1, dtype=np.int32)
        for ind in range(self.nb_subtask):
            pool_id = self.subtask_id_list[ind]
            self._index_to_pool[ind] = pool_id
            self._pool_to_index[pool_id] = ind

        self.tind_by_layer = []
        bias = 0
        for num in self.numP:
          self.tind_by_layer.append(list(range(bias, bias+num)))
          bias += num

        # sanity check: tind_by_layer needs to be mutually exclusive
        for j, _ in enumerate(self.tind_by_layer):
          if j > 0 and (set(self.tind_by_layer[j - 1]) & set(self.tind_by_layer[j])):
            assert False, "tind_by_layer is not a partition: {}".format(self.tind_by_layer)

    def get_elig(self, completion):
        ANDmat = self.ANDmat
        b_AND = self.b_AND
        ORmat = self.ORmat
        b_OR = self.b_OR

        tp = completion.astype(np.float)*2-1  # \in {-1,1}
        # sign(A x tp + b) (+1 or 0)
        ANDout = np.not_equal(
            np.sign((ANDmat.dot(tp)-b_AND)), -1).astype(np.float)
        elig = np.not_equal(np.sign((ORmat.dot(ANDout)-b_OR)), -1)
        return elig

    @property
    def pool_to_index(self) -> np.ndarray:
      return self._pool_to_index

    @property
    def index_to_pool(self) -> np.ndarray:
      return self._index_to_pool

    def __repr__(self):
      return "<SubtaskGraphCache: {}, {} graphs>".format(self.filename, len(self.graph_list))


class SubtaskGraphGenerator(SubtaskGraph):
  #TODO: should define a interface compliant to SubtaskGraphCache.

  def __init__(self,
               game_config: MazeConfig,
               env_name: str,
               ):
    self._env_name = env_name
    self.graph_index = None
    self._game_config = game_config

  def _set_graph_config(self, level: int):
    # must be called every time the graph is reset (not static any more)
    game_config = self._game_config
    self.max_task = self._game_config.nb_subtask_type

    if self._env_name == 'playground':
      max_layer = 6
      self.level = level
      self.noc_range = np.array([[1, 2]]).repeat(max_layer, axis=0).T
      self.nac_range = np.array([[1, 3]]).repeat(max_layer, axis=0).T
      self.nanc_range= np.array([[0, 2],[0, 1],[0, 1],[0, 0],[0, 0],[0, 0]]).T
      self.ndnp_range= np.array([[2, 3],[2, 3],[2, 3],[0, 0],[0, 0],[0, 0]]).T
      if level == 1:
        self.ntasks = 13
        self.rew_range  = np.array([ [.1, .2], [.3, .4], [.7, .9], [1.8, 2.0] ]).T
        self.ndt_layer  = [2, 1, 0, 0]
        self.nt_layer   = [4, 3, 2, 1]

        self.na_range   = np.array([ [3, 5], [3, 4], [2, 2] ]).T
        self.nanc_range = np.array([ [0, 2], [0, 1], [0, 0]]).T
      elif level == 2:
        self.ntasks = 15
        self.rew_range  = np.array([ [.1, .2], [.3, .4], [.7, .9], [1.8, 2.0] ]).T
        self.ndt_layer  = [2, 2, 0, 0]
        self.nt_layer   = [5, 3, 2, 1]

        self.na_range   = np.array([ [3, 5], [3, 4], [2, 2] ]).T
        self.nanc_range = np.array([ [0, 2], [0, 1], [0, 0]]).T
      elif level == 3:
        self.ntasks = 16
        self.rew_range  = np.array([ [.1, .2], [.3, .4], [.6, .7], [1.0, 1.2], [2.0, 2.2] ]).T
        self.ndt_layer  = [1, 1, 1, 0, 0]
        self.nt_layer   = [4, 3, 3, 2, 1]

        self.na_range   = np.array([ [3, 5], [3, 4], [3, 4], [2, 2] ]).T
      elif level == 4:
        self.ntasks = 16
        self.rew_range  = np.array([ [.1, .2], [.3, .4], [.6, .7], [1.0, 1.2], [1.4, 1.6], [2.4, 2.6] ]).T
        self.ndt_layer  = [0, 0, 0, 0, 0, 0]
        self.nt_layer   = [4, 3, 3, 3, 2, 1]

        self.na_range   = np.array([ [3, 5], [3, 4], [3, 4], [3, 4], [2, 2] ]).T
        self.nanc_range = np.array([ [0, 2], [0, 2], [0, 1], [0, 1], [0, 0]]).T
      else:
        raise NotImplementedError(f"Unknown level: {level}")

      # the total number of layers (<=6)
      self.nlayer = len(self.nt_layer)
      assert self.ntasks == sum(self.ndt_layer + self.nt_layer)
      assert self.nlayer == len(self.ndt_layer) and self.nlayer == self.na_range.shape[1] + 1 and \
             self.nlayer == self.rew_range.shape[1]

      # 2. fillout others
      # TODO: Should we move this to generate() ?
      tbias, self.tlist, self.dtlist = 0, [], []
      self.r_high = np.zeros(self.ntasks)
      self.r_low = np.zeros(self.ntasks)

      for lind in range(self.nlayer):
          nt, ndt = self.nt_layer[lind], self.ndt_layer[lind]
          self.tlist += list( range(tbias, tbias+nt) )
          self.dtlist += list( range(tbias+nt, tbias+nt+ndt) )

          # TODO msgi has a bug here (high vs low)
          low, high = self.rew_range[:,lind]
          assert isinstance(high, (np.floating, float)) and isinstance(low, (np.floating, float))
          self.r_high[tbias:tbias+nt+ndt] = high
          self.r_low[tbias:tbias+nt+ndt] = low

          tbias += (nt + ndt)

      assert np.all(self.r_low <= self.r_high)
      assert self.noc_range.shape == (2, max_layer)
      assert self.rew_range.shape == (2, self.nlayer)

    elif self._env_name == 'mining':
      max_layer = 10   # L
      self.layer_range = [6, max_layer]
      self.ntask_range = np.array([ [3, 3], [4, 4], [2, 2], [3, 3], [1, 2], [2, 3], [1, 2], [1, 2], [1, 3], [1, 2]   ]).T
      self.na_range    = np.array([         [4, 4], [2, 2], [3, 3], [2, 2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]   ]).T
      self.noc_range  = np.array([[1, 2]]).repeat(max_layer, axis=0).T  # [2, L]
      self.nac_range  = np.array([[1, 3]]).repeat(max_layer, axis=0).T  # [2, L]
      self.nanc_range = np.array([[0, 0]]).repeat(max_layer, axis=0).T  # [2, L]
      self.ndnp_range = np.array([[0, 0]]).repeat(max_layer, axis=0).T  # [2, L]
      self.rew_range  = np.array([ [.1, .1], [.1, .2], [.3, .5], [.3, .5], [.8, 1.0], [1.2, 1.5], [1.8, 2.5], [3.0, 4.5], [4.5, 5.5], [5.0, 7.0]  ]).T
      assert self.ntask_range.shape == (2, max_layer)
      assert self.noc_range.shape == (2, max_layer)
      assert self.rew_range.shape == (2, max_layer)

      max_layer = 10
      self.nlayer = np.random.randint(self.layer_range[0], self.layer_range[1]+1, 1 ).item()
      self.nt_layer = []

      for lind in range(self.nlayer):
        nt = np.random.randint(self.ntask_range[0][lind], self.ntask_range[1][lind]+1, 1 ).item()
        self.nt_layer.append(nt)
      self.ndt_layer = [0] * max_layer

      self.ntasks    = sum(self.nt_layer)

      # 2. fillout others
      tbias, self.tlist, self.dtlist = 0, [], []
      self.r_high = np.zeros(self.ntasks)
      self.r_low = np.zeros(self.ntasks)

      for lind in range(self.nlayer):
        nt, ndt = self.nt_layer[lind], self.ndt_layer[lind]
        self.tlist += list( range(tbias, tbias+nt) )
        self.dtlist += list( range(tbias+nt, tbias+nt+ndt) )

        low, high = self.rew_range[:,lind]
        self.r_high[tbias:tbias+nt+ndt] = high
        self.r_low[tbias:tbias+nt+ndt] = low

        tbias += (nt + ndt)

      assert np.all(self.r_low <= self.r_high)

    else:
      raise NotImplementedError(self._env_name)

    self.subtasks = np.random.choice(self.max_task, self.ntasks, replace=False)

  def _set_custom_graph_config(self,
                               num_layers: int,
                               subtask_pool_size: int,
                               subtasks: Union[int, Sequence[int]],
                               subtask_design: List[Dict[str, Any]]
                               ):
    """Initialize graph configuration given control parameters."""
    assert self._env_name == 'playground', ('Mining is not supported yet.')

    if not num_layers:
        raise ValueError("num_layers must be given")
    if not subtask_pool_size:
        raise ValueError("subtask_pool_size must be given")
    if not subtasks:
        raise ValueError("subtasks (int or Sequence[int]) must be given")
    if subtask_design is None:
        raise ValueError("subtask_design must be given")

    # Total pool.
    self.max_task = subtask_pool_size
    assert self.max_task > 0

    if self.max_task > self._game_config.nb_subtask_type:
        raise NotImplementedError(
            f"Currently the subtask pool size ({subtask_pool_size}) cannot "
            f"exceed the limit of Playground's configuration "
            f"({self._game_config.nb_subtask_type})")

    # ntasks
    if isinstance(subtasks, (int, np.integer)):
        self.ntasks = subtasks
        self.subtasks = np.random.choice(self.max_task, self.ntasks, replace=False)
    else:
        self.ntasks = len(subtasks)
        self.subtasks = list(subtasks)
        for t in self.subtasks:
            if not (0 <= t < self.max_task):
                raise ValueError(f"Invalid subtask {t} (< {subtask_pool_size}")

    # determine the number of layers.
    self.nlayer = num_layers
    if self.nlayer > 6:
        raise NotImplementedError("nlayer larger than 6 is not yet supported.")

    # Fill in:
    # noc_range, nac_range, nanc_range, ndnp_range,
    # rew_range, ndt_layer, nt_layer, na_range, nanc_range
    # r_high, r_low
    # tlist, dtlist

    # TODO: This range parameters are subject to change.
    self.noc_range = np.array([[1, 2]]).repeat(self.nlayer, axis=0).T
    self.nac_range = np.array([[1, 3]]).repeat(self.nlayer, axis=0).T
    self.nanc_range= np.array([[0, 2],[0, 1],[0, 1]] + [[0, 0]] * (self.nlayer - 3)).T
    self.ndnp_range= np.array([[2, 3],[2, 3],[2, 3]] + [[0, 0]] * (self.nlayer - 3)).T
    self.rew_range = np.array([[.1, .2], [.3, .4], [.6, .7],
                               [1.0, 1.2], [1.4, 1.6], [2.4, 2.6] ]).T[:, :self.nlayer]

    # Place subtasks to layer.
    #self.ndt_layer  = [0, 0, 0, 0, 0, 0]       # distractors per layer
    #self.nt_layer   = [4, 3, 3, 3, 2, 1]       # subtasks per layer
    self.ndt_layer = np.zeros(self.nlayer, dtype=int)
    self.nt_layer = np.zeros(self.nlayer, dtype=int)
    _seen = {}
    for i, design in enumerate(subtask_design):
        t, l = design['id'], design['layer']
        if t < 0:
            continue       # Ignore
        if not (0 <= l < self.nlayer):
            raise ValueError(f"Invalid Layer {l} at design index {i}")
        if not (t in self.subtasks):
            # TODO If this is random generated, we cannot know them in advance
            raise ValueError(f"Invalid subtask {t}, expected one of {self.subtasks}")
        if t in _seen:
            raise ValueError(f"Subtask {t} is duplicated.")
        if design['distractor']:
            self.ndt_layer[l] += 1
        else:
            self.nt_layer[l] += 1
        _seen[t] = l

    # remaining tasks should be non-distractors.
    remaining_tasks = self.ntasks - (sum(self.ndt_layer) + sum(self.nt_layer))
    assert remaining_tasks >= 0
    # we need to assign at least one subtask for each layer.
    remaining_tasks -= np.sum(self.nt_layer == 0)
    self.nt_layer[self.nt_layer == 0] = 1
    if remaining_tasks < 0:
        raise ValueError("Insufficient number subtasks are provided, "
                         "cannot assign subtasks to existing layers.")
    self.nt_layer += np.random.multinomial(remaining_tasks,
                                           np.ones(self.nlayer) / self.nlayer)

    assert np.all([nt > 0 for nt in self.nt_layer]), str(self.nt_layer)

    # 2. fillout others
    # TODO: Remove duplicates with _set_graph_config()
    tbias, self.tlist, self.dtlist = 0, [], []
    self.r_high = np.zeros(self.ntasks)
    self.r_low = np.zeros(self.ntasks)

    self.na_range   = np.array([ [3, 5], [3, 4], [3, 4], [3, 4], [2, 2] ]).T[:, :self.nlayer-1]
    self.nanc_range = np.array([ [0, 2], [0, 2], [0, 1], [0, 1], [0, 0] ]).T[:, :self.nlayer-1]

    for lind in range(self.nlayer):
        nt, ndt = self.nt_layer[lind], self.ndt_layer[lind]
        self.tlist += list( range(tbias, tbias+nt) )
        self.dtlist += list( range(tbias+nt, tbias+nt+ndt) )

        low, high = self.rew_range[:,lind]
        assert isinstance(high, (np.floating, float)) and isinstance(low, (np.floating, float))
        self.r_high[tbias:tbias+nt+ndt] = high
        self.r_low[tbias:tbias+nt+ndt] = low

        tbias += (nt + ndt)

    # the total number of layers (<=6)
    assert self.ntasks == sum(self.ndt_layer + self.nt_layer)
    assert self.nlayer == len(self.ndt_layer)
    assert self.nlayer == self.na_range.shape[1] + 1
    assert self.nlayer == self.rew_range.shape[1]

    assert self.noc_range.shape == (2, self.nlayer)
    assert self.rew_range.shape == (2, self.nlayer)


  def reset_graph(self, graph_index=None, *,
                  num_layers: Optional[int] = None,
                  subtask_pool_size: Optional[int] = None,
                  subtasks: Optional[Union[int, Sequence[int]]] = None,
                  subtask_design: Optional[List[Dict[str, Any]]] = None,
                  ):
    """Reset with a random subtask graph.

    Args:
      seed: A random seed to use.
      num_layers: The number of layers.
      subtask_pool_size: The size `N` of subtask domain. Subtask ids are assumed
        to be continuous numbers [0..max_subtasks). If not given, uses the
        default setting defined by Mining and Playground.
      subtasks: A set of subtasks to appear in the generated environment/graph.
        If it is an integer `n`, a random subset of size `n` of the subtask pool
        will be chosen. If it is a list, this should denote the pre-populated
        subset of subtasks.
      subtask_design: Each element (as a dict) consists of:
        - 'id': The subtask id. Must be an element of subtasks (if given).
          Negative values (-1) are ignored entries.
        - 'layer': The layer (0 <= layer < L) of the subtask.
        - 'distractor': Indicates whether this is a distractor or not.
    """
    # First populate graph parameters (ntasks, layer, range, etc.)
    if (num_layers is None and subtask_pool_size is None
        and subtasks is None and subtask_design is None):
        self._set_graph_config(level=4)   # Uses preset. TODO parametrize this.
    else:
        self._set_custom_graph_config(num_layers=num_layers,
                                      subtask_pool_size=subtask_pool_size,
                                      subtasks=subtasks,
                                      subtask_design=subtask_design)

    # Regenerate graph.
    self.generate(seed=graph_index)

    # XXX This is a dummy value that has no meaning. (used in step())
    self.graph_index = 1


  def generate(self, *, seed=None):
    # TODO: use np_random.
    np_random = np.random.RandomState(seed) if seed is not None else np.random

    # Hardcoded for now.
    # The code originates from BatchSubtaskGraph, but we will be using only
    # single-batch environment. So we must assume that nbatch = 1 to ensure
    # the compatibility with the MazeEnv interface (e.g. ind_to_id are assumed
    # to be a non-batch dict/list, etc.).
    self.nbatch = 1

    Nprob = np.array([0.25, 0.4, 0.1])
    nbatch = self.nbatch
    self.rmag      = np.zeros([nbatch, self.ntasks], dtype=np.float32)
    self.id_mask   = np.zeros([nbatch, self.max_task], dtype=np.float32)
    self._ind_to_id = np.zeros([nbatch, self.ntasks], dtype=np.int32)  # [B, n]
    self._id_to_ind = np.full([nbatch, self.max_task], -1)

    # 1. ind_to_id
    # Choose the set of subtasks out of the entire pool.
    for bind in range(nbatch):
        id_tensor = self.subtasks
        self.id_mask[bind, id_tensor] = 1
        self._ind_to_id[bind, :] = id_tensor     # same across batch?

    base = np.arange(self.ntasks)[np.newaxis, :].repeat(nbatch, axis=0)
    #self.id_to_ind.scatter_(1, self.ind_to_id, base)
    # TODO: This is a wrong translation (only works with nbatch=1)
    self._id_to_ind[:, self._ind_to_id] = base

    # 2. rmag
    self.rmag = graph_utils._sample_layer_wise(nbatch, self.r_high, self.r_low)

    # 3. na_layer, ndnp_layer
    no_layer    = self.nt_layer
    ndt_layer   = self.ndt_layer
    self.max_NP = sum(no_layer + ndt_layer)

    if self.nlayer > 1:
      na_layer_  = graph_utils._sample_int_layer_wise(
        nbatch, self.na_range[1], self.na_range[0])  # nbatch x (nlayer-1)
      self.max_NA = na_layer_.sum(axis=1).max()
    else:  # flat case (there will be no AND nodes)
      na_layer_ = [[]] * nbatch
      self.max_NA = 0

    # 4.
    self.ANDmat = np.zeros([nbatch, self.max_NA, self.max_NP], dtype=int)
    self.ORmat  = np.zeros([nbatch, self.max_NP, self.max_NA])
    self.b_AND  = np.zeros([nbatch, self.max_NA, 1])
    self.b_OR   = np.zeros([nbatch, self.max_NP, 1])
    self.numP, self.numA = [], []
    self.tind_by_layer = []

    for bind in range(nbatch):
      # prepare
      atable, otable = set(), set()
      na_layer = na_layer_[bind]
      ANDmat  = self.ANDmat[bind]
      ORmat   = self.ORmat[bind]
      obias, abias = (no_layer[0]+ndt_layer[0]), 0
      nump, numa, tlayer  = [obias], [], [ [ *range(obias) ] ]
      ocand = [*range(no_layer[0])]
      for lind in range(self.nlayer-1):
        # 4-0. prepare
        na, no, ndt = na_layer[lind].item(), no_layer[lind+1], ndt_layer[lind+1]
        nt_prev = no_layer[lind] + ndt_layer[lind]
        nanc_low, nanc_high = self.nanc_range[0, lind], self.nanc_range[1, lind]+1
        Nweights = Nprob[nanc_low:nanc_high]

        # 4-1. AND node (non-distractors)
        i = 0
        count = 0
        while i < na:
          aind = abias + i
          # sample #pos-child & #neg-child (nac/nanc)
          nac = np.random.randint(self.nac_range[0, lind], self.nac_range[1, lind]+1, 1 ).item()
          nanc = 0

          if nanc_high > nanc_low:
            nanc = np.where(np.random.multinomial(1, Nweights) > 0)[0][0] + nanc_low
            #nanc = torch.multinomial(Nweights, 1, replacement=True).item() + nanc_low
          #
          and_row = np.zeros(self.ntasks, dtype=int)
          # sample one direct_child (non-distractors)
          oind_ = obias - nt_prev + np.random.choice(no_layer[lind])  # only non-distrators
          if oind_ not in ocand:
            continue  # XXX this does not exist, need to resample
          and_row[oind_] = 1

          # sample nac-1 pos-children and nanc neg-children (non-distractors)
          ocand_copy = [ o for o in ocand ]
          ocand_copy.remove(oind_)

          ocand_copy_tensor = np.array(ocand_copy, dtype=int)
          oinds_ = np.random.permutation(len(ocand_copy))
          ac_oind = ocand_copy_tensor[oinds_[:(nac - 1)]]
          neg_ac_oind = ocand_copy_tensor[oinds_[(nac - 1) : (nac + nanc - 1)]]
          and_row[ac_oind] = 1
          and_row[neg_ac_oind] = -1
          code = graph_utils.batch_bin_encode(and_row)
          if not code in atable: # if not duplicated
            atable.add(code)
            i = i + 1
            ANDmat[aind] = ANDmat[aind] + and_row
          count += 1
          if count > 100:
            # warning: infinite loop, not satisfiable. We do not add any more.
            break
        and_added = i

        # 4-2. OR node
        i = 0
        count = 0
        while i < no + ndt:
          oind = obias + i
          noc = np.random.randint(self.noc_range[0, lind], self.noc_range[1, lind]+1, 1).item()
          ainds = abias + np.random.choice(na, noc, replace=False)
          #or_row = torch.zeros(self.max_NA).index_fill_(0, ainds, 1)
          or_row = np.zeros(self.max_NA)
          or_row[ainds] = 1
          code = graph_utils.batch_bin_encode(or_row)
          if not code in otable: # if not duplicated
            otable.add(code)
            i = i + 1
            ORmat[oind] += or_row
          count += 1
          if count > 100:
            # warning: infinite loop, not satisfiable. We do not add any more.
            break
        ocand += [ *range(obias, obias+no) ]
        or_added = i

        tlayer.append( [*range(obias, obias + or_added)] )
        obias += or_added      # instead of (no + ndt), increment by the actual number added
        abias += and_added     # instead of (na), increment by the actual number added

        nump.append(or_added)   # <= no+ndt
        numa.append(and_added)  # <= na

      # these are for batches
      self.numP.append(nump)
      self.numA.append(numa)
      self.tind_by_layer.append(tlayer)

      # should be done after determining NA
      # 4-3. distractor
      total_a = abias
      dt_inds, count = np.array(self.dtlist, dtype=int), 0
      abias = 0
      for lind in range(self.nlayer-1):
        ndt = ndt_layer[lind]
        ndnp_ = np.random.randint(self.ndnp_range[0, lind], self.ndnp_range[1, lind]+1, ndt )
        for i in range(ndt):
          oind = dt_inds[count]
          assert np.all(ANDmat[:, oind] < 1)
          ndnp = ndnp_[i].item()

          assert total_a - abias >= ndnp
          par_ainds = abias + np.random.choice(total_a - abias, ndnp, replace=False)
          #column = torch.zeros(self.max_NA).scatter_(0, par_ainds, -1)
          column = np.zeros(self.max_NA)
          column[par_ainds] = -1
          #ANDmat[:, oind].copy_(column)
          ANDmat[:, oind] = column
          count += 1
        abias += numa[lind]

    # 5. b_AND & b_OR
    self.b_AND = (self.ANDmat != 0).sum(axis=2).astype(np.float32)
    self.b_OR = np.ones([nbatch, self.ntasks])
    for i in range(nbatch):
      self.b_OR[i, :self.numP[i][0]] = 0  # first layer

    # sanity check: tind_by_layer needs to be mutually exclusive
    for j, _ in enumerate(self.tind_by_layer):
      if j > 0 and (set(self.tind_by_layer[j - 1]) & set(self.tind_by_layer[j])):
        assert False, "tind_by_layer is not a partition: {}".format(self.tind_by_layer)

  @property
  def subtask_reward(self):
    assert self.nbatch == self.rmag.shape[0] == 1
    return self.rmag[0]

  @property
  def pool_to_index(self):
    assert self.nbatch == self._id_to_ind.shape[0] == 1
    return self._id_to_ind[0]

  @property
  def index_to_pool(self):
    assert self.nbatch == self._ind_to_id.shape[0] == 1
    return self._ind_to_id[0]

  @property
  def subtask_id_list(self) -> List[int]:
    assert self.nbatch == self._ind_to_id.shape[0] == 1
    return list(self._ind_to_id[0])

  def get_elig(self, completion):
    # TODO: We require completion to be non-batched for now. Need shape check:
    assert len(completion.shape) == 1

    # whereas all internal implementations (ANDmat, b_AND, ORmat, b_OR)
    # are batched ([1, ...]).
    assert self.nbatch == 1
    ANDmat = self.ANDmat[0]
    b_AND = self.b_AND[0]
    ORmat = self.ORmat[0]  # nb_or x
    b_OR = self.b_OR[0]

    indicator = completion.astype(np.float32)

    ANDout = (np.sign(-b_AND + (ANDmat @ indicator)) != -1).astype(np.float32) #sign(A x indic + b) (+1 or 0)
    elig_hard = np.sign(-b_OR + (ORmat @ ANDout)) != -1
    return elig_hard

  def __str__(self):
    header = f"{type(self).__name__}[\n"
    s = header
    def _p(attr_name):
      nonlocal s
      s += f"  {attr_name}={getattr(self, attr_name)},\n"
    try:
      _p('nlayer')
      _p('ntasks')
    except AttributeError:
      return header[:-1] + "Not Initialized]"
    s += "]"
    return s
