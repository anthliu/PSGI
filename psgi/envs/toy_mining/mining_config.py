'''Configuration script for Target environment.'''
from collections import OrderedDict
from typing import Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.utils.predicate_utils import Predicate, Symbol
from psgi.envs import base_config_v2 as base
from psgi.envs.base_predicate_config import BasePredicateConfig
from psgi.envs.logic_graph import SubtaskLogicGraph
from psgi.envs.predicate_graph import PredicateLogicGraph

# Define parameters (no intersection allowed)
RES_HARD0 = ['wood', 'stone', 'grass']
RES_HARD1 = ['coal', 'iron', 'bronze', 'copper']
RES_HARD2 = ['gold', 'silver', 'diamond', 'ruby', 'platinum']
RESOURCES = RES_HARD0 + RES_HARD1 + RES_HARD2

CRAFTS = ['pickaxe', 'sword', 'necklace', 'bracelet', 'ring']
FURNACE = ['furnace']
PARAMS = RESOURCES + CRAFTS + FURNACE
assert len(set(PARAMS)) == len(PARAMS), "Error! duplicated params!"

TRAIN_ONLY = ['diamond', 'necklace']
EVAL_ONLY = ['grass', 'ruby', 'bracelet']
TRAIN_MUST = ['stone', 'coal', 'iron', 'diamond', 'necklace', 'pickaxe', 'furnace'] # usually due to initial completion and reward
EVAL_MUST = ['stone', 'coal', 'iron', 'ruby', 'bracelet', 'pickaxe', 'furnace'] # usually due to initial completion and reward
assert all([par in PARAMS for par in TRAIN_ONLY+EVAL_ONLY]), "Unknown parameter in TRAIN_ONLY and/or EVAL_ONLY"
assert all([par not in EVAL_ONLY for par in TRAIN_ONLY]), "there is a overlap between TRAIN_ONLY and EVAL_ONLY"
TRAIN_PARAMS = [par for par in PARAMS if par not in EVAL_ONLY]
EVAL_PARAMS = [par for par in PARAMS if par not in TRAIN_ONLY]

# Manually set the feature
SMELTABLE = ['iron', 'bronze', 'copper'] + ['gold', 'silver', 'platinum']
FEATURE_POSITIVE_NAME_PARAMS = {
  'f_hardness0': RES_HARD0,
  'f_hardness1': RES_HARD1,
  'f_hardness2': RES_HARD2,
  'f_lightable': ['wood', 'furnace', 'grass'],
  'f_craftable': CRAFTS,
  'f_smeltable': SMELTABLE,
  'f_isingredient': ['stone', 'diamond', 'ruby'] + SMELTABLE
}
assert all([set(param_set).issubset(set(PARAMS)) for param_set in FEATURE_POSITIVE_NAME_PARAMS.values()] ), "Error! 'FEATURE_POSITIVE_NAME_PARAMS' contains unknown parameter"

for param_list in FEATURE_POSITIVE_NAME_PARAMS.values():
  param_set = set(param_list)
  assert param_set.issubset(set(PARAMS)), f"Error! the positive parameter set {param_list} contains unknown parameter(s)."

class ETMining(BasePredicateConfig):
  environment_id = 'ET-mining'

  def __init__(self, seed: int, graph_param: str, keep_pristine: bool = False, feature_mode: str = 'cluster-s', visualize: bool = False):
    self.num_graphs = 1
    self.max_step = 30
    super().__init__(
      seed=seed,
      graph_param=graph_param,
      keep_pristine=keep_pristine,
      feature_mode=feature_mode,
      visualize=visualize,
    )

  @property
  def parameters(self):
    return set(PARAMS)

  # Manually set
  def _construct_predicate_graph(self):
    """Implement predicate precondition&effect
    """
    g = PredicateLogicGraph('Mining')
    # Add all features
    HARD0_X = '(f_hardness0, A)'
    HARD1_X = '(f_hardness1, A)'
    HARD2_X = '(f_hardness2, A)' # Mine
    LIGHTABLE_X = '(f_lightable, A)' # Light
    CRAFTABLE_Y = '(f_craftable, B)'
    SMELTABLE_X = '(f_smeltable, A)' # Craft
    ISINGREDIENT_X = '(f_isingredient, A)'
    g.add_features([HARD0_X, HARD1_X, HARD2_X, LIGHTABLE_X, CRAFTABLE_Y, SMELTABLE_X, ISINGREDIENT_X])

    # Add all subtask nodes (just names)
    CRAFT_STONE_PICKAXE = '(craft, stone, pickaxe)'
    CRAFT_IRON_PICKAXE = '(craft, iron, pickaxe)'
    LIGHT_FURNACE = '(light, furnace)'
    GET_X = '(get, A)' # get
    GET_COAL = '(get, coal)'
    LIGHT_X = '(light, A)' # light
    SMELT_X = '(smelt, A)' # smelt
    CRAFT_X_Y = '(craft, A, B)' # craft
    g.add_subtasks([CRAFT_STONE_PICKAXE, CRAFT_IRON_PICKAXE, LIGHT_FURNACE, GET_X, GET_COAL, LIGHT_X, SMELT_X, CRAFT_X_Y])

    # Add each option one-by-one
    OP_GET_X = '(op_get, A)' # get
    g.add_option(
      name=OP_GET_X,
      precondition=g[HARD0_X] | (g[HARD1_X] & g[CRAFT_STONE_PICKAXE]) | (g[HARD2_X] & g[CRAFT_IRON_PICKAXE]),
      effect=g[GET_X]
    )
    OP_LIGHT_X = '(op_light, A)' # light
    g.add_option(
      name=OP_LIGHT_X,
      precondition=g[GET_COAL] & g[LIGHTABLE_X],
      effect=g[LIGHT_X]
    )
    OP_SMELT_X = '(op_smelt, A)' # smelt
    g.add_option(
      name=OP_SMELT_X,
      precondition=g[GET_X] & g[LIGHT_FURNACE] & g[SMELTABLE_X],
      effect=g[SMELT_X]
    )
    OP_CRAFT_X_Y = '(op_craft, A, B)' # craft
    AND1 = g[GET_X] & g[CRAFTABLE_Y] & g[ISINGREDIENT_X] & g[SMELTABLE_X] & g[SMELT_X]
    AND2 = g[GET_X] & g[CRAFTABLE_Y] & g[ISINGREDIENT_X] & (~g[SMELTABLE_X])
    g.add_option(
      name=OP_CRAFT_X_Y,
      #precondition=g[GET_X] & g[CRAFTABLE_Y] & ( g[SMELT_X] | (~g[SMELTABLE_X]) ),
      precondition=AND1 | AND2,
      effect=g[CRAFT_X_Y]
    )
    return g
  
  def _sample_param_set(self, graph_param):
    params = []
    if graph_param == 'train':
      superset = TRAIN_PARAMS
      mustset = TRAIN_MUST
    elif graph_param == 'eval':
      superset = EVAL_PARAMS
      mustset = list(set(EVAL_ONLY+EVAL_MUST))
    else:
      raise ValueError(f"Error! Got unknown graph_param: {graph_param}")
    assert set(mustset).issubset(set(superset)), "Error: mustset is not a subset of superset"
    sampling_params =[
      (RES_HARD0, 2, 2),
      (RES_HARD1, 3, 3),
      (RES_HARD2, 3, 3),
      (CRAFTS, 3, 3),
      (FURNACE, 1, 1),
    ]
    for elem in sampling_params:
      subset, min_p, max_p = elem
      samples = self._random_sample(self.intersect(superset, subset), min_p, max_p, mustset)
      params.extend(samples)
    return params

  # Manually set
  def _set_reward_and_termination(self, params):
    # randomly find two of RES_HARD2 as X
    # randomly find one of CRAFTS (except pickaxs) as Y
    # Add reward to craft X Y
    # Add reward to get X
    X_pool = self.intersect(RES_HARD2, params)
    Xs = np.random.permutation(X_pool)[:2]    
    Y_pool = self.intersect([par for par in CRAFTS if par != 'pickaxe'], params)
    Y = np.random.permutation(Y_pool)[0]
    
    for X in Xs:
      assert self._add_reward(name=f'(craft, {X}, {Y})', reward=1.0, terminal=True, success=True)
      assert self._add_reward(name=f'(get, {X})', reward=0.3, terminal=False, success=False)
      self._add_reward(name=f'(smelt, {X})', reward=0.3, terminal=False, success=False)

    #assert self._add_reward(name='(light, furnace)', reward=0.2, terminal=False, success=False)
    assert self._add_reward(name='(get, iron)', reward=0.1, terminal=False, success=False)
    
    #assert self._add_reward(name='(craft, stone, pickaxe)', reward=0.3, terminal=False, success=False)
    assert self._add_reward(name='(craft, iron, pickaxe)', reward=0.3, terminal=False, success=False)
      
  # Manually set
  def _construct_initial_completion(self, unrolled_graph):
    # Initialize
    completion = OrderedDict()
    for node in unrolled_graph.subtask_nodes:
      completion[node.name] = False

    assert not any(completion.values()), "Error! intially nothing should be completed"
    return completion

  def _construct_feature_positives(self):
    return FEATURE_POSITIVE_NAME_PARAMS

  def _perturb_subtasks(self, rng):
    return
