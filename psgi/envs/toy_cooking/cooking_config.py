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
FOOD_FRUIT = ['apple', 'pear', 'orange', 'strawberry']    # pick X
FOOD_VEGI = ['cabbage', 'carrot', 'lettuce', 'cucumber'] # pick X / slice X
FOOD_MEAT = ['beef', 'pork', 'chicken', 'lamb']      # pick X / slice X / cook X
FOOD_PARAMS = FOOD_FRUIT + FOOD_VEGI + FOOD_MEAT
###
OBJECT_COOKWARE = ['pan', 'pot', 'wok']  # pick X / put Y / cook Y
PLACE_RECEP = ['fridge','shelf','rack', 'cupboard']  # put Y
PLACE_OVEN = ['stove'] # put Y / cook Z
PLACE_BOARD = ['table', 'desk']  # put Y / slice Y
PLACE_PARAMS = PLACE_RECEP + PLACE_OVEN + PLACE_BOARD
PARAMS = FOOD_PARAMS + OBJECT_COOKWARE + PLACE_PARAMS
assert len(set(PARAMS)) == len(PARAMS), "Error! duplicated params!"

TRAIN_ONLY = ['apple', 'cabbage', 'beef', 'table']
EVAL_ONLY = ['pear', 'carrot', 'pork', 'desk']
TRAIN_MUST = ['shelf'] # usually due to initial completion and reward
EVAL_MUST = ['shelf', 'pork'] # usually due to initial completion and reward
assert all([par in PARAMS for par in TRAIN_ONLY+EVAL_ONLY]), "Unknown parameter in TRAIN_ONLY and/or EVAL_ONLY"
assert all([par not in EVAL_ONLY for par in TRAIN_ONLY]), "there is a overlap between TRAIN_ONLY and EVAL_ONLY"
TRAIN_PARAMS = [par for par in PARAMS if par not in EVAL_ONLY]
EVAL_PARAMS = [par for par in PARAMS if par not in TRAIN_ONLY]

# Manually set the feature
FEATURE_POSITIVE_NAME_PARAMS = {
  'f_pickupable': FOOD_PARAMS + OBJECT_COOKWARE,
  'f_isplace': PLACE_PARAMS + OBJECT_COOKWARE,
  'f_isfood': FOOD_PARAMS,
  'f_isboard': PLACE_BOARD,
  'f_iscookware': OBJECT_COOKWARE,
  'f_needslice': FOOD_VEGI,
  'f_iscookable': FOOD_MEAT,
}
assert all([set(param_set).issubset(set(PARAMS)) for param_set in FEATURE_POSITIVE_NAME_PARAMS.values()] ), "Error! 'FEATURE_POSITIVE_NAME_PARAMS' contains unknown parameter"

class Cooking(BasePredicateConfig):
  environment_id = 'cooking'

  def __init__(self, seed: int, graph_param: str, keep_pristine: bool = False, feature_mode: str = 'cluster-s', visualize: bool = False):
    self.num_graphs = 1
    self.max_step = 25
    super().__init__(
      seed=seed,
      graph_param=graph_param,
      keep_pristine=keep_pristine,
      feature_mode=feature_mode,
      visualize=visualize,
    )

  # Manually set
  def _construct_predicate_graph(self):
    """Implement predicate precondition&effect
    """
    g = PredicateLogicGraph('Cooking')
    # Add all features
    #PUTABLE_Y = '(putable, B)' # put
    PICKUPABLE_X = '(f_pickupable, A)' # pick
    ISPLACE_Y = '(f_isplace, B)'
    ISFOOD_X = '(f_isfood, A)' # slice
    ISBOARD_Y = '(f_isboard, B)'
    ISCOOKWARE_X = '(f_iscookware, A)' # cook
    ISCOOKWARE_Y = '(f_iscookware, B)' # cook
    NEEDSLICE_X = '(f_needslice, A)'
    g.add_features([PICKUPABLE_X, ISPLACE_Y, ISFOOD_X, ISBOARD_Y, ISCOOKWARE_X, ISCOOKWARE_Y, NEEDSLICE_X])

    # Add all subtask nodes (just names)
    PUT_X_Y = '(put, A, B)' # put
    PICKUP_X = '(pickup, A)' # pick
    SLICE_X = '(slice, A)' # slice
    PUT_Y_STOVE = '(put, B, stove)' # cook
    COOK_X = '(cook, A)'
    g.add_subtasks([PUT_X_Y, PICKUP_X, SLICE_X, PUT_Y_STOVE, COOK_X])

    # Add each option one-by-one
    OP_PUT_X_Y = '(op_put, A, B)'
    AND1 = g[PICKUP_X] & g[PICKUPABLE_X] & g[ISPLACE_Y] & (~g[ISCOOKWARE_Y])
    AND2 = g[PICKUP_X] & g[PICKUPABLE_X] & g[ISPLACE_Y] & (~g[ISCOOKWARE_X])
    g.add_option(
      name=OP_PUT_X_Y,
      #precondition=g[PICKUPABLE_X] & g[PICKUP_X] & g[ISPLACE_Y],
      precondition=AND1 | AND2,
      effect=(~g[PICKUP_X]) & g[PUT_X_Y]
    )
    OP_PICKUP_X_Y = '(op_pickup, A, B)'
    g.add_option(
      name=OP_PICKUP_X_Y,
      precondition=g[PUT_X_Y] & g[PICKUPABLE_X]& g[ISPLACE_Y],
      effect=(~g[PUT_X_Y]) & g[PICKUP_X]
    )
    # Slice option
    OP_SLICE_X_Y = '(op_slice, A, B)'
    g.add_option(
      name=OP_SLICE_X_Y,
      precondition=g[PUT_X_Y] & g[ISFOOD_X] & g[ISBOARD_Y],
      effect=g[SLICE_X]
    )
    # Cook option
    OP_COOK_X_Y = '(op_cook, A, B)'
    # g[PUT_X_Y] & g[ISFOOD_X] & g[ISCOOKWARE_Y] & g[PUT_Y_STOVE] &(g[NEEDSLICE_X]|g[SLICE_X])
    AND1 = g[ISFOOD_X] & g[PUT_X_Y] & g[ISCOOKWARE_Y] & g[PUT_Y_STOVE] & (~g[NEEDSLICE_X])
    AND2 = g[ISFOOD_X] & g[PUT_X_Y] & g[ISCOOKWARE_Y] & g[PUT_Y_STOVE] & g[SLICE_X]
    g.add_option(
      name=OP_COOK_X_Y,
      precondition=AND1|AND2,
      effect=g[COOK_X]
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
      (FOOD_FRUIT, 1, 1),
      (FOOD_VEGI, 2, 2),
      (FOOD_MEAT, 1, 1),
      (OBJECT_COOKWARE, 2, 2),
      (PLACE_RECEP, 1, 1),
      (PLACE_OVEN, 1, 1),
      (PLACE_BOARD, 1, 1),
    ]
    for elem in sampling_params:
      subset, min_p, max_p = elem
      samples = self._random_sample(self.intersect(superset, subset), min_p, max_p, mustset)
      params.extend(samples)
    return params
  
  # Manually set
  def _set_reward_and_termination(self, params):
    # randomly find one of MEAT as X
    X_pool = self.intersect(FOOD_MEAT, params)
    X = np.random.permutation(X_pool)[0] 
    assert self._add_reward(name=f'(cook, {X})', reward=1.0, terminal=True, success=True)
    self._add_reward(name=f'(put, {X}, pan)', reward=0.2, terminal=False, success=False)
    self._add_reward(name=f'(put, {X}, pot)', reward=0.2, terminal=False, success=False)
    self._add_reward(name=f'(put, {X}, wok)', reward=0.2, terminal=False, success=False)
    
    # partial
    self._add_reward(name='(put, pan, stove)', reward=0.2, terminal=False, success=False)
    self._add_reward(name='(put, pot, stove)', reward=0.2, terminal=False, success=False)
    self._add_reward(name='(put, wok, stove)', reward=0.2, terminal=False, success=False)
      
  # Manually set
  def _construct_initial_completion(self, unrolled_graph):
    # Initialize
    completion = OrderedDict()
    for node in unrolled_graph.subtask_nodes:
      completion[node.name] = False

    # Manually set the initial completion
    for x in FOOD_PARAMS + OBJECT_COOKWARE:
      subtask_name = f'(put, {x}, shelf)'
      if subtask_name in completion:
        completion[subtask_name] = True
    assert any(completion.values()), "Error! no completed subtask in the beginning!"
    return completion

  @property
  def parameters(self):
    return set(PARAMS)

  def _construct_feature_positives(self):
    return FEATURE_POSITIVE_NAME_PARAMS

  def _perturb_subtasks(self, rng):
    return
