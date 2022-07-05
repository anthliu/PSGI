'''Configuration script for Target environment.'''
from collections import OrderedDict
from typing import Dict
import numpy as np

from psgi.envs.base_predicate_config import BasePredicateConfig
from psgi.envs.predicate_graph import PredicateLogicGraph

# Define parameters (no intersection allowed)
FOOD_FRUIT = ['apple', 'pear']    # pick X
FOOD_VEGI = ['cabbage', 'carrot'] # pick X / slice X
FOOD_MEAT = ['beef', 'pork']      # pick X / slice X / cook X
FOOD_PARAMS = FOOD_FRUIT + FOOD_VEGI + FOOD_MEAT

OBJECT= ['dish']
PLACE = ['table']
###
PARAMS = FOOD_PARAMS + OBJECT + PLACE
assert len(set(PARAMS)) == len(PARAMS), "Error! duplicated params!"

TRAIN_ONLY = ['apple', 'beef']
EVAL_ONLY = ['pear', 'pork']
assert all([par in PARAMS for par in TRAIN_ONLY+EVAL_ONLY]), "Unknown parameter in TRAIN_ONLY and/or EVAL_ONLY"
TRAIN_PARAMS = [par for par in PARAMS if par not in EVAL_ONLY]
EVAL_PARAMS = [par for par in PARAMS if par not in TRAIN_ONLY]

# Manually set the feature
FEATURE_POSITIVE_NAME_PARAMS = {
  'f_pickupable': FOOD_PARAMS+OBJECT,
  'f_sliceable': FOOD_PARAMS,
  'f_needslice': FOOD_VEGI + FOOD_MEAT,
  'f_cookable': FOOD_MEAT,
}
assert all([set(param_set).issubset(set(PARAMS)) for param_set in FEATURE_POSITIVE_NAME_PARAMS.values()] ), "Error! 'FEATURE_POSITIVE_NAME_PARAMS' contains unknown parameter"

class CookingClassic(BasePredicateConfig):
  environment_id = 'cooking_classic'

  def __init__(self, seed: int, graph_param: str, keep_pristine: bool = False, feature_mode: str = 'cluster-s', visualize: bool = False):
    self.num_graphs = 1
    self.max_step = 8
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
    g = PredicateLogicGraph('CookingClassic')
    # Add all features
    PICKUPABLE_X = '(f_pickupable, A)' # pick
    SLICEABLE_X = '(f_sliceable, A)' # slice
    NEEDSLICE_X = '(f_needslice, A)'
    COOKABLE_X = '(f_cookable, A)' # slice
    
    g.add_features([PICKUPABLE_X, SLICEABLE_X, NEEDSLICE_X, COOKABLE_X])

    # Add all subtask nodes (just names)
    PICKUP_X = '(pickup, A)' # pick
    SLICE_X = '(slice, A)' # slice
    COOK_X = '(cook, A)'
    g.add_subtasks([PICKUP_X, SLICE_X, COOK_X])

    # Add each option one-by-one
    OP_PICKUP_X = '(op_pickup, A)'
    g.add_option(
      name=OP_PICKUP_X,
      precondition=g[PICKUPABLE_X],
      effect=g[PICKUP_X]
    )
    # Slice option
    OP_SLICE_X = '(op_slice, A)'
    g.add_option(
      name=OP_SLICE_X,
      precondition=g[SLICEABLE_X] & g[PICKUP_X],
      effect=g[SLICE_X]
    )
    # Cook option
    OP_COOK_X = '(op_cook, A)'
    AND1 = g[COOKABLE_X] & g[PICKUP_X] & (~g[NEEDSLICE_X])
    AND2 = g[COOKABLE_X] & g[PICKUP_X] & g[SLICE_X]
    g.add_option(
      name=OP_COOK_X,
      precondition=AND1|AND2,
      effect=g[COOK_X]
    )
    return g

  def _sample_param_set(self, seed, graph_param):
    if graph_param == 'train':
      params = TRAIN_PARAMS
    elif graph_param == 'eval':
      params = EVAL_PARAMS
    else:
      raise ValueError(f"Error! Got unknown graph_param: {graph_param}")
    return params
  
  # Manually set
  def _set_reward_and_termination(self):
    if self.graph_param == 'train':
      self._add_reward(name='(cook, beef)', reward=1.0, terminal=True, success=True)
    elif self.graph_param == 'eval':
      self._add_reward(name='(cook, pork)', reward=1.0, terminal=True, success=True)
      
  # Manually set
  def _construct_initial_completion(self, unrolled_graph):
    # Initialize
    completion = OrderedDict()
    for node in unrolled_graph.subtask_nodes:
      completion[node.name] = False

    # Nothing is completed in the beginning
    return completion

  def _construct_feature_positives(self):
    return FEATURE_POSITIVE_NAME_PARAMS

  def _perturb_subtasks(self, rng):
    return
