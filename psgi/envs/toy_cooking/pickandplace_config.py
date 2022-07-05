'''Configuration script for Target environment.'''
from collections import OrderedDict
from typing import Dict
import numpy as np

from psgi.envs.base_predicate_config import BasePredicateConfig
from psgi.envs.predicate_graph import PredicateLogicGraph

# Define parameters (no intersection allowed)
FOOD_FRUIT = ['apple', 'pear']    # pick A
FOOD_VEGI = ['cabbage'] # pick A / slice A
FOOD_PARAMS = FOOD_FRUIT + FOOD_VEGI
###
PLACE_RECEP = ['fridge']  # put B
PLACE_BOARD = ['table', 'sideboard']  # put B / slice B
PLACE_PARAMS = PLACE_RECEP + PLACE_BOARD
###
DISTRACTOR = ['microwave']
PARAMS = FOOD_PARAMS + PLACE_PARAMS + DISTRACTOR
assert len(set(PARAMS)) == len(PARAMS), "Error! duplicated params!"

TRAIN_ONLB = ['apple', 'table']
EVAL_ONLB = ['pear', 'sideboard']
assert all([par in PARAMS for par in TRAIN_ONLB+EVAL_ONLB]), "Unknown parameter in TRAIN_ONLB and/or EVAL_ONLB"
TRAIN_PARAMS = [par for par in PARAMS if par not in EVAL_ONLB]
EVAL_PARAMS = [par for par in PARAMS if par not in TRAIN_ONLB]

# Manually set the feature
FEATURE_POSITIVE_NAME_PARAMS = {
  'f_pickupable': FOOD_PARAMS,
  'f_isplace': PLACE_PARAMS,
}

class PickPlace(BasePredicateConfig):
  environment_id = 'pickplace'

  def __init__(self, seed: int, graph_param: str, keep_pristine: bool = False, feature_mode: str = 'cluster-s', visualize: bool = False):
    self.num_graphs = 1
    self.max_step = 10
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
    g = PredicateLogicGraph('PickPlace')
    # Add all features
    PICKUPABLE_A = '(f_pickupable, A)'
    PUTABLE_B = '(f_isplace, B)'
    g.add_features([PICKUPABLE_A, PUTABLE_B])
    # Add all subtask nodes (just names)
    PUT_A_B = '(put, A, B)'
    PICKUP_A = '(pickup, A)'
    g.add_subtasks([PUT_A_B, PICKUP_A])

    # Add each option one-by-one
    OP_PUT_A_B = '(op_put, A, B)'
    g.add_option(
      name=OP_PUT_A_B,
      precondition=g[PICKUP_A] & g[PUTABLE_B],
      effect=(~g[PICKUP_A]) & g[PUT_A_B]
    )
    OP_PICKUP_A_B = '(op_pickup, A, B)'
    g.add_option(
      name=OP_PICKUP_A_B,
      precondition=g[PUT_A_B] & g[PICKUPABLE_A],
      effect=(~g[PUT_A_B]) & g[PICKUP_A]
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
      self._add_reward(name='(put, apple, fridge)', reward=1.0, terminal=True, success=True)
    elif self.graph_param == 'eval':
      self._add_reward(name='(put, pear, fridge)', reward=1.0, terminal=True, success=True)

  # Manually set
  def _construct_initial_completion(self, unrolled_graph):
    # Initialize
    completion = OrderedDict()
    for node in unrolled_graph.subtask_nodes:
      completion[node.name] = False

    # Manually set the initial completion
    _init_place = 'table' if self.graph_param == 'train' else 'sideboard'
    for x in FOOD_PARAMS:
      subtask_name = f'(put, {x}, {_init_place})'
      if subtask_name in completion:
        completion[subtask_name] = True
    assert any(completion.values()), "Error! no completed subtask in the beginning!"
    return completion
  
  def _construct_feature_positives(self):
    return FEATURE_POSITIVE_NAME_PARAMS

  def _perturb_subtasks(self, rng):
    return
