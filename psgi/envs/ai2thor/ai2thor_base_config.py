'''Configuration script for Target environment.'''
from collections import OrderedDict
from typing import Dict
import numpy as np

from psgi.envs.base_predicate_config import BasePredicateConfig
from psgi.envs.predicate_graph import PredicateLogicGraph
from psgi.envs.ai2thor.entity_constants import FP_TO_ENTITY, FP_TO_FEATURES_POSITIVES, FP_TO_COMPAT_FEATURES, COMPAT_FEATURES

class AI2ThorConfig(BasePredicateConfig):
  environment_id = 'ai2thor'

  def __init__(self, seed: int, graph_param: str, keep_pristine: bool = False, feature_mode: str = 'cluster-s', visualize: bool = False):
    self.num_graphs = 1
    self.max_step = 14
    self.floorplan = None# initialize floorplan when calling _sample_param_set
    super().__init__(
      seed=seed,
      graph_param=graph_param,
      keep_pristine=keep_pristine,
      feature_mode=feature_mode,
      visualize=visualize,
    )
  
  @property
  def parameters(self):
    return set(self.param_set)
  
  # Manually set
  def _construct_predicate_graph(self):
    """Implement predicate precondition&effect
    """
    g = PredicateLogicGraph('AI2Thor')
    # Add all features
    ISPICKUPABLE_A = '(f_Pickupable, A)'
    ISRECEPTABLE_B = '(f_Receptacle, B)'
    ISSLICEABLE_A = '(f_Sliceable, A)'
    ISCOOKABLE_A = '(f_Cookable, A)'
    ISBREAD_A = '(f_Bread, A)'
    ISSTOVEBURNER_B = '(f_StoveBurner, B)'
    ISTOASTER_B = '(f_Toaster, B)'
    ISMICROWAVE_B = '(f_Microwave, B)'
    ISFLOOR_B = '(f_Floor, B)'
    g.add_features([
      ISPICKUPABLE_A, ISRECEPTABLE_B,
      ISSLICEABLE_A, ISCOOKABLE_A,
      ISBREAD_A,
      ISSTOVEBURNER_B, ISTOASTER_B, ISMICROWAVE_B,
      ISFLOOR_B
    ])
    for compat_feat in COMPAT_FEATURES:
      g.add_features([f'({compat_feat}, A)', f'({compat_feat}, B)'])

    # Add all subtask nodes (just names)
    PUT_AB = '(put, A, B)'
    PICKUP_A = '(pickup, A)'
    FULL_B = '(full, B)'
    HANDFULL = '(handfull)'
    SLICE_A = '(slice, A)'
    COOK_A = '(cook, A)'
    PUT_A_PAN = '(put, A, Pan)'
    PUT_A_POT = '(put, A, Pot)'
    PUT_PAN_B = '(put, Pan, B)'
    PUT_POT_B = '(put, Pot, B)'
    g.add_subtasks([
      PUT_AB, PICKUP_A, FULL_B, HANDFULL,
      SLICE_A, COOK_A,
      PUT_A_PAN, PUT_A_POT,
      PUT_PAN_B, PUT_POT_B
    ])

    # Add each option one-by-one
    OP_PUT_AB = '(op_put, A, B)'
    AND1 = None
    for compat_feat in COMPAT_FEATURES:
      AND1_next = g[PICKUP_A] & g[f'({compat_feat}, A)'] & g[f'({compat_feat}, B)'] & g[ISRECEPTABLE_B] & (~g[FULL_B])
      if AND1 is None:
        AND1 = AND1_next
      else:
        AND1 = AND1 | AND1_next
    AND2 = g[PICKUP_A] & g[ISFLOOR_B]
    g.add_option(
      name=OP_PUT_AB,
      precondition=AND1 | AND2,
      effect=(~g[PICKUP_A]) & g[PUT_AB] & g[FULL_B] & (~g[HANDFULL])
    )
    OP_PICKUP_AB = '(op_pickup, A, B)'
    g.add_option(
      name=OP_PICKUP_AB,
      precondition=g[PUT_AB] & g[ISPICKUPABLE_A] & (~g[HANDFULL]),
      effect=(~g[PUT_AB]) & g[PICKUP_A] & (~g[FULL_B]) & g[HANDFULL]
    )
    OP_SLICE_AB = '(op_slice, A, B)'
    g.add_option(
      name=OP_SLICE_AB,
      precondition=g[ISSLICEABLE_A] & g[PUT_AB],
      effect=g[SLICE_A]
    )
    OP_COOK_AB = '(op_cook, A, B)'
    #AND1 = g[ISCOOKABLE_A] & (~g[COOK_A]) & g[ISSTOVEBURNER_B] & g[PUT_A_PAN] & g[PUT_PAN_B] & (~g[ISSLICEABLE_A])
    #AND2 = g[ISCOOKABLE_A] & (~g[COOK_A]) & g[ISSTOVEBURNER_B] & g[PUT_A_PAN] & g[PUT_PAN_B] & g[SLICE_A]
    #AND3 = g[ISCOOKABLE_A] & (~g[COOK_A]) & g[ISSTOVEBURNER_B] & g[PUT_A_POT] & g[PUT_POT_B] & (~g[ISSLICEABLE_A])
    #AND4 = g[ISCOOKABLE_A] & (~g[COOK_A]) & g[ISSTOVEBURNER_B] & g[PUT_A_POT] & g[PUT_POT_B] & g[SLICE_A]
    #AND5 = g[ISCOOKABLE_A] & (~g[COOK_A]) & g[ISMICROWAVE_B] & g[PUT_AB] & g[SLICE_A]
    #AND6 = g[ISCOOKABLE_A] & (~g[COOK_A]) & g[ISMICROWAVE_B] & g[PUT_AB] & (~g[ISSLICEABLE_A])
    #AND7 = g[ISCOOKABLE_A] & (~g[COOK_A]) & g[ISTOASTER_B] & g[PUT_AB] & g[SLICE_A]
    AND1 = g[ISCOOKABLE_A] & g[ISSTOVEBURNER_B] & g[PUT_A_PAN] & g[PUT_PAN_B] & (~g[ISSLICEABLE_A])
    AND2 = g[ISCOOKABLE_A] & g[ISSTOVEBURNER_B] & g[PUT_A_PAN] & g[PUT_PAN_B] & g[SLICE_A]
    AND3 = g[ISCOOKABLE_A] & g[ISSTOVEBURNER_B] & g[PUT_A_POT] & g[PUT_POT_B] & (~g[ISSLICEABLE_A])
    AND4 = g[ISCOOKABLE_A] & g[ISSTOVEBURNER_B] & g[PUT_A_POT] & g[PUT_POT_B] & g[SLICE_A]
    AND5 = g[ISCOOKABLE_A] & g[ISMICROWAVE_B] & g[PUT_AB] & g[SLICE_A]
    AND6 = g[ISCOOKABLE_A] & g[ISMICROWAVE_B] & g[PUT_AB] & (~g[ISSLICEABLE_A])
    AND7 = g[ISBREAD_A] & g[ISTOASTER_B] & g[PUT_AB] & g[SLICE_A]
    g.add_option(
      name=OP_COOK_AB,
      precondition=AND1 | AND2 | AND3 | AND4 | AND5 | AND6 | AND7,
      effect=g[COOK_A]
    )
    return g
  
  def _sample_param_set(self, graph_param):
    if graph_param == 'train':
      self.floorplan = 1 + np.random.choice(20)
    else:
      self.floorplan = 21 + np.random.choice(10)

    entities = FP_TO_ENTITY[self.floorplan]
    feats = FP_TO_FEATURES_POSITIVES[self.floorplan]
    essential = []
    nonessential = []
    for entity in entities:
      ent_type = entity.split('_')[-1]
      if ent_type in ['Floor', 'StoveBurner', 'Pot', 'Pan', 'Microwave', 'Toaster']:
        essential.append(entity)
      elif entity in feats['f_Cookable']:
        essential.append(entity)
      else:
        nonessential.append(entity)

    sampled = essential + list(np.random.default_rng(seed=self.seed).choice(nonessential, size=10, replace=False))
    return sampled

  # Manually set
  def _set_reward_and_termination(self, params):
    cookables = list(set(FP_TO_FEATURES_POSITIVES[self.floorplan]['f_Cookable']))
    target_food = cookables[np.random.choice(len(cookables))]# random sample from cookables
    self._add_reward(name=f'(cook, {target_food})', reward=1.0, terminal=True, success=True)
    self._add_reward(name=f'(put, {target_food}, Pan)', reward=0.2, terminal=False, success=False)
    self._add_reward(name=f'(put, {target_food}, Pot)', reward=0.2, terminal=False, success=False)
    self._add_reward(name=f'(put, Pan, StoveBurner)', reward=0.2, terminal=False, success=False)
    self._add_reward(name=f'(put, Pot, StoveBurner)', reward=0.2, terminal=False, success=False)

  # Manually set
  def _construct_initial_completion(self, unrolled_graph):
    # Initialize
    completion = OrderedDict()
    for node in unrolled_graph.subtask_nodes:
      completion[node.name] = False

    params = self.parameters
    _init_place = 'Floor'
    for x in FP_TO_FEATURES_POSITIVES[self.floorplan]['f_Pickupable']:
      subtask_name = f'(put, {x}, {_init_place})'
      if subtask_name in completion:
        completion[subtask_name] = True
    assert any(completion.values()), "Error! no completed subtask in the beginning!"
    return completion
  
  def _construct_feature_positives(self):
    features = FP_TO_FEATURES_POSITIVES[self.floorplan]
    features.update(FP_TO_COMPAT_FEATURES[self.floorplan])
    return features

  def _perturb_subtasks(self, rng):
    return

def _testing():
  from pprint import pprint
  #print(FP_TO_ENTITY[28])
  #print(list(FP_TO_FEATURES_POSITIVES[28].keys()))
  #pprint(OBJ_REC_NOTCOMP)
  cfg = AI2ThorConfig(0, 'train', feature_mode='cluster-s', visualize=True)

if __name__ == '__main__':
  _testing()
