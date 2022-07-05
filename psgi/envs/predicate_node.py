from copy import deepcopy

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
from psgi.utils.predicate_utils import Predicate, Symbol
from psgi.envs.logic_graph import LogicOp, SubtaskVertex, _extract_leaf_nodes

def _parse_name(name: str):
  import re
  name = name.replace(' ', '') # remove space
  param_name_splitted = re.split('[(),]',name)
  param_name_splitted = [name for name in param_name_splitted if len(name) > 0] # remove empty splits
  return param_name_splitted

def name_to_param(name: str) -> Predicate:
  tokens = _parse_name(name)
  param_list = []
  for token in tokens:
    if token.isupper():
      param_list.append(Symbol(set(), token))
    else:
      param_list.append(token)
  return Predicate(param_list)

class PNode(SubtaskVertex):
  def __init__(self, name: str, node_type: str):
    super().__init__(name)
    self._node_type = node_type
    self._param = name_to_param(name)
    assert self._param.pretty() == name, f"Error: the name {name} and parameter {self._param.pretty()} does not match!"

  def get_predicate_with_param(self, param_pool):
    predicate_params = deepcopy(self._param.parameters)
    for param in predicate_params:
        if isinstance(param, Symbol):
          param.set_values(param_pool)
    return Predicate(predicate_params)
    
  def update_name(self):
    self._name = self._param.pretty()
  @property
  def param(self) -> tuple:
    return self._param
  @property
  def node_type(self) -> tuple:
    return self._node_type
    
### Additional features to logic graph
class OptionNode(PNode):
  def __init__(self, name: str, *, precondition: LogicOp, effect: LogicOp):
    super().__init__(name=name, node_type='option')
    self._precondition = precondition
    self._effect = effect
    self._validate()

  @property
  def precondition(self) -> LogicOp:
    return self._precondition
  @property
  def effect(self) -> LogicOp:
    return self._effect
    
  def _validate(self):
    # Check if precond, effect has any parameter that was never used in option
    option_param_names = {str(symbol) for symbol in self._param}
    precond_leaf_node_names = _extract_leaf_nodes(self.precondition)
    effect_leaf_node_names = _extract_leaf_nodes(self.effect)
    arg_set = set()
    for node_name in precond_leaf_node_names + effect_leaf_node_names:
      curr_argset = node_name.split(',')[1:]
      arg_set.update( curr_argset )
    arg_set = {arg for arg in arg_set if 'Symbol' in arg}
    if not arg_set.issubset(option_param_names):
      print(f'Error! Precondition or Effect has free variable that was never used in option! {arg_set} not in {option_param_names}')
      import ipdb; ipdb.set_trace()
