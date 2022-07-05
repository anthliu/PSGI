from copy import deepcopy
from typing import Union
import itertools

import numpy as np

class Symbol(object):
  def __init__(self, values: set, name=None):
    assert isinstance(values, set)
    self.values = values
    self.name = name

  def merge(self, other):
    if isinstance(other, Symbol):
      self.values.update(other.values)
    else:
      self.values.add(other)

  def set_values(self, values: set):
    self.values = values

  def __eq__(self, other):
    return isinstance(other, Symbol) and self.name == other.name and self.values == other.values

  def issubset(self, other):
    return isinstance(other, Symbol) and self.values.issubset(other.values)

  def __hash__(self):
    return hash(repr(self))

  def __repr__(self):
    if self.values == set():
      return f'Symbol({self.name})'
    else:
      if self.name is None:
        return f'Symbol{repr(list(sorted(self.values)))}'
      else:
        return f'{self.name}{repr(list(sorted(self.values)))}'

  def pretty(self):
    return str(self.name)

  @staticmethod
  def ground_assignments(symbols):
    symb_names = []
    symb_values = []
    for symb in symbols:
      symb_names.append(symb.name)
      symb_values.append(list(sorted(symb.values)))
    new_psubtasks = []
    for assignment in itertools.product(*symb_values):
      yield dict(zip(symb_names, assignment))

class Feature(object):
  def __init__(self, symbol: Symbol, name=None):
    assert symbol.name is not None
    self.symbol = symbol
    self.name = name

  def __eq__(self, other):
    return isinstance(other, Feature) and self.symbol == other.symbol

  def __hash__(self):
    return hash(repr(self))

  def __repr__(self):
    if self.name is None:
      return f'f({repr(self.symbol)})'
    else:
      return f'({self.name}, {repr(self.symbol)})'

  def compute(self, assignments):
    for symbol_name, new_value in assignments.items():
      if self.symbol.name == symbol_name:
        assert new_value in self.symbol.values, 'Invalid feature symbol assignment'
        return new_value
    raise ValueError('No matching symbol found to compute feature')

  def pretty(self):
    if self.name is None:
      return f'f({self.symbol.pretty()})'
    else:
      return f'({self.name}, {self.symbol.pretty()})'

class Predicate(object):
  def __init__(self, parameters):
    self.parameters = list(parameters)# list of ints/strings or Symbols
    self.nparams = len(parameters)

  def __getitem__(self, key):
    return self.parameters[key]

  def __eq__(self, other):
    return (self.nparams == other.nparams) and all(param == oparam for param, oparam in zip(self.parameters, other.parameters))

  def issubset(self, other):
    if self.nparams != other.nparams:
      return False

    for s_a, s_b in zip(self.parameters, other.parameters):
      if isinstance(s_a, Symbol):
        if not s_a.issubset(s_b):
          return False
      else:
        if isinstance(s_b, Symbol):
          if not s_a in s_b.values:
            return False
        else:
          if s_a != s_b:
            return False
    return True

  def equal_except(self, other, idx):
    # return whether subtasks are equal at all parameters except at idx
    return (self.nparams == other.nparams)\
        and all(param == oparam for param, oparam in zip(self.parameters[:idx], other.parameters[:idx]))\
        and all(param == oparam for param, oparam in zip(self.parameters[idx+1:], other.parameters[idx+1:]))

  def merge_at(self, other, idx):
    # merge with other subtask at index idx
    # ex. (0, 1, 2).merge_at((0, 2, 2), 1) = (0, {1, 2}, 2)
    assert self.equal_except(other, idx)
    if not isinstance(self.parameters[idx], Symbol):
      self.parameters[idx] = Symbol({self.parameters[idx]})
    self.parameters[idx].merge(other[idx])

  def __repr__(self):
    return f'Predicate{repr(self.parameters)}'

  def pretty(self):
    pparams = ', '.join([p.pretty() if isinstance(p, Symbol) else str(p) for p in self.parameters])
    return f'({pparams})'

  def __hash__(self):
    return hash(repr(self))

  def has_symbol(self):
    return any([isinstance(p, Symbol) for p in self.parameters])

  @property
  def symbols(self):
    return [p for p in self.parameters if isinstance(p, Symbol)]

  def symbol_by_name(self, name):
    return next(symb for symb in self.symbols if symb.name == name)

  def substitute(self, assignments):
    assign_lookup = dict(assignments)
    for idx in range(len(self.parameters)):
      if isinstance(self.parameters[idx], Symbol):
        if self.parameters[idx].name in assign_lookup:
          self.parameters[idx] = assign_lookup[self.parameters[idx].name]

  def get_reducing_assignment(self, other):
    # ex. Pred(0, A).get_reducing_assignment(Pred(0, 1)) => {A: 1}
    assert other.issubset(self)
    assignment = dict()
    for idx in range(len(self.parameters)):
      if isinstance(self.parameters[idx], Symbol):
        assignment[self.parameters[idx].name] = other.parameters[idx]
    return assignment

  def ground_assignments(self, except_symbols=[]):
    symbols = [symb for symb in self.symbols if symb.name not in except_symbols]
    yield from Symbol.ground_assignments(symbols)

  def ground(self, except_symbols=[]):
    new_psubtasks = []
    for assignment in self.ground_assignments(except_symbols):
      _new_psubtask = deepcopy(self)
      _new_psubtask.substitute(assignment)
      new_psubtasks.append(_new_psubtask)
    return new_psubtasks

  @staticmethod
  def permuted_subset_matching(symbols_from, symbols_to):
    # find all possible combinations of substitutions of symbols.
    # A Symbol_from can substitute symbol_to if it is a subset.
    if len(symbols_from) == 0 or len(symbols_to) == 0:
      return []
    substitutions = []
    for idx_from, s_from in enumerate(symbols_from):
      s_candidates = [s_to for s_to in symbols_to if s_from.issubset(s_to)]
      for s_cand in s_candidates:
        substitution = (s_from.name, s_cand.name)
        new_symbols_from = symbols_from[:idx_from] + symbols_from[idx_from+1:]
        new_symbols_to = [s_to for s_to in symbols_to if s_to.name != s_cand.name]
        new_substitutions = Predicate.permuted_subset_matching(new_symbols_from, new_symbols_to)
        if len(new_substitutions) > 0:
          for n_sub in new_substitutions:
            n_sub.insert(0, substitution)
        else:
          new_substitutions.append([substitution])
        substitutions.extend(new_substitutions)
    return substitutions

  def predicate_lub(self, other):
    assert all(symb.name is not None for symb in self.symbols+other.symbols)
    substitutions = Predicate.permuted_subset_matching(self.symbols, other.symbols)
    lub = []
    for substitution in substitutions:
      new_psubtask = deepcopy(other)
      fix_symbol_names = []
      sub_dict = {}
      for s_from, s_to in substitution:
        fix_symbol_names.append(s_from)
        sub_dict[s_to] = self.symbol_by_name(s_from)
      new_psubtask.substitute(sub_dict)
      # Try all possible groundings:
      for fix_n in range(1, len(fix_symbol_names)+1):
        for fix_n_symbol_names in itertools.combinations(fix_symbol_names, fix_n):
          for grounded_p_subtask in new_psubtask.ground(except_symbols=fix_n_symbol_names):
          # TODO XXX Do not add grounded subtasks where there is a duplicated symbol, e.g. (place, X, X). Do we want to consider these general cases
            if len(set(symb.name for symb in grounded_p_subtask.symbols)) != len(grounded_p_subtask.symbols):
              continue
            lub.append(grounded_p_subtask)

    return list(set(lub))

class PCandidateGraph(object):
  def __init__(self, option_literals, subtask_literals,
      infer_literal_subtasks=True,
      infer_partial_subtasks=True,
      infer_predicate_subtasks=True,
      infer_literal_options=False,
      infer_predicate_options=True):
    self.option_literals = option_literals
    self.subtask_literals = subtask_literals

    self.infer_literal_subtasks = infer_literal_subtasks
    self.infer_partial_subtasks = infer_partial_subtasks
    self.infer_predicate_subtasks = infer_predicate_subtasks
    self.infer_literal_options = infer_literal_options
    self.infer_predicate_options = infer_predicate_options

    self._compute_predicates()
    self._index_symbols()
    self._compute_candidates()

  def _substitution_pass(self, p_subtasks):
    # Try to make a substitution at each parameter location at each subtask with every other subtask
    start_idx = 0
    while start_idx < len(p_subtasks):
      # XXX try to substitute later parameters first (priority)
      # XXX Do not substitute first parameter (verb) TODO remove this condition?
      for param_idx in reversed(range(1, p_subtasks[start_idx].nparams)):
        _p_subtasks = p_subtasks[:start_idx]
        subtask_merge = p_subtasks[start_idx]
        subtasks_not_merged = []
        for next_idx in range(start_idx+1, len(p_subtasks)):
          if subtask_merge.equal_except(p_subtasks[next_idx], param_idx):
            subtask_merge.merge_at(p_subtasks[next_idx], param_idx)
          else:
            subtasks_not_merged.append(p_subtasks[next_idx])
        p_subtasks = _p_subtasks + [subtask_merge] + subtasks_not_merged
      start_idx += 1
    return p_subtasks

  def _substitution(self, p_subtasks):
    _p_subtasks = self._substitution_pass(deepcopy(p_subtasks))
    while p_subtasks != _p_subtasks:
      p_subtasks = _p_subtasks
      _p_subtasks = self._substitution_pass(p_subtasks)
    return p_subtasks

  def _compute_predicates(self):
    # Simple greedy substitution to find predicates
    # TODO can result in different predicates. Is there optimum
    self.literal_p_options = [Predicate(st) for st in self.option_literals]
    self.literal_p_option_to_idx = {st: idx for idx, st in enumerate(self.literal_p_options)}
    self.target_p_options = self._substitution(self.literal_p_options)

    self.literal_p_subtasks = [Predicate(st) for st in self.subtask_literals]
    self.literal_p_subtask_to_idx = {st: idx for idx, st in enumerate(self.literal_p_subtasks)}
    self.target_p_subtasks = self._substitution(self.literal_p_subtasks)

  def _index_symbols(self):
    cur_name = 'A'
    self.symbols = []
    for psubtask in self.target_p_options+self.target_p_subtasks:
      used_names = set()
      for symb in psubtask.symbols:
        for seen_symb in self.symbols:
          if symb.values == seen_symb.values:
            if seen_symb.name in used_names:# can't have same symbol name in same subtask
              continue
            symb.name = seen_symb.name
            used_names.add(symb.name)
            break
        if symb.name is None:
          symb.name = cur_name
          cur_name = chr(ord(cur_name)+1)
          self.symbols.append(symb)
          used_names.add(symb.name)

  def _compute_candidates(self):
    self.candidates = {}
    all_candidates = set(self.target_p_subtasks)
    for target_poption in self.target_p_options:
      self.candidates[target_poption] = []
      for psubtask in self.target_p_subtasks:
        candidate_list = target_poption.predicate_lub(psubtask)
        if not self.infer_partial_subtasks:
          # filter to only candidates with only symbols except one
          candidate_list = [cand for cand in candidate_list if len(cand.symbols) + 1 >= cand.nparams]
        self.candidates[target_poption].extend(candidate_list)
        all_candidates.update(candidate_list)

    self.candidate_p_subtasks = list(sorted(all_candidates, key=lambda d: (d.nparams - len(d.symbols), d.pretty())))# sort from fully parameterized to partial
    self.candidate_p_subtask_to_idx = {d: idx for idx, d in enumerate(self.candidate_p_subtasks)}

  def precompute_pilp_indices(self, num_parameters, feature_dim, feature_labels=None, param_to_idx=None):
    # precompute all pilp indices at once

    if param_to_idx is None:
      param_to_idx = list(range(num_parameters))# assume parameters are already in idx form, use identity func
    if feature_labels is None:
      feature_labels = list(range(feature_dim))
    assert len(feature_labels) == feature_dim

    # Indexing computation. First calculate total size of resulting table (including all predicate subtasks and feature functions)
    self.all_p_options = []
    self.all_p_subtasks = []
    self.all_features = []
    self.feature_dim = feature_dim
    num_literal_options = len(self.literal_p_options)
    num_target_p_options = len(self.target_p_options)
    self.all_p_option_size = 0
    if self.infer_predicate_options:
      self.all_p_options.extend(self.target_p_options)
      self.all_p_option_size += num_target_p_options
    if self.infer_literal_options:
      self.all_p_options.extend(self.literal_p_options)
      self.all_p_option_size += num_literal_options

    self.all_p_option_to_idx = {st: idx for idx, st in enumerate(self.all_p_options)}
    #self.target_p_option_idxs = [self.all_p_option_to_idx[target_p_option] for target_p_option in self.target_p_options]
    assert self.all_p_option_size == len(self.all_p_options)

    num_literal_subtasks = len(self.literal_p_subtasks)
    num_candidate_p_subtasks = len(self.candidate_p_subtasks)
    self.all_p_subtask_size = 0
    if self.infer_predicate_subtasks:
      self.all_p_subtasks.extend(self.candidate_p_subtasks)
      self.all_p_subtask_size += num_candidate_p_subtasks
    if self.infer_literal_subtasks:
      self.all_p_subtasks.extend(self.literal_p_subtasks)
      self.all_p_subtask_size += num_literal_subtasks
    self.feature_start_idx = self.all_p_subtask_size
    self.all_p_subtask_size += len(self.symbols) * feature_dim

    # check for no symbol case, if so set dummy symbol value
    if self.all_p_option_size == num_literal_options and\
        self.all_p_subtask_size == num_literal_subtasks + len(self.symbols) * feature_dim:
      for symbol in self.symbols:
        symbol.values = {symbol.values.pop()}

    for idx, symbol in enumerate(self.symbols):
      for feature_idx, feature in enumerate(feature_labels):
        feature = Feature(symbol, name=feature)
        self.all_features.append(feature)
        self.all_p_subtasks.append(feature)

    self.all_p_subtask_to_idx = {st: idx for idx, st in enumerate(self.all_p_subtasks)}
    assert self.all_p_subtask_size == len(self.all_p_subtasks)

    self.precomputed_indices = {}
    self.precomputed_indices['eligibility_idx_mat'] = []
    self.precomputed_indices['completion_idx_mat'] = []
    self.precomputed_indices['feature_param_mat'] = []
    self.precomputed_indices['feature_idx_mat'] = []

    assignments = list(Symbol.ground_assignments(self.symbols))
    self.precomputed_indices['nx_rows'] = len(assignments)
    self.precomputed_indices['n_columns'] = self.all_p_option_size

    for assignment in assignments:
      elig_vec = []
      for p_option in self.all_p_options:
        subbed_p_option = deepcopy(p_option)
        subbed_p_option.substitute(assignment)
        elig_vec.append(self.literal_p_option_to_idx[subbed_p_option])

      comp_vec = []
      for p_subtask in self.all_p_subtasks[:self.feature_start_idx]:
        subbed_p_subtask = deepcopy(p_subtask)
        subbed_p_subtask.substitute(assignment)
        comp_vec.append(self.literal_p_subtask_to_idx[subbed_p_subtask])

      self.precomputed_indices['eligibility_idx_mat'].append(elig_vec)
      self.precomputed_indices['completion_idx_mat'].append(comp_vec)
      
      feat_param_vec = []
      feat_idx_vec = []
      for feature in self.all_features:
        feature_param = param_to_idx[feature.compute(assignment)]
        feat_param_vec.append(feature_param)
        feature_idx = feature_labels.index(feature.name)
        feat_idx_vec.append(feature_idx)

      self.precomputed_indices['feature_param_mat'].append(feat_param_vec)
      self.precomputed_indices['feature_idx_mat'].append(feat_idx_vec)

    self.precomputed_indices['eligibility_idx_mat'] = np.asarray(self.precomputed_indices['eligibility_idx_mat'], dtype=np.int32)
    self.precomputed_indices['completion_idx_mat'] = np.asarray(self.precomputed_indices['completion_idx_mat'], dtype=np.int32)
    self.precomputed_indices['feature_param_mat'] = np.asarray(self.precomputed_indices['feature_param_mat'], dtype=np.int32)
    self.precomputed_indices['feature_idx_mat'] = np.asarray(self.precomputed_indices['feature_idx_mat'], dtype=np.int32)

    # mask(i, j) = True => allow subtask_j to be a precondition of option_i
    self.precomputed_indices['candidate_mask_mat'] = np.zeros((self.all_p_option_size, self.all_p_subtask_size), dtype=np.bool)
    for i, p_option in enumerate(self.all_p_options):
      for j, other_p_subtask in enumerate(self.all_p_subtask_to_idx):
        if isinstance(other_p_subtask, Predicate):
          is_conflicting = other_p_subtask.issubset(p_option) or p_option.issubset(other_p_subtask)
          is_candidate = other_p_subtask in self.candidates.get(p_option, [])
          is_literal = other_p_subtask in self.literal_p_subtasks
          self.precomputed_indices['candidate_mask_mat'][i, j] = (not is_conflicting) and (is_literal or is_candidate)
        if isinstance(other_p_subtask, Feature):
          share_symbol = other_p_subtask.symbol in p_option.symbols
          self.precomputed_indices['candidate_mask_mat'][i, j] = share_symbol

    # Effect indices calculation
    # option_parents(i, j) = True => option_i is a subset of (predicate)option_j
    # effect_idx_mat(i, j) = if action is literal_option_i, then put p_subtask_i in candidate effect
    # effect_mask_mat(i, j) = True => if action is literal_option_i, then allow p_subtask_i to be a candidate effect
    self.precomputed_indices['option_parents'] = np.zeros((num_literal_options, self.all_p_option_size), dtype=np.bool)
    self.precomputed_indices['effect_idx_mat'] = np.zeros((num_literal_options, self.all_p_subtask_size), dtype=np.int32)
    self.precomputed_indices['effect_mask_mat'] = np.zeros((num_literal_options, self.all_p_subtask_size), dtype=np.bool)
    for i, literal_option in enumerate(self.literal_p_options):
      for j, other_p_option in enumerate(self.all_p_options):
        self.precomputed_indices['option_parents'][i, j] = literal_option.issubset(other_p_option)
        if literal_option.issubset(other_p_option):
          assignment = other_p_option.get_reducing_assignment(literal_option)
          need_symbols = set(assignment.keys())
          for h, p_subtask in enumerate(self.all_p_subtasks):
            if not isinstance(p_subtask, Predicate):
              continue
            is_effect_candidate = set(symb.name for symb in p_subtask.symbols).issubset(need_symbols)
            self.precomputed_indices['effect_mask_mat'][i, h] = is_effect_candidate
            if is_effect_candidate:
              subbed_p_subtask = deepcopy(p_subtask)
              subbed_p_subtask.substitute(assignment)
              self.precomputed_indices['effect_idx_mat'][i, h] = self.literal_p_subtask_to_idx[subbed_p_subtask]

    # Subtask intersection matrix (used for effect calculation)
    # subtask_intersection(i, j) = True if subtask_i intersects subtask_j
    # used for if effect subtask_i, then effects subtask j if they intersect
    self.precomputed_indices['subtask_intersection'] = np.zeros((self.all_p_subtask_size, self.all_p_subtask_size), dtype=np.bool)
    for i, st_a in enumerate(self.all_p_subtasks):
      for j, st_b in enumerate(self.all_p_subtasks):
        if isinstance(st_a, Predicate) and isinstance(st_b, Predicate):
          self.precomputed_indices['subtask_intersection'][i, j] = st_a.issubset(st_b) or st_b.issubset(st_a)

    # Subtask completion Priority calculation (used for layer-wise-PILP)
    self.precomputed_indices['completion_priority'] = []
    for p_subtask in self.all_p_subtasks:
      if p_subtask in self.literal_p_subtasks:
        priority = 0
      elif isinstance(p_subtask, Feature):
        priority = 2
      else:
        if len(p_subtask.symbols) < p_subtask.nparams - 1:
          # partially parameterized subtask
          priority = 1
        else:
          # fully parameterized subtask
          priority = 2
      self.precomputed_indices['completion_priority'].append(priority)
    self.precomputed_indices['completion_priority'] = np.asarray(self.precomputed_indices['completion_priority'], dtype=np.int32)

    return self.precomputed_indices

  def _check_precomputed_indices(self, parameters, feature_mat=None):
    print('*'*10 + 'Predicate option masks:' + '*'*10)
    for i in range(self.all_p_option_size):
      p_option = self.all_p_options[i]
      candidates = [st.pretty() for j, st in enumerate(self.all_p_subtasks) if self.precomputed_indices['candidate_mask_mat'][i, j]]
      print(f'{p_option.pretty()}: {candidates}')

    print('*'*10 + 'Predicate table construction:' + '*'*10)
    for i, assignment in enumerate(Symbol.ground_assignments(self.symbols)):
      print('-'*20)
      print('-'*10 + 'eligibility' + '-'*10)
      print(' '.join(f'{symb}->{val}' for symb, val in assignment.items()))
      for j, literal_idx in enumerate(self.precomputed_indices['eligibility_idx_mat'][i]):
        print(f'{self.all_p_options[j].pretty()}->{self.literal_p_options[literal_idx].pretty()}')
      print('-'*10 + 'completion' + '-'*10)
      print(' '.join(f'{symb}->{val}' for symb, val in assignment.items()))
      for j, literal_idx in enumerate(self.precomputed_indices['completion_idx_mat'][i]):
        print(f'{self.all_p_subtasks[j].pretty()}->{self.literal_p_subtasks[literal_idx].pretty()}')
      for j, (feat_param, feat_idx) in enumerate(zip(self.precomputed_indices['feature_param_mat'][i], self.precomputed_indices['feature_idx_mat'][i])):
        if feature_mat is not None:
          print(f'{self.all_features[j].pretty()}->f{feat_idx}({parameters[feat_param]})={feature_mat[feat_param,feat_idx]}')
        else:
          print(f'{self.all_features[j].pretty()}->f{feat_idx}({parameters[feat_param]})')

    print('*'*10 + 'Effect table construction:' + '*'*10)
    for i, literal_option in enumerate(self.literal_p_options):
      print('-'*20)
      print(literal_option.pretty(), [self.all_p_options[j].pretty() for j in np.flatnonzero(self.precomputed_indices['option_parents'][i])])
      print('-'*10 + 'masked effected subtasks' + '-'*10)
      print([self.all_p_subtasks[j].pretty() for j in np.flatnonzero(self.precomputed_indices['effect_mask_mat'][i])])
      print('-'*10 + 'effect subtask indices' + '-'*10)
      for j, literal_idx in enumerate(self.precomputed_indices['effect_idx_mat'][i]):
        if self.precomputed_indices['effect_mask_mat'][i, j]:
          print(f'{self.all_p_subtasks[j].pretty()}->{self.literal_p_subtasks[literal_idx].pretty()}')

    print('*'*10 + 'Subtask completion priorities:' + '*'*10)
    for i, priority in enumerate(self.precomputed_indices['completion_priority']):
      print(self.all_p_subtasks[i].pretty(), priority)

def _test(options, subtasks, nfeatures, params, param_to_idx=None, feat_names=None, feat_mat=None):
  ONLY_PREDICATE_OPTIONS = True
  ONLY_PREDICATE_SUBTASKS = True
  from pprint import pprint
  pcand_graph = PCandidateGraph(
      options, subtasks,
      infer_literal_subtasks=True,
      infer_partial_subtasks=False,
      infer_predicate_subtasks=False,
      infer_literal_options=True,
      infer_predicate_options=False,
  )
  print('*'*10 + 'Target predicate options' + '*'*10)
  pprint(pcand_graph.target_p_options)
  print('*'*10 + 'Target predicate subtasks' + '*'*10)
  pprint(pcand_graph.target_p_subtasks)
  print('*'*10 + 'Predicate option-subtask candidates' + '*'*10)
  pprint(pcand_graph.candidates)

  pcand_graph.precompute_pilp_indices(num_parameters=len(param_to_idx), feature_dim=nfeatures, feature_labels=feat_names, param_to_idx=param_to_idx)
  #pprint(pcand_graph.all_p_subtask_to_idx)
  pcand_graph._check_precomputed_indices(params, feature_mat=feat_mat)

def _test_lub():
  A = Symbol(set((1,2,3)), 'A')
  B = Symbol(set((1,2,3)), 'B')
  #pick = Predicate(['pick', A])
  pick = Predicate(['pick', A, B])
  place = Predicate(['place', A, B])
  print(place.predicate_lub(pick))

if __name__ == '__main__':
  #_test_lub()
  env_name = 'cooking'
  if env_name == 'toythor':
    from psgi.envs.toythor.constants import COOKING_TASKS, COOKING_TRAINING_SUBTASKS, COOKING_TEST_SUBTASKS, COOKING_PARAMETER_EMBEDDINGS
    options = COOKING_TEST_SUBTASKS
    subtasks = options
    nfeatures = 3
    feat_mat = None
    params = list(sorted(set(p for st in options+subtasks for p in st)))
    param_to_idx = {p: idx for idx, p in enumerate(params)}
  elif env_name == 'simple':
    options = [('pickup', 'apple', 'table'), ('pickup', 'apple', 'pan'), ('pickup', 'pear', 'table'), ('pickup', 'pear', 'pan'), ('place', 'apple', 'table'), ('place', 'apple', 'pan'), ('place', 'pear', 'table'), ('place', 'pear', 'pan')]
    subtasks = [('holding', 'apple'), ('holding', 'pear'), ('place', 'apple', 'table'), ('place', 'apple', 'pan'), ('place', 'pear', 'table'), ('place', 'pear', 'pan')]
    nfeatures = 3
    feat_mat = None
    params = list(sorted(set(p for st in options+subtasks for p in st)))
    param_to_idx = {p: idx for idx, p in enumerate(params)}
  elif env_name == 'simple2':
    objects = ['apple', 'pear', 'table', 'chair']
    options = [('pickup', o, l) for o in objects for l in objects]
    options.extend([('place', o, l) for o in objects for l in objects])
    subtasks = [('holding', o) for o in objects]
    subtasks.extend([('place', o, l) for o in objects for l in objects])
    nfeatures = 3
    feat_mat = None
    params = list(sorted(set(p for st in options+subtasks for p in st)))
    param_to_idx = {p: idx for idx, p in enumerate(params)}

  elif env_name == 'cooking':
    from psgi import envs
    from psgi.envs.base_predicate_env import BasePredicateEnv
    config = envs.Cooking
    environment = envs.BasePredicateEnv(rank=0, graph_param='eval', config_factory=[config], keep_pristine=True, verbose_level=1)
    environment = envs.wrappers.WoBWrapper(environment)
    environment.reset_task(task_index=0)
    options = environment._environment.option_param
    subtasks = environment._environment.subtask_param
    feat_names = environment._environment.config.feature_func_names
    feat_mat = environment._environment.config.feature_mat
    _, param_emb_size = environment.observation_spec()['parameter_embeddings'].shape
    nfeatures = param_emb_size
    param_to_idx = environment._environment.parameter_name_to_index
    idx_to_param = {idx: p for p, idx in param_to_idx.items()}
    params = [idx_to_param[i] for i in range(len(param_to_idx))]

  _test(options, subtasks, nfeatures, params=params, param_to_idx=param_to_idx, feat_names=feat_names, feat_mat=feat_mat)
