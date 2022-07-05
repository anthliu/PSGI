import os
import sys
import predicate_graph
from psgi.utils.predicate_utils import Predicate, Symbol

class TestLogicOp:
  def test_primitive(self):
    A = predicate_graph.SubtaskVertex("A").as_op()
    B = predicate_graph.SubtaskVertex("B").as_op()
    C = predicate_graph.SubtaskVertex("C").as_op()
    print("", A, B)

    A_and_B = A & B
    A_or_B = A | B
    not_A = ~A
    assert str(A_and_B) == "AND(A, B)"
    assert str(A_or_B) == "OR(A, B)"
    assert str(not_A) == "NOT(A)"

class TestLogicGraph:

  def test_trivial_graph(self):
    g = predicate_graph.PredicateLogicGraph('Cooking')
    X = Symbol(set(), 'x')
    Y = Symbol(set(), 'y')
    # Add all features
    PICKUPABLE_X = Predicate(['pickupable', X])
    PUTABLE_Y = Predicate(['putable', Y])
    g.add_features([PICKUPABLE_X, PUTABLE_Y])
    # Add all subtask nodes (just names)
    PUT_X_Y = Predicate(['put', X, Y])
    PICKUP_X = Predicate(['pickup', X])
    g.add_subtasks([PUT_X_Y, PICKUP_X])

    # Add each option one-by-one
    OP_PUT_X_Y = Predicate(['op_put', X, Y])
    g.add_option(
      param=OP_PUT_X_Y,
      precondition=g[PICKUP_X] & g[PUTABLE_Y],
      effect=(~g[PICKUP_X]) & g[PUT_X_Y]
    )
    OP_PICKUP_X_Y = Predicate(['op_pickup', X, Y])
    g.add_option(
      param=OP_PICKUP_X_Y,
      precondition=g[PUT_X_Y] & g[PUTABLE_Y],
      effect=(~g[PUT_X_Y]) & g[PICKUP_X]
    )
    
    g.print_graph()
    ##########################

    _FOOD_PARAMS = ['apple', 'cabbage', 'beef']
    _BOARD_PARAMS = ['fridge', 'table', 'stove']
    _COOKWARE_PARAMS = ['pan']
    PARAMS = _FOOD_PARAMS + _BOARD_PARAMS + _COOKWARE_PARAMS
    
    fg = g.unroll_graph(param_pool=PARAMS)
    #
    print('======')
    completion = dict()

    # Init with all False
    for x in PARAMS:
      completion[f'pickup,{x}'] = False
      completion[f'pickupable,{x}'] = False
      completion[f'putable,{x}'] = False
      for y in PARAMS:
        completion[f'put,{x},{y}'] = False
    
    for x in _FOOD_PARAMS + _COOKWARE_PARAMS:
      completion[f'put,{x},table'] = True
      completion[f'pickupable,{x}'] = True
    for x in _COOKWARE_PARAMS + _BOARD_PARAMS:
      completion[f'putable,{x}'] = True

    print('======')
    print('Completed subtasks:')
    print([subtask for subtask, comp in completion.items() if comp])
    eligibility = fg.compute_eligibility(completion=completion)
    
    print('======')
    print('Eligibile subtasks:')
    print([subtask for subtask, elig in eligibility.items() if elig])
    #
    option = 'op_put,apple,stove'
    print(f'option = {option}')
    assert option in fg, f"Error. Option {option} does not exist"
    effect = fg.compute_effect(option=option)
    completion.update(effect)
    print('Effect')
    print(effect)

    option = 'op_pickup,apple,stove'
    print(f'option = {option}')
    assert option in fg, f"Error. Option {option} does not exist"
    effect = fg.compute_effect(option=option)
    completion.update(effect)
    print('Effect')
    print(effect)

if __name__ == '__main__':
  test = TestLogicGraph()
  test.test_trivial_graph()
  #sys.exit(pytest.main(["-s", "-v"] + sys.argv))
