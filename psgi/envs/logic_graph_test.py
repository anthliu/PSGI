import pytest
import sys

from psgi.envs import logic_graph


class TestLogicOp:

  def test_primitive(self):
    A = logic_graph.SubtaskVertex("A").as_op()
    B = logic_graph.SubtaskVertex("B").as_op()
    C = logic_graph.SubtaskVertex("C").as_op()
    print("", A, B)

    A_and_B = A & B
    A_or_B = A | B
    not_A = ~A
    assert str(A_and_B) == "AND(A, B)"
    assert str(A_or_B) == "OR(A, B)"
    assert str(not_A) == "NOT(A)"

    # TODO: concatenation? or should we have n-ary operator?
    #assert str(A & B & C) == "AND(A, B, C)"


class TestLogicGraph:

  def test_trivial_graph(self):
    g = logic_graph.SubtaskLogicGraph(name="G")
    assert g.name == "G"

    # Add node (auto create)
    node_a = g.add_node("A")
    assert isinstance(node_a, logic_graph.Node)

    # Add node (externally created)
    node_b = logic_graph.SubtaskVertex("B")
    assert g.add_node(node_b) is node_b

    assert list(g.nodes) == [node_a, node_b]

    with pytest.raises(ValueError):
      g.add_node("B")  # duplicate

    with pytest.raises(KeyError):
      g["404"]

    # Create a node with dependency edge.
    g["C"] = (~g["A"]) & g["B"]
    print(g["C"].precondition)

    elig = g.compute_eligibility(completion={"A": False, "B": True})
    assert elig['A'] == True     # leaf node has no deps (tautology)
    assert elig['B'] == True
    assert elig['C'] == True

    elig = g.compute_eligibility(completion={"A": True, "B": True})
    assert elig['A'] == True     # leaf node has no deps (tautology)
    assert elig['B'] == True
    assert elig['C'] == False

  def test_flatten_associative_and_or(self):
    g = logic_graph.SubtaskLogicGraph(name="G")
    node_a = g.add_node("A")
    node_b = g.add_node("B")
    node_c = g.add_node("C")

    g["and2"] = g["A"] & g["B"]
    print(g["and2"].precondition)
    assert len(g["and2"].precondition._children) == 2

    g["and3"] = g["A"] & g["B"] & g["C"]
    print(g["and3"].precondition)
    assert len(g["and3"].precondition._children) == 3
    assert str(g["and3"].precondition) == "AND(A, B, C)"

    g["or3"] = g["A"] | g["B"] | g["C"]
    print(g["or3"].precondition)
    assert len(g["or3"].precondition._children) == 3
    assert str(g["or3"].precondition) == "OR(A, B, C)"

  def test_multiple_node_connection(self):
    g = logic_graph.SubtaskLogicGraph(name="G")
    g.add_node("A")

    # One to many
    g.connect_nodes(sources="A", sinks=["B", "C", "D"])
    assert len(g["B"].precondition._children) == 1
    assert len(g["C"].precondition._children) == 1
    assert len(g["D"].precondition._children) == 1
    assert str(g["B"].precondition) == "A"
    assert str(g["C"].precondition) == "A"
    assert str(g["D"].precondition) == "A"

    # Many to one
    g.connect_nodes(sources=["B", "C", "D"], sinks="E")
    assert len(g["E"].precondition._children) == 3
    assert str(g["E"].precondition) == "AND(B, C, D)"

    # Many to many
    g.connect_nodes(sources=["B", "C", "D"], sinks=["F", "G"])
    assert len(g["F"].precondition._children) == 3
    assert str(g["F"].precondition) == "AND(B, C, D)"
    assert len(g["G"].precondition._children) == 3
    assert str(g["G"].precondition) == "AND(B, C, D)"

  def test_cyclic_graph_error(self):
    pass

  def test_complex_graph(self):
    pass

  def test_hierarchical_graph(self):
    pass


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
