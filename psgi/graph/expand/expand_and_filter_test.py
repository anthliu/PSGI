import sys

import numpy as np
import pytest


try:
  from psgi.graph.expand import expand_and_filter
except ImportError:
  expand_and_filter = None


X = -1  # don't care

class TestExpandAndFilter:

  @pytest.mark.skipif(expand_and_filter is None,
                      reason="pybind extension has not been built")
  def test_simple(self):

    print("")
    V = 3

    prior_comp = np.zeros([3, V], dtype=np.int8)
    prior_elig = np.zeros([3, V], dtype=np.int8)
    prior_comp[0, :], prior_elig[0, :] = [0, 0, X], [1, 0, X]
    prior_comp[1, :], prior_elig[1, :] = [1, 0, X], [1, 1, X]
    prior_comp[2, :], prior_elig[2, :] = [1, 1, X], [1, 1, X]

    curr_comp = np.zeros([4, V], dtype=np.int8)
    curr_elig = np.zeros([4, V], dtype=np.int8)
    curr_comp[0, :], curr_elig[0, :] = [0, 0, 0], [1, 0, 1]
    curr_comp[1, :], curr_elig[1, :] = [0, 0, 1], [1, 0, 1]
    curr_comp[2, :], curr_elig[2, :] = [1, 0, 0], [1, 1, 1]
    curr_comp[3, :], curr_elig[3, :] = [1, 0, 1], [1, 0, 1]

    expanded_comp, expanded_elig = expand_and_filter.expand_and_filter(
      prior_comp, prior_elig, curr_comp, curr_elig)

    print(expanded_comp)
    print(expanded_elig)

    assert expanded_comp.shape == (5, V)
    assert expanded_elig.shape == (5, V)

    np.testing.assert_array_equal(expanded_comp, [
      [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]
    ])
    np.testing.assert_array_equal(expanded_elig, [
      [1, 0, -1], [1, 0, -1], [1, 1, -1], [1, 1, -1], [1, 1, -1]
    ])


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
