import sys
import pytest

from acme import specs
from acme.utils import tree_utils

import mockito
from mockito import spy2

from psgi import envs
from psgi import agents
from psgi import environment_loop
from psgi.utils import env_utils
from psgi.envs import wrappers
from psgi.testing import fakes


@pytest.mark.parametrize("env_id, graph_param, seed, num_envs", [
    ('playground', 'D1_eval', 1, 4),
    ('mining', 'eval', 1, 4),
    ('walmart_v2', 'eval', 1, 4)
])
def test_environment_loop(env_id, graph_param, seed, num_envs):
  """Tests Meta RL Loop."""
  # Create a single adaptation environment.
  adapt_envs = env_utils.create_environment(
      env_id=env_id,
      graph_param=graph_param,
      seed=seed,
      batch_size=num_envs,
      num_adapt_steps=100,
      add_fewshot_wrapper=True,
      use_multi_processing=False,
  )
  spy2(adapt_envs.step)   # To keep track of # of invocations.
  environment_spec = specs.make_environment_spec(adapt_envs)

  # Create test environment.
  test_envs = env_utils.create_environment(
      env_id=env_id,
      graph_param=graph_param,
      seed=seed,
      batch_size=num_envs,
      num_adapt_steps=100,
      add_fewshot_wrapper=True,
      use_multi_processing=False
  )

  # Create meta agent (default: MSGI-Random).
  meta_agent = agents.MSGI(
      environment_spec=environment_spec,
      num_adapt_steps=100,
      num_trial_splits=5,
      branch_neccessary_first=True,
      exploration='random',
      temp=200,
      w_a=3.0,
      beta_a=8.0,
      ep_or=0.8,
      temp_or=2.0
  )

  # Run meta loop.
  meta_loop = environment_loop.EnvironmentMetaLoop(
      adapt_environment=adapt_envs,
      test_environment=test_envs,
      meta_agent=meta_agent,
      label='meta_eval'
  )

  meta_loop.run(
      num_trials=3,
      num_adapt_steps=100,   # should be greater than TimeLimit
      num_test_episodes=1,
      num_trial_splits=5
  )

  expected_steps = 3 * 100
  mockito.verify(adapt_envs, times=expected_steps).step(...)


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
