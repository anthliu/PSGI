from typing import Optional
import typing
import pytest
import collections
import sys

from acme import specs
import tensorflow as tf

from psgi import agents
from psgi import envs
from psgi import environment_loop
from psgi.system import multiprocessing
from psgi.utils import env_utils
from psgi.utils import snt_utils


try:
  multiprocessing.enable_interactive_mode()
except ValueError:
  pass  # already initialized


class EnvParameter(typing.NamedTuple):
  env_id: str
  graph_param: Optional[str]
  num_envs: int
  num_trials: int
  num_adapt_steps: int
  num_test_episodes: int
  num_trial_splits: int


TEST_PARAMS = [
    # TODO: num_envs = 1 (fix the single-actor assumption)
    EnvParameter('playground', 'D1_train', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    # TODO: num_envs = 1 (fix the single-actor assumption)
    EnvParameter('mining', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('walmart', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('dicks', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('converse', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('bestbuy', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('apple', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('amazon', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('ebay', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('samsung', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('ikea', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
    EnvParameter('target', 'eval', num_envs=1,
                 num_trials=2, num_adapt_steps=100,
                 num_test_episodes=1, num_trial_splits=5),
]


class TestHRL:

  @pytest.mark.parametrize("param", [
    pytest.param(v, id=v.env_id) for v in TEST_PARAMS
  ])
  def test_hrl(self, param):
    # Create an environment.
    adapt_env = env_utils.create_environment(
        env_id=param.env_id,
        batch_size=param.num_envs,
        graph_param=param.graph_param,
        num_adapt_steps=param.num_adapt_steps,
        add_fewshot_wrapper=True,
        seed=1
    )
    env_spec = specs.make_environment_spec(adapt_env)

    test_env = env_utils.create_environment(
        env_id=param.env_id,
        batch_size=param.num_envs,
        graph_param=param.graph_param,
        num_adapt_steps=param.num_adapt_steps,
        add_fewshot_wrapper=True,
        seed=1
    )

    # Build network.
    if param.env_id in {'playground', 'mining'}:
      # spatial observation.
      network = snt_utils.CombinedNN(env_spec.actions)
    else:
      network = snt_utils.RecurrentNN(env_spec.actions)

    # Create HRL agent.
    batch_size = param.num_adapt_steps // param.num_trial_splits
    meta_agent = agents.HRL(
        environment_spec=env_spec,
        network=network,
        n_step_horizon=batch_size
    )

    # Run meta loop.
    meta_loop = environment_loop.EnvironmentMetaLoop(
      adapt_environment=adapt_env,
      test_environment=test_env,
      meta_agent=meta_agent,
      label='meta_eval')

    # Simply checks if the loop completes without any error raised.
    meta_loop.run(
      num_trials=param.num_trials,
      num_adapt_steps=param.num_adapt_steps,
      num_test_episodes=param.num_test_episodes,
      num_trial_splits=param.num_trial_splits,
    )


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
