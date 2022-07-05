"""
Testing all agents on several environments.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # use cpu

import sys
import pytest

from acme import specs

import mockito
from mockito import spy2

from psgi import envs
from psgi import agents
from psgi import environment_loop
from psgi.envs import wrappers
from psgi.utils import env_utils, snt_utils


def make_envs(env_id: str, graph_param: str, seed: int, num_envs: int):
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

  # Create test environment.
  test_envs = env_utils.create_environment(
      env_id=env_id,
      graph_param=graph_param,
      seed=seed,
      batch_size=num_envs,
      num_adapt_steps=100,
      add_fewshot_wrapper=True,
      use_multi_processing=False,
  )
  return adapt_envs, test_envs


@pytest.mark.parametrize("env_id, graph_param, seed, num_envs", [
    ('playground', 'D1_eval', 1, 4),
    ('mining', 'eval', 1, 4),
    ('walmart_v2', 'eval', 1, 4)
])
def test_msgi(env_id, graph_param, seed, num_envs):
  adapt_envs, test_envs = make_envs(env_id, graph_param, seed, num_envs)
  environment_spec = specs.make_environment_spec(adapt_envs)

  # Create meta agent.
  meta_agent = agents.MSGI(
      environment_spec=environment_spec,
      num_adapt_steps=20,
      num_trial_splits=5,
      environment_id=env_id,
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
      num_adapt_steps=20,   # should be greater than TimeLimit
      num_test_episodes=1,
      num_trial_splits=5
  )


@pytest.mark.parametrize("env_id, graph_param, seed, num_envs", [
    ('playground', 'D1_train', 1, 4),
    ('mining', 'train', 1, 4),
    ('walmart_v2', 'train', 1, 4)
])
def test_mtsgi(env_id, graph_param, seed, num_envs):
  adapt_envs, test_envs = make_envs(env_id, graph_param, seed, num_envs)
  environment_spec = specs.make_environment_spec(adapt_envs)

  # Create meta agent.
  meta_agent = agents.MTSGI(
    environment_spec=environment_spec,
    num_adapt_steps=100,
    num_trial_splits=1,
    branch_neccessary_first=True,
    exploration='random',
  )

  # Run meta loop.
  meta_loop = environment_loop.EnvironmentMetaLoop(
      adapt_environment=adapt_envs,
      test_environment=test_envs,
      meta_agent=meta_agent,
      label='meta_train'  # XXX meta train
  )

  meta_loop.run(
      num_trials=3,
      num_adapt_steps=100,   # should be greater than TimeLimit
      num_test_episodes=1,
      num_trial_splits=1
  )

@pytest.mark.parametrize("env_id, graph_param, seed, num_envs", [
    ('playground', 'D1_train', 1, 4),
    ('mining', 'train', 1, 4),
    ('walmart_v2', 'train', 1, 4),
])
def test_rlrl(env_id, graph_param, seed, num_envs):
  adapt_envs, test_envs = make_envs(env_id, graph_param, seed, num_envs)
  environment_spec = specs.make_environment_spec(adapt_envs)

  if env_id in {'playground', 'mining'}:
    # spatial observation.
    network = snt_utils.CombinedNN(environment_spec.actions)
  else:
    network = snt_utils.RecurrentNN(environment_spec.actions)

  # Create meta agent.
  meta_agent = agents.RLRL(
      environment_spec=environment_spec,
      network=network,
      n_step_horizon=10,
      minibatch_size=10
  )

  # Run meta loop.
  meta_loop = environment_loop.EnvironmentMetaLoop(
      adapt_environment=adapt_envs,
      test_environment=test_envs,
      meta_agent=meta_agent,
      label='meta_train'  # XXX meta train
  )

  meta_loop.run(
      num_trials=3,
      num_adapt_steps=100,   # should be greater than TimeLimit
      num_test_episodes=1,
      num_trial_splits=1
  )



if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
