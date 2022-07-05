import sys

import acme
import tree
import dm_env
import pytest
import numpy as np

import psgi.envs as envs
from psgi.envs import wrappers
from psgi.envs.base_config import OPTION_NAMES


# (env config, keep_pristine)
@pytest.fixture(params=[
    (envs.Walmart,  True), (envs.Walmart,  False),
    (envs.Dicks,    True), (envs.Dicks,    False),
    (envs.Converse, True), (envs.Converse, False),
    (envs.BestBuy,  True), (envs.BestBuy,  False),
    (envs.Apple,    True), (envs.Apple,    False),
    (envs.Amazon,   True), (envs.Amazon,   False),
    (envs.Ebay,     True), (envs.Ebay,     False),
    (envs.Samsung,  True), (envs.Samsung,  False),
    (envs.IKEA,     True), (envs.IKEA,     False),
    (envs.Target,   True), (envs.Target,   False),
])
def web_params(request):
  return request.param


class TestWebEnvironment:

  def test_timestep(self, web_params):
    config, keep_pristine = web_params
    environment = envs.BaseWoBEnv(rank=0, config_factory=[config],
                                  keep_pristine=keep_pristine)
    environment = wrappers.WoBWrapper(environment)

    # Reset task.
    environment.reset_task(task_index=1234)

    # Verify if actual TimeStep values are compliant with the spec.
    ts = environment.reset()
    assert isinstance(ts, dm_env.TimeStep)
    assert ts.first()
    print(ts)

    for _ in range(100):
      mask_elig = ts.observation['mask'] * ts.observation['eligibility']
      action = np.argmax(mask_elig)
      ts = environment.step(action)
      assert isinstance(ts, dm_env.TimeStep)

  def test_observation_spec(self, web_params):
    config, keep_pristine = web_params
    environment = envs.BaseWoBEnv(rank=0, config_factory=[config],
                                  keep_pristine=keep_pristine)
    environment = wrappers.WoBWrapper(environment)
    environment_spec = acme.specs.make_environment_spec(environment)

    observation_spec = environment_spec.observations
    action_spec = environment_spec.actions
    print("observation_spec: ", observation_spec)
    print("action_spec: ", action_spec)

    # Verify observation and action spaces.
    assert action_spec.num_values == len(OPTION_NAMES)

    assert isinstance(observation_spec, dict)
    assert 'mask' in observation_spec
    assert 'completion' in observation_spec
    assert 'eligibility' in observation_spec
    assert 'termination' in observation_spec
    assert 'action_mask' in observation_spec
    assert 'remaining_steps' in observation_spec
    assert 'step_done' in observation_spec
    assert 'action_mask' in observation_spec

    assert observation_spec['mask'].shape[0:] == (len(OPTION_NAMES), )
    assert observation_spec['eligibility'].shape[0:] == (len(OPTION_NAMES), )

  def test_observation(self, web_params):
    config, keep_pristine = web_params
    environment = envs.BaseWoBEnv(rank=0, config_factory=[config],
                                  keep_pristine=keep_pristine)
    environment.reset_task(task_index=1234)

    ob = environment.reset()
    done = False

    while not done:
      action = np.argmax(ob['mask'] * ob['eligibility'])
      subtask = environment._unwrap_action(action=action)

      # Check mask, completion, eligibility.
      assert environment.mask[subtask] == True
      assert environment.completion[subtask] == False
      assert environment.eligibility[subtask] == True

      ob, reward, done, step_count = environment.step(action)

      # Check mask, completion.
      assert environment.mask[subtask] == False
      assert environment.completion[subtask] == True

  def test_mask(self, web_params):
    config, keep_pristine = web_params
    environment = envs.BaseWoBEnv(rank=0, config_factory=[config],
                                  keep_pristine=keep_pristine)
    environment.reset_task(task_index=1234)

    ob = environment.reset()
    done = False

    while not done:
      action = np.argmax(ob['mask'] * ob['eligibility'])
      subtask_id = environment._unwrap_action(action=action)

      ob, reward, done, step_count = environment.step(action)

      if subtask_id == 'Click Step1':
        for subtask in environment.subtasks:
          if subtask == 'Click Step1': break
          assert environment.mask[subtask] == False
      if subtask_id == 'Click Step2':
        for subtask in environment.subtasks:
          if subtask == 'Click Step2': break
          assert environment.mask[subtask] == False


@pytest.fixture(params=[
    envs.Walmart,
    envs.Dicks,
    envs.Converse,
    envs.BestBuy,
    envs.Apple,
    envs.Amazon,
    envs.Ebay,
    envs.Samsung,
    envs.IKEA,
    envs.Target
])
def env_config(request):
  return request.param


class TestWebCongfigurations:

  def test_pristine(self, env_config):
    # Create environment (pristine ver.).
    web_config = env_config(seed=0, keep_pristine=True)
    web_config2 = env_config(seed=0, keep_pristine=True)

    assert 'Click Place Order' in web_config.subtasks
    assert 'Click Place Order' in web_config2.subtasks

    # Check if their subtasks match.
    for name in web_config.subtask_reward.keys():
      assert web_config.subtask_reward[name] == web_config2.subtask_reward[name]

    assert web_config.subtasks == web_config2.subtasks
    assert web_config.terminal_subtasks == web_config2.terminal_subtasks

    assert all(web_config._pool_to_index == web_config2._pool_to_index)
    assert all(web_config._index_to_pool == web_config2._index_to_pool)

  def test_perturbation(self, env_config):
    # Create environment configs with random perturbation (w/ same seed).
    web_config = env_config(seed=1234, keep_pristine=False)
    web_config2 = env_config(seed=1234, keep_pristine=False)

    assert 'Click Place Order' in web_config.subtasks
    assert 'Click Place Order' in web_config2.subtasks

    # Check if their subtasks match.
    for name in web_config.subtask_reward.keys():
      assert web_config.subtask_reward[name] == web_config2.subtask_reward[name]

    assert web_config.subtasks == web_config2.subtasks
    assert web_config.terminal_subtasks == web_config2.terminal_subtasks

    assert all(web_config._pool_to_index == web_config2._pool_to_index)
    assert all(web_config._index_to_pool == web_config2._index_to_pool)


@pytest.mark.parametrize("env_id, graph_param, action_dim, max_steps", [
    ('playground', 'D1_train', 16, 2),
    ('mining', 'eval', 26, 2),
])
def test_playground_mining(
    env_id: type,
    graph_param: str,
    action_dim: int,
    max_steps: int):
  # Create environment.
  raw_env = envs.MazeOptionEnv(
      game_name=env_id,
      graph_param=graph_param,
      gamma=0.99
  )

  environment = wrappers.MazeWrapper(raw_env)
  environment = wrappers.FewshotWrapper(environment, max_steps=max_steps)
  environment_spec = acme.specs.make_environment_spec(environment)

  observation_spec = environment_spec.observations
  action_spec = environment_spec.actions
  print("observation_spec: ", observation_spec)
  print("action_spec: ", action_spec)

  # Verify observation and action spaces.
  assert action_spec.num_values == action_dim

  assert isinstance(observation_spec, dict)
  assert 'mask' in observation_spec
  assert observation_spec['mask'].shape == (action_dim, )

  # Verify if actual TimeStep values are compliant with the spec.
  ts = environment.reset()
  assert isinstance(ts, dm_env.TimeStep)
  assert ts.first()
  print(ts)

  # TODO: WobConfig and MazeEnv are yet incompatible in terms of API
  # e.g. reset_task, config, etc. Make it unified.


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
