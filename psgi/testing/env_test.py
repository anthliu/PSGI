import sys
import pytest
import numpy as np

from acme import specs

import mockito
from mockito import spy2

import psgi
from acme.utils import paths
from psgi.utils import env_utils
from psgi.utils import log_utils

from psgi import envs
from psgi import environment_loop
from psgi.envs import wrappers
from psgi.utils.graph_utils import transform_obs
from psgi.graph.ilp import ILP
from psgi.agents.grprop import GRPropActor
from psgi.agents.eval_actor import EvalWrapper

## CONSTANTS for testing ##
ENV_NAME = 'mining'
SEED = 1
NUM_ADAPT_STEPS = 100
NUM_TASKS = 20
NUM_TEST_EPI = 4
GRAPH_PARAM = "eval"
ROOT='./psgi/testing/debug_data'
## CONSTANTS for testing ##

@pytest.mark.parametrize("spy2, env_class", [
    (spy2, envs.MazeOptionEnv),
])
def test_env(spy2, env_class: type):
  # Prepare test phase environment.
  envs = env_utils.create_environment(
      env_id=ENV_NAME,
      batch_size=1,
      graph_param=GRAPH_PARAM,
      use_multi_processing=False,
      seed=SEED,
      gamma=.99
  )
  #environment_spec = specs.make_environment_spec(environment)

  #
  filename = f'{ROOT}/traj_{ENV_NAME}_eval_{SEED}.npy'
  obss, feats, actions, rewards, time_costs, actives = np.load(filename, allow_pickle=True)
  for task_idx, (gt_obs, gt_feat, gt_action, gt_reward, gt_time_cost, gt_active) in \
    enumerate(zip(obss, feats, actions, rewards, time_costs, actives)):
    #
    map_filename = f'{ROOT}/{ENV_NAME}_eval_{SEED}_map{task_idx}.npy'
    envs.reset_task(task_index=task_idx)
    envs._envs[0].load_debugging_map(map_filename)
    envs._envs[0]._environment.map._calculate_distance(blocked_list=[])  # passing empty list, since the map from "map_filename" has no block
    action_dim = envs.pool_to_index[0].shape[0]
    #
    timestep = envs.reset()
    timesteps = [timestep]
    step_count, episode_count = 0, 0
    while not episode_count >= NUM_TEST_EPI:
      action = gt_action[step_count].squeeze(-1)
      timestep = envs.step(action)
      timesteps.append(timestep)
      # Book-keeping
      if timestep.last():
        episode_count += timestep.last()  # increment if last
      else:
        step_count += 1
    gt_feat = [feat[0:, :action_dim*3] for feat in gt_feat]
    gt_reward = [rew[0] for rew in gt_reward]
    mb_obs, mb_feat, mb_reward, mb_time_cost = _parse_timestep(timesteps)

    np.testing.assert_allclose(mb_feat, gt_feat[:-1])
    np.testing.assert_allclose(mb_reward, gt_reward)
    np.testing.assert_allclose(mb_reward, gt_reward)
    np.testing.assert_allclose(mb_time_cost, gt_time_cost)

def _parse_timestep(timesteps):
  mb_obs, mb_feat, mb_reward, mb_active, mb_time_cost = [], [], [], [], []
  prev_rem_step = -1
  for timestep in timesteps:
    observation = timestep.observation
    if not timestep.last():
      obs = observation['observation']
      feat = np.concatenate((observation['mask'], observation['completion'], observation['eligibility']), axis=1)
      mb_obs.append(obs)
      mb_feat.append(feat)

    if not timestep.first():
      assert prev_rem_step >= 0
      reward = timestep.reward
      time_cost = prev_rem_step - int(observation['remaining_steps'])
      mb_reward.append(reward)
      mb_time_cost.append([time_cost])
    prev_rem_step = int(observation['remaining_steps'])
  return mb_obs, mb_feat, mb_reward, mb_time_cost

if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
