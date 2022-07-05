import sys
import numpy as np
import os.path

from acme import specs

import psgi
from acme.utils import paths
from psgi.utils import env_utils
from psgi.utils import log_utils

from psgi import envs
from psgi import environment_loop
from psgi.envs import wrappers
#from psgi.utils.env_utils import add_batch_dim
from psgi.agents.msgi import msgi
from psgi.agents.psgi import mtsgi
from psgi.graph.ilp import ILP

## CONSTANTS for testing ##
ENV_NAME='mining'
TASK_IDX = 0
SEED = 1
NUM_ADAPT_STEPS = 500
GRAPH_PARAM = "eval"
OPTION_INDEX_SEQUENCE = [0, 3, 8, 1, 7]
## CONSTANTS for testing ##

def save_ilp(environment, filename):
  environment_spec = specs.make_environment_spec(environment)
  # Create actor
  meta_agent = msgi.MSGIRandom(
      environment_spec=environment_spec,
      num_adapt_steps=NUM_ADAPT_STEPS,
      visualize=False,
      directory=None,
      environment_id=ENV_NAME
    )
  agent = meta_agent.instantiate_adapt_agent() # MSGIActor() (Fast agent)

  # Set task
  environment.reset_task(task_index=TASK_IDX)
  meta_agent.reset_agent(environment=environment)
  # Run RL loop
  ### Reset
  timestep = environment.reset()
  agent.observe_first(timestep)

  ### Loop
  cumulative_reward = 0.
  for step_count in range(NUM_ADAPT_STEPS):
    action = agent.select_action(timestep.observation)
    print(action)
    timestep = environment.step(action)

    agent.observe(action, next_timestep=timestep)

    # Book-keeping.
    cumulative_reward += sum(timestep.reward)  # TODO: needs check.

  meta_agent._ilp.save(filename)

def infer_graph(environment, filename):
  environment_spec = specs.make_environment_spec(environment)
  ilp = ILP(
    environment_spec=environment_spec,
    num_adapt_steps=NUM_ADAPT_STEPS,
    visualize=False,
    directory=None,
    environment_id=ENV_NAME
  )
  ilp.reset(environment)
  ilp.load(filename)
  graph = ilp.infer_task()
  print(graph[0].ANDmat.shape)
  print(graph[0].ORmat.shape)

def eval_grprop(environment, filename):
  environment_spec = specs.make_environment_spec(environment)
  ilp = ILP(
    environment_spec=environment_spec,
    num_adapt_steps=NUM_ADAPT_STEPS,
    visualize=False,
    directory=None,
    environment_id=ENV_NAME
  )
  ilp.reset(environment)
  ilp.load(filename)
  graph = ilp.infer_task()


if __name__ == '__main__':
  filename = '../msgi/grid-world/ilp_data/ilp_data_101.npy'
  # Prepare adaptation phase environment.
  environment = env_utils.create_environment(
      env_id=ENV_NAME,
      batch_size=1,
      graph_param=GRAPH_PARAM,
      use_multi_processing=False,
      seed=SEED,
      gamma=.99
  )
  test_loop = EnvironmentLoop(
    environment, self._test_actor,
    label='test_loop')
  if not os.path.exists(filename):
    save_ilp(environment, filename)
  for task_idx in range(20):
    environment.reset_task(task_index=task_idx)
    filename = f'../msgi/grid-world/ilp_data/ilp_task{task_idx}.npy'
    #infer_graph(environment, filename)
    eval_grprop(environment, filename, test_loop)
  #eval_ilp(environment, filename)
