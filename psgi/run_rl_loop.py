# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs meta agent on toywob environment locally."""

import functools
import datetime
import time

from absl import app
from absl import flags
from acme import specs
from acme.utils import counting
from acme.utils import loggers
import numpy as np

import psgi
from psgi.system import multiprocessing
from psgi.utils import log_utils
from psgi.utils import env_utils
from psgi import agents

# Environment loop flags.
flags.DEFINE_string('work_dir', 'logs/default', 'Logging directory.')
flags.DEFINE_string('feature_mode', 'cluster-s', 'feature mode.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('exp_id', 1, 'Experiment ID.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_string('env_id', 'mining', 'Environment name/ID.')
flags.DEFINE_integer('num_envs', 1, 'Batch size of environments (parallel or serial_wrapper).')
flags.DEFINE_integer('num_tasks', 1, 'Number tasks to test over.')
flags.DEFINE_integer('num_test_episodes', 4, 'Number of test episodes to average performance over.')
flags.DEFINE_enum('label', 'eval', ['train', 'eval'],
                    'Environment label/mode (train or eval).')
flags.DEFINE_string('graph_param', 'D1_train', 'Difficulty of subtask graph.')

# Agent flags.
# TODO: add more agent types.
flags.DEFINE_enum('algorithm', 'random', ['random', 'greedy', 'grprop', 'fixed'],
                    'Name of the algorithm to run (e.g. random, greedy).')
### GRProp
# see also: default parameters in grprop.GRProp()
flags.DEFINE_float('grprop_temp', 200., 'GRProp parameter.')
flags.DEFINE_float('w_a', 3., 'GRProp parameter.')
flags.DEFINE_float('beta_a', 8., 'GRProp parameter.')
flags.DEFINE_float('ep_or', 0.8, 'GRProp parameter.')
flags.DEFINE_float('temp_or', 2., 'GRProp parameter.')

# Logger flags.
flags.DEFINE_boolean('save_logs', True, 'Whether to save the loggings.')
flags.DEFINE_boolean('verbose', False, 'Whether to print out detailed trajectories.')

# Debugging flags.

FLAGS = flags.FLAGS

AGENT_CLASS_DICT = {
  'fixed': (lambda environment_spec: agents.FixedActor(
    environment_spec, option_sequence=np.load(
      '../msgi/grid-world/debug_data/action_list.npy', allow_pickle=True)
  )),
  'random': agents.RandomActor,
  'greedy': agents.GreedyActor,
  'grprop': agents.GRPropActor,
}
EVAL_ONLY = ['random', 'greedy', 'grprop']

def main(_):
  # Assertions
  if FLAGS.algorithm in EVAL_ONLY:
    assert FLAGS.label == 'eval', \
      '%s only supports meta-eval, not meta-train'%(FLAGS.algorithm)

  if FLAGS.env_id == 'webnav':
    assert FLAGS.graph_param == 'eval', 'RL Loop only runs on eval tasks.'

  time_str = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
  log_dir = f"{FLAGS.work_dir}/{FLAGS.env_id}_{FLAGS.graph_param}/{FLAGS.algorithm}/run{FLAGS.exp_id}_{FLAGS.seed}_{time_str}"

  # Create loggers for meta RL loop.
  logger, agent_logger = log_utils.create_loggers(
      logdir=log_dir,
      label='eval',
      save_data=FLAGS.save_logs,
  )

  # Create an environment and grab the spec.
  environment = env_utils.create_environment(
      env_id=FLAGS.env_id,
      batch_size=FLAGS.num_envs,
      feature_mode=FLAGS.feature_mode,
      graph_param=FLAGS.graph_param,
      use_multi_processing=False,
      seed=FLAGS.seed,
      gamma=FLAGS.gamma,
  )
  environment_spec = specs.make_environment_spec(environment)

  # Create agent/actor.
  actor_class = AGENT_CLASS_DICT[FLAGS.algorithm]
  actor_args = dict(environment_spec=environment_spec)
  if FLAGS.algorithm == 'grprop':
    actor_args.update(
        temp=FLAGS.grprop_temp,
        w_a=FLAGS.w_a,
        beta_a=FLAGS.beta_a,
        ep_or=FLAGS.ep_or,
        temp_or=FLAGS.temp_or,
    )
  actor = actor_class(**actor_args)
  actor = agents.EvalWrapper(actor)

  # Run multi-task RL loop
  loop = psgi.EnvironmentLoop(
      environment, actor,
      label='test'
  )
  #
  _start_time = time.time()
  for task_idx in range(FLAGS.num_tasks):
    # Sample a new task
    environment.reset_task(task_index=task_idx)
    actor.observe_task(environment.task_embedding)

    # Run loop
    _, counts = loop.run(
        num_episodes=FLAGS.num_test_episodes,
        reset=True,
        update=False,
    )

    # Time
    elapsed = (time.time() - _start_time) / 60.
    remain = elapsed / (task_idx + 1) * (FLAGS.num_tasks - task_idx - 1)
    print(f'[{task_idx+1}/{FLAGS.num_tasks}] remain={remain:.1f}min  | elapsed={elapsed:.1f}min')

    # Logging
    counts['tasks'] = task_idx
    counts['test/cumulative_reward'] /= counts['test/episodes']
    counts['test/success'] /= counts['test/episodes']
    logger.write(counts)

if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
