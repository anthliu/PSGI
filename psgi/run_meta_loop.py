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

import datetime
import functools
import glob
import os
import numpy as np
np.set_printoptions(precision=3, suppress=True)

from absl import app
from absl import flags
from acme import specs
from acme.utils import paths

from psgi.system import multiprocessing

import psgi
from psgi import agents
from psgi.utils import env_utils
from psgi.utils import log_utils
from psgi.utils import snt_utils

import tensorflow as tf

# Environment loop flags.
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('load_seed', 1, 'Random seed.')
flags.DEFINE_integer('exp_id', 1, 'Experiment ID.')
flags.DEFINE_integer('load_exp_id', -1, 'ID of experiment to load from.')

flags.DEFINE_integer('num_envs', 1, 'Batch size of environments (parallel or serial_wrapper).')
flags.DEFINE_boolean('use_multi_processing', False, 'multi-processing/serial.')
flags.DEFINE_integer('num_trials', 10, 'Number updates to perform over tasks distribution.')
flags.DEFINE_integer('num_adapt_steps', 1000, 'Number of adaptation steps to task in a trial.')
flags.DEFINE_integer('num_test_episodes', 4, 'Number of test episodes to average performance over.')
flags.DEFINE_integer('num_trial_splits', 40, 'Number of iterations of adaptation-test phases in a trial.')
flags.DEFINE_enum('label', 'meta_eval', ['meta_train', 'meta_eval'],
                    'Environment label/mode (meta_train or meta_eval).')

# Environment flags.
flags.DEFINE_string('feature_mode', 'cluster-s', 'feature mode.')
flags.DEFINE_string('work_dir', 'logs/default', 'Logging directory.')
flags.DEFINE_string('env_id', 'mining', 'Environment name/ID.')
flags.DEFINE_string('graph_param', 'eval', 'Difficulty of subtask graph.')
flags.DEFINE_string('load_graph_param', 'train', 'Difficulty of subtask graph.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')

# Debugging
flags.DEFINE_integer('task_idx_debug', -1, 'Fixing task index for debugging.')
flags.DEFINE_integer('verbose_level', 0, 'verbose level.')

# Agent flags.
# TODO: add more agent types.
flags.DEFINE_enum('algorithm', 'msgi', ['msgi', 'msgi_plus', 'mtsgi', 'hrl', 'rlrl', 'psgi', 'np_psgi', 'greedy'],
                  'Name of the algorithm to run (e.g. msgi, rlrl).')
flags.DEFINE_integer('n_step_horizon', 16, 'n-step learning horizon.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('gae_lambda', 0.99, 'GAE lambda parameter.')
flags.DEFINE_float('entropy_cost', 0.01, 'Coefficient for entropy loss.')
flags.DEFINE_float('baseline_cost', 0.5, 'Coefficient for critic/baseline loss.')
flags.DEFINE_float('max_abs_reward', None, 'Reward clipping.')
flags.DEFINE_float('max_gradient_norm', 10., 'Maximum gradient norm.')

# ILP
flags.DEFINE_boolean('neccessary_first', True, 'Whether to use neccessary-first branching in CART.')

# Cycle GRProp
flags.DEFINE_float('grprop_temp', 200., 'temperature for logit.')
flags.DEFINE_float('grprop_sigma', -1.0, 'default logit for infeasible option.')

flags.DEFINE_float('w_a', 3., 'GRProp parameter.')
flags.DEFINE_float('beta_a', 8., 'GRProp parameter.')
flags.DEFINE_float('ep_or', 0.8, 'GRProp parameter.')
flags.DEFINE_float('temp_or', 2., 'GRProp parameter.')

# MSGI & MTSGI
flags.DEFINE_enum('exploration', 'ucb', ['random', 'count', 'grprop', 'ucb', 'mtucb'],
                  'Adaptation exploration strategy for MSGI.')
flags.DEFINE_float('ucb_temp', 20.0, 'UCB temperature parameter.')

# MTSGI
flags.DEFINE_enum('prior_sample_mode', 'coverage', ['uniform', 'coverage', 'debug'], 'Mode to sample prior.')
flags.DEFINE_enum('posterior_mode', 'policy', ['policy', 'ilp', 'debug'], 'Posterior mixing mode.')
flags.DEFINE_integer('num_prior_load', 5, 'Number of priors (tasks) to load.')

# Logger flags.
# TODO: If False, CSVDumper won't be used but this will break logging
# as the data  (split_avg_log) to be logged is a list.
flags.DEFINE_boolean('save_logs', True, 'Whether to save the loggings.')

# Snapshot flags.
flags.DEFINE_boolean('save_snapshots', False, 'Whether to save the model snapshots.')

# MSGI flags.
flags.DEFINE_boolean('visualize', False, 'Whether to visualize subtask graph.')
FLAGS = flags.FLAGS

AGENT_CLASS_DICT = {
  'msgi': agents.MSGI,
  'msgi_plus': agents.MSGI_plus,
  'mtsgi': agents.MTSGI,
  'rlrl': agents.RLRL,
  'hrl': agents.HRL,
  'psgi': agents.PSGI,
  'np_psgi': agents.PSGI,
  'greedy': agents.MetaGreedyActor,
}
META_EVAL_ONLY = ['msgi', 'hrl', 'msgi_plus', 'np_psgi', 'greedy']
META_TRAIN = ['mtsgi', 'rlrl', 'psgi']
USE_GRPROP = ['msgi', 'mtsgi']
USE_CYCLE_GRPROP = ['psgi', 'msgi_plus', 'np_psgi']
USE_UCB = ['msgi', 'msgi_plus', 'mtsgi', 'psgi', 'np_psgi']
USE_ILP = ['msgi', 'msgi_plus', 'mtsgi', 'psgi', 'np_psgi']
USE_A2C = ['hrl', 'rlrl']

def main(_):
  # Assertions
  argument_assertions()

  ### runs: logs/{env_name}_{task_name}/{method_hparam}/{seed}
  hparam_str, load_hparam_str = get_hparam_str(FLAGS)
  env_str = f'{FLAGS.env_id}_{FLAGS.graph_param}'
  time_str = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
  root_dir = f"{FLAGS.work_dir}/{env_str}/{FLAGS.algorithm}_{hparam_str}/run{FLAGS.exp_id}_{FLAGS.seed}_{time_str}"
  if FLAGS.visualize:
    graph_dir = f"visualize/{env_str}/{FLAGS.algorithm}_{hparam_str}/run{FLAGS.exp_id}_{FLAGS.seed}_{time_str}"
  else:
    graph_dir = None
  save_filename = f"{root_dir}/train_data.npy"

  ### Loading
  load_filenames = None
  if FLAGS.load_exp_id >= 0:
    assert len(load_hparam_str) > 0, 'Error: hparam string for loading has not been set!'
    load_prefixes, load_postfixes = [], []
    load_filenames = []

    if FLAGS.env_id == 'webnav':  # 3:7 fixed web envs.
      configs = psgi.envs.WEBNAV_TASKS[f'train_{FLAGS.seed}']
      configs = [config.environment_id for config in configs]
      for config in configs:
        load_prefixes.append(f"{FLAGS.work_dir}/{config}_train_1")
        load_postfixes.append(f"run{FLAGS.load_exp_id}_{FLAGS.load_seed}_*/train_data.npy")

    elif FLAGS.env_id in psgi.envs.web_configs_dict:  # single web env.
      configs = env_utils.sample_web_configs(
          eval_env=FLAGS.env_id,
          seed=FLAGS.seed,
          size=FLAGS.num_prior_load
      )
      for config in configs:
        load_prefixes.append(f"{FLAGS.work_dir}/{config}_train_1")  # XXX seed always fixed to 1
        load_postfixes.append(f"run{FLAGS.load_exp_id}_1_*/train_data.npy") # XXX seed always fixed to 1

    elif FLAGS.env_id in psgi.envs.web_configs_v2_dict:  # single web env. (XXX ver. 2)
      configs = env_utils.sample_web_configs(
          eval_env=FLAGS.env_id,
          seed=FLAGS.seed,
          size=FLAGS.num_prior_load
      )
      for config in configs:
        load_prefixes.append(f"{FLAGS.work_dir}/{config}_train_1")  # XXX seed always fixed to 1
        load_postfixes.append(f"run{FLAGS.load_exp_id}_1_*/train_data.npy") # XXX seed always fixed to 1

    else:  # playground / mining
      load_prefixes.append(f'{FLAGS.work_dir}/{FLAGS.env_id}_{FLAGS.load_graph_param}')
      if FLAGS.algorithm == 'rlrl':
        load_postfixes.append(f"run{FLAGS.load_exp_id}_{FLAGS.load_seed}_*/train_data.npy/*/snapshots")
      else:
        load_postfixes.append(f"run{FLAGS.load_exp_id}_{FLAGS.load_seed}_*/train_data.npy")

    # Collect all the files to load.
    for prefix, postfix in zip(load_prefixes, load_postfixes):
      load_glob_str = f"{prefix}/{FLAGS.algorithm}_{load_hparam_str}/{postfix}"
      matching_paths = glob.glob(load_glob_str)
      assert len(matching_paths) == 1, f'Error: cannot load from {len(matching_paths)} files: {load_glob_str}'
      load_filenames.append(matching_paths[0])

  # Create loggers for meta RL loop.
  environment_logger, agent_logger = log_utils.create_loggers(
      logdir=root_dir,
      label='meta',
      save_data=FLAGS.save_logs,
  )

  # TODO: support environment creation of individual web navigation tasks.
  # Prepare adaptation phase environment.
  adapt_environment = env_utils.create_environment(
      env_id=FLAGS.env_id,
      batch_size=FLAGS.num_envs,
      feature_mode=FLAGS.feature_mode,
      graph_param=FLAGS.graph_param,
      use_multi_processing=FLAGS.use_multi_processing,
      num_adapt_steps=FLAGS.num_adapt_steps,
      add_fewshot_wrapper=True,
      seed=FLAGS.seed,
      gamma=FLAGS.gamma,
      verbose_level=FLAGS.verbose_level,
  )
  environment_spec = specs.make_environment_spec(adapt_environment)

  # Prepare test phase environment.
  test_environment = env_utils.create_environment(
      env_id=FLAGS.env_id,
      batch_size=FLAGS.num_envs,
      feature_mode=FLAGS.feature_mode,
      graph_param=FLAGS.graph_param,
      use_multi_processing=FLAGS.use_multi_processing,
      num_adapt_steps=FLAGS.num_adapt_steps,
      add_fewshot_wrapper=True,
      seed=FLAGS.seed,
      gamma=FLAGS.gamma,
      verbose_level=FLAGS.verbose_level,
  )

  # Create directory for graph visualization.
  meta_agent_class = AGENT_CLASS_DICT[FLAGS.algorithm]
  agent_args = dict(
      environment_spec=environment_spec,
      mode=FLAGS.label,
      logger=agent_logger,
      verbose_level=FLAGS.verbose_level,
  )
  if FLAGS.algorithm in USE_ILP:
    agent_args.update(
        num_adapt_steps=FLAGS.num_adapt_steps,
        num_trial_splits=FLAGS.num_trial_splits,
        exploration=FLAGS.exploration,
        visualize=FLAGS.visualize,
        directory=graph_dir,
        environment_id=FLAGS.env_id,
        branch_neccessary_first=FLAGS.neccessary_first,
    )
  if FLAGS.algorithm in USE_GRPROP:
    agent_args.update(
        temp=FLAGS.grprop_temp,
        w_a=FLAGS.w_a,
        beta_a=FLAGS.beta_a,
        ep_or=FLAGS.ep_or,
        temp_or=FLAGS.temp_or,
    )
  if FLAGS.algorithm in USE_CYCLE_GRPROP:
    agent_args.update(
        grprop_temp=FLAGS.grprop_temp,
    )
  if FLAGS.algorithm in USE_UCB:  # add ucb hparam
    agent_args.update(ucb_temp=FLAGS.ucb_temp)

  if FLAGS.algorithm in USE_A2C:
    assert FLAGS.num_adapt_steps % FLAGS.n_step_horizon == 0, \
        'Number of adaptation steps must be a multiple of n-step horizon.'

    if FLAGS.env_id in {'playground', 'mining'}: # spatial observation.
      network = snt_utils.CombinedNN(environment_spec.actions)
    elif FLAGS.env_id in {'pickplace', 'cookclassic', 'cooking', 'ETmining', 'ai2thor'}:
      network = snt_utils.RecurrentAttentionNN()
    else:
      network = snt_utils.RecurrentNN(environment_spec.actions)

    agent_args.update(
        network=network,
        n_step_horizon=FLAGS.n_step_horizon,
        learning_rate=FLAGS.learning_rate,
        discount=FLAGS.gamma,
        entropy_cost=FLAGS.entropy_cost,
        baseline_cost=FLAGS.baseline_cost,
        max_abs_reward=FLAGS.max_abs_reward,
        max_gradient_norm=FLAGS.max_gradient_norm
    )
  if 'mtsgi' in FLAGS.algorithm:
    agent_args.update(
        prior_sample_mode=FLAGS.prior_sample_mode if FLAGS.label == 'meta_eval' else None,
        posterior_mode=FLAGS.posterior_mode,
    )
  elif FLAGS.algorithm == 'hrl':
    assert FLAGS.num_envs == 1, 'Currently HRL only supports single worker.'
    assert FLAGS.n_step_horizon <= FLAGS.num_adapt_steps // FLAGS.num_trial_splits, \
        'Make sure n-step horizon is less than equal to adaptation_steps.'
  elif FLAGS.algorithm == 'rlrl':
    minibatch_size = FLAGS.num_adapt_steps  // FLAGS.n_step_horizon
    agent_args.update(
        snapshot_dir=root_dir.replace('logs', 'snapshot') if FLAGS.save_snapshots else None, #
        minibatch_size=1, #
    )
  elif 'msgi' in FLAGS.algorithm:
    pass
  elif 'psgi' in FLAGS.algorithm:
    # TODO check to make sure this is correct behavior
    pass
  elif 'greedy' in FLAGS.algorithm:
    pass
  else:
    raise NotImplementedError

  meta_agent = meta_agent_class(**agent_args)

  # Run meta RL loop
  meta_loop = psgi.EnvironmentMetaLoop(
      adapt_environment=adapt_environment,
      test_environment=test_environment,
      meta_agent=meta_agent,
      logger=environment_logger,
      label=FLAGS.label,
      verbose_level=FLAGS.verbose_level
  )
  if FLAGS.label == 'meta_train':
    assert FLAGS.algorithm in META_TRAIN, \
        f'Error! {FLAGS.algorithm} does not support meta-training.'
    meta_loop.run(
        num_trials=FLAGS.num_trials,
        num_adapt_steps=FLAGS.num_adapt_steps,
        num_test_episodes=FLAGS.num_test_episodes,
        num_trial_splits=FLAGS.num_trial_splits,
        task_idx_debug=FLAGS.task_idx_debug
    )  # pytype: disable=attribute-error
    meta_agent.save(save_filename)
  else:
    if FLAGS.algorithm in META_TRAIN:
      if load_filenames is not None:
        meta_agent.load(load_filenames)
      else:
        raise ValueError('load_exp_id flag not set')
    meta_loop.run(
        num_trials=FLAGS.num_trials,
        num_adapt_steps=FLAGS.num_adapt_steps,
        num_test_episodes=FLAGS.num_test_episodes,
        num_trial_splits=FLAGS.num_trial_splits,
        task_idx_debug=FLAGS.task_idx_debug
    )  # pytype: disable=attribute-error

def get_hparam_str(FLAGS):
  hparam_str = ""
  load_hparam_str = ""
  # Exploration stretegy.
  if FLAGS.algorithm in USE_ILP:
    hparam_str += f"_explore={FLAGS.exploration}"
    if FLAGS.algorithm in {'mtsgi', 'psgi'}:
      if FLAGS.exploration == 'mtucb':
        load_hparam_str += f"_explore=ucb"  # loads ucb (not mtucb)
      elif FLAGS.exploration == 'grprop':
        load_hparam_str += f"_explore=count"  # loads ucb (not mtucb)
      else:
        load_hparam_str += f"_explore={FLAGS.exploration}"

    if FLAGS.exploration in {'ucb', 'mtucb'}:
      hparam_str += f"_temp={FLAGS.ucb_temp}"
      load_hparam_str += f"_temp={FLAGS.ucb_temp}"
    if FLAGS.label == 'meta_eval':
      hparam_str += f"_ILPnf={FLAGS.neccessary_first}" # only matters for meta-eval
  if FLAGS.algorithm in USE_CYCLE_GRPROP:
     if FLAGS.label == 'meta_eval':
        hparam_str += f"_Tgrprop={FLAGS.grprop_temp}"
        hparam_str += f"_sigma={FLAGS.grprop_sigma}"

  if FLAGS.algorithm in USE_A2C:
    hparam_str += f"_lr={FLAGS.learning_rate}"
    hparam_str += f"_ent={FLAGS.entropy_cost}"
    hparam_str += f"_crit={FLAGS.baseline_cost}"
    hparam_str += f"_nstep={FLAGS.n_step_horizon}"
  if 'mtsgi' in FLAGS.algorithm:
    if FLAGS.label == 'meta_eval':  # number of priors to load
      hparam_str += f"_nprior={FLAGS.num_prior_load}"
      hparam_str += f"_posterior={FLAGS.posterior_mode}"
      hparam_str += f"_prior={FLAGS.prior_sample_mode}"

  if FLAGS.load_exp_id >= 0:
    hparam_str += f"_load={FLAGS.load_exp_id}-{FLAGS.seed}"
  else:
    load_hparam_str = ""

  if len(hparam_str) == 0:
    hparam_str = "_default"
  return hparam_str, load_hparam_str

def argument_assertions():
  if FLAGS.algorithm in META_EVAL_ONLY:
    assert FLAGS.label == 'meta_eval', \
      '%s only supports meta-eval, not meta-train'%(FLAGS.algorithm)

  if FLAGS.label == 'meta_train':
    pass
    #assert FLAGS.num_trial_splits == 1
    #assert 'train' in FLAGS.graph_param # XXX: temporarily allow for debugging environment.
  else:
    assert FLAGS.num_trial_splits > 1
    assert 'eval' in FLAGS.graph_param

  assert FLAGS.num_trial_splits > 0, 'Evaluation period must be greater than zero.'
  assert FLAGS.num_adapt_steps % FLAGS.num_trial_splits == 0, \
      'The number of training steps must be a multiple of num_trial_splits.'

  if FLAGS.exploration == 'mtucb':
    assert FLAGS.algorithm == 'mtsgi' and FLAGS.label == 'meta_eval'

if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
