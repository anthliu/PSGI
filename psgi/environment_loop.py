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
"""A simple agent-environment training loop."""

import time
import numpy as np
from tqdm import tqdm

import dm_env
from acme import core
from acme.utils import counting
from acme.utils import loggers

from psgi import agents
from psgi.agents import meta_agent
from psgi.utils import log_utils


class EnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """
  def __init__(
      self,
      environment: dm_env.Environment,
      actor: core.Actor, *,
      validate_action: bool = True,
      label: str = 'environment_loop',
      verbose_level: int = 0
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actor = actor
    self._batch_size = environment.batch_size
    self._validate_action = validate_action

    # Initialize counter and loggers
    self._label = label
    self._counter = counting.Counter()

    # Maintain the adaptation timestep (for meta-eval)
    self._prev_timestep = None

    # Debugging
    self._verbose_level = verbose_level

  def run(
      self,
      num_steps: int = None,
      num_episodes: int = None,
      reset: bool = False,
      update: bool = False
  ):
    """Perform the run loop.

    Run the environment loop for `num_episodes` episodes. Each episode is itself
    a loop which interacts first with the environment to get an observation and
    then give that observation to the agent in order to retrieve an action. Upon
    termination of an episode a new episode will be started. If the number of
    episodes is not given then this will interact with the environment
    infinitely.

    Args:
      num_episodes: number of episodes to run the loop for. If `None` (default),
        runs without limit.
    """
    self._check_argument(num_steps, num_episodes)
    is_step_based = num_steps is not None
    is_episode_based = num_steps is None

    # Maybe reset environment
    if reset:
      timestep = self._environment.reset()
      self._prev_timestep = timestep
      # Make the first observation.
      self._actor.observe_first(timestep)
    else:
      # During meta-eval, we want to continue our adaptation phase
      # from the last timestep the agent saw.
      assert self._prev_timestep is not None, 'Environment is not reset'
      timestep = self._prev_timestep

    # Reset any counts and start the environment.
    step_count = 0
    episode_count = np.zeros(self._batch_size, dtype=np.int32)
    envs_terminated = np.full(self._batch_size, False, dtype=np.bool)
    cumulative_reward = 0
    success = 0

    def should_terminate(episode_count: np.ndarray, step_count: np.ndarray) -> bool:
      return ((is_episode_based and all(episode_count >= num_episodes)) or
              (is_step_based and step_count >= num_steps))

    # Run loop
    while not should_terminate(episode_count, step_count):
      # Generate an action from the agent's policy and step the environment.
      action = self._actor.select_action(timestep.observation)

      if self._validate_action:
        # Validate action eligibility (TODO: move it into environment).
        action_eligs = np.take_along_axis(
          timestep.observation['eligibility'],
          indices=np.expand_dims(action, axis=-1), axis=1).squeeze(-1)
        assert action_eligs.shape == timestep.step_type.shape, 'Shape is wrong'
        assert np.all(np.logical_or(
          action_eligs == 1, timestep.step_type == dm_env.StepType.LAST)), (
            "Selected actions {action} is ineligible")

      # XXX: Actions on the episode boundary will be ignored.
      # This is a workaround needed to not crash webnav envs (due to ineligible actions)
      action[timestep.step_type == dm_env.StepType.LAST] = -1   # NO_OP
      timestep = self._environment.step(action)

      self._actor.observe(action, next_timestep=timestep)

      # Book-keeping.
      rewards = timestep.reward * np.logical_not(envs_terminated).astype(np.float32)
      cumulative_reward += sum(rewards)
      success += timestep.observation['task_success'].sum()

      step_count += 1
      episode_count += timestep.last() * np.logical_not(envs_terminated)   # increment if last
      envs_terminated = episode_count >= num_episodes if is_episode_based else envs_terminated
      self._prev_timestep = timestep

      if update: # 83% of time
        # Adapt-agent: fast-learning & posterior update. test-actor: nothing.
        # RL: Update policy
        self._actor.update() # See _learner.step()
        """
        if hasattr(self._actor, 'is_learner_updated') and self._actor.is_learner_updated:
          self._environment.evaluate_inferred_task(self._actor.inferred_task) # TODO: implement prec-rec
          self._actor.sync_learner() # see agent.py """

    # Result over current run()
    result = {
        'cumulative_reward': cumulative_reward,
        'success': success,
        'steps': step_count * self._batch_size,
        'episodes': episode_count.sum(),
    }
    result = {f'{self._label}/{k}': v for k, v in result.items()} # Append 'label' as prefix
    # Accumulate result (result over the life time of this EnvLoop)
    counts = self._counter.increment(**result)
    return result, counts

  def _check_argument(self, num_steps, num_episodes):
    is_step_based = num_steps is not None and num_episodes is None
    is_episode_based = num_steps is None and num_episodes is not None
    assert is_episode_based or is_step_based, \
      'either one of {num_steps or num_episodes} should be specified'

  def reset_task(self, task_index=None): # if None, randomly sample task.
    tasks = self._environment.reset_task(task_index=task_index)
    self._counter = counting.Counter()
    return tasks

  def set_task(self, tasks): # task_index is required
    self._environment.set_task(tasks=tasks)
    self._counter = counting.Counter()


class EnvironmentMetaLoop(core.Worker):
  """A meta RL loop for meta-training or meta-testing."""

  def __init__(
      self,
      adapt_environment: dm_env.Environment,
      test_environment: dm_env.Environment,
      meta_agent: meta_agent.MetaAgent,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      label: str = "meta_train",
      verbose_level: int = 0,
  ):
    # Internalize counter and loggers.
    self._adapt_env = adapt_environment
    self._test_env = test_environment
    self._batch_size = self._adapt_env.batch_size
    self._counter = counter or counting.Counter() # Not used. TODO: consider removing
    self._logger = logger
    self._label = label
    self._verbose_level = verbose_level

    # Create train_loop and test_loop.
    # TODO: This looks not pretty. Improve interfaces.
    self._meta_agent = meta_agent
    self._agent = meta_agent.instantiate_adapt_agent()  # Fast agent
    self._test_actor = meta_agent.instantiate_test_actor()  # Test actor (no learning)

    self._adaptation_loop = EnvironmentLoop(
        self._adapt_env, self._agent,
        label='adaptation', verbose_level=self._verbose_level)
    self._test_loop = EnvironmentLoop(
        self._test_env, self._test_actor,
        label='test', verbose_level=self._verbose_level)

  def run(
      self,
      num_trials: int = 1,
      num_adapt_steps: int = 1000,
      num_test_episodes: int = 8,
      num_trial_splits: int = 10, # should be > 1 for meta-eval. should be == 1 for meta-train
      task_idx_debug: int = -1
  ):
    """Perform the meta RL loop.

    Args:
      num_trials: number of meta trials to perform.
      num_adapt_steps: total number of steps for a single adaptation loop.
      num_test_steps: total number of steps for a single test loop.
      num_trial_splits: number of splits of adaptation-test phases in a trial.
    """
    split_counters = [counting.Counter() for _ in range(num_trial_splits)]

    adaptation_steps_per_split = num_adapt_steps // num_trial_splits

    # Initialize average return over evaluation period.
    _start_time = time.time()
    for trial_idx in range(num_trials):  # for each task
      # Sample a new task. Set test env with the same task.
      # Note: this does not reset adapt/test envs.
      task_idx = trial_idx  #if label == 'meta_eval' else None # XXX: currently different graph across batch is not supported
      task_idx = task_idx_debug if task_idx_debug >= 0 else task_idx  # for debugging
      tasks = self._adaptation_loop.reset_task(task_index=task_idx)
      self._test_loop.set_task(tasks=tasks)

      # Reset the agent with a new task.
      self._meta_agent.reset_agent(environment=self._adapt_env)

      # Initialize loop variables & stats
      for split_iter in tqdm(range(num_trial_splits), desc=f'Task [{trial_idx}/{num_trials}]'):
        # Run adaptation phase (continued in the next loop)
        adaptation_result, adaptation_counts = self._adaptation_loop.run(
            num_steps=adaptation_steps_per_split,
            reset=True if split_iter == 0 else False, # reset episode
            update=True)

        self._meta_agent.update_adaptation_progress(split_iter + 1, num_trial_splits)
        if isinstance(self._meta_agent, agents.PSGI):
          print('# Executed options=', self._meta_agent._actor.num_option_executed)

        # TODO: Handle this better.
        if self._label == 'meta_train' and hasattr(self._test_actor, '_state'):  # RLRL
          self._test_actor._state = self._agent._actor._state

        # Run test phases.
        if not hasattr(self._meta_agent, 'skip_test') or not self._meta_agent.skip_test:
          test_result, _ = self._test_loop.run(
              num_episodes=num_test_episodes,
              reset=True,
              update=False)
          # Normalize data
          test_result['test/cumulative_reward'] /= test_result['test/episodes']
          test_result['test/success'] /= test_result['test/episodes']
          print(test_result)
        else:
          test_result = {}

        # Normalize data
        adaptation_counts['adaptation/steps'] /= self._batch_size
        adaptation_result['adaptation/cumulative_reward'] /= adaptation_result['adaptation/episodes']
        adaptation_result['adaptation/success'] /= adaptation_result['adaptation/episodes']

        # Accumulate
        split_counters[split_iter].increment(tasks=1)
        split_counters[split_iter].increment(**adaptation_counts) # use 'count' since adapt_loop is continued
        split_counters[split_iter].increment(**test_result) # use 'result' since test_loop is independent

        # Adapdation-test is done, do meta-learning updates.
        if self._label == 'meta_train':
          self._meta_agent.update(last=(split_iter == num_trial_splits - 1))

      # Log & print summarized data
      if self._logger and self._label == 'meta_eval':
        split_counts = [counter.get_counts() for counter in split_counters]
        # Average over trials
        # TODO: The data is a list, so only compatible with CSVDumper logger.
        split_avg_log = [{key: (val if key == 'tasks' else val / counts['tasks']) \
                          for key, val in counts.items()} for counts in split_counts]
        self._logger.write(split_avg_log) # overwrite
      elif self._logger and self._label == 'meta_train':
        adaptation_result.update({'trial': trial_idx})
        adaptation_result.update(test_result)
        self._logger.write(adaptation_result)

      # Time log
      elapsed = (time.time() - _start_time) / 60.
      remain = elapsed / (trial_idx + 1) * (num_trials - trial_idx - 1)
      print(f'remain={remain:.1f}min  | elapsed={elapsed:.1f}min')
