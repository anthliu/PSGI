from typing import Optional, List, Callable, Dict
from psgi.utils.graph_utils import SubtaskGraph
import abc
import numpy as np

from acme import types, specs

from psgi.envs.predicate_graph import _parse_name
from psgi.envs.logic_graph import eval_precision_recall
from psgi.envs import BasePredicateEnv

ENGINE_OPTIONS = ["pick_up", "put_on", "slice", "cook_a_with_b"]

ENGINE_OP_MAP = {
    "op_put": "put_a_on_b",
    "op_pickup": "pick_up_a_from_b",
    "op_slice": "slice_a_at_b",
    "op_cook": "cook_a_with_b",
    "op_clean": "clean_a_with_b",
    "op_fillcoffee": "make_coffee",
    "op_fillwater": "fill_a_with_water_using_b",
    "op_spill": "spill"
}

class AI2ThorEnv(BasePredicateEnv):
  def __init__(
      self,
      rank: int,
      config_factory: List[Callable],
      feature_mode: str,
      graph_param: str,
      keep_pristine: bool,
      verbose_level: Optional[int] = 0,
  ):
    # Set environment configurations
    self.rank = rank
    self._graph_param = graph_param
    self._feature_mode = feature_mode

    # Set task-info. (For multi-task env, set this in reset_task())
    self._configs = config_factory
    self._keep_pristine = keep_pristine
    self._verbose_level = verbose_level
    assert all([callable(config) for config in self._configs])

    # Sample at least one valid task config (default seed = 0).
    self.reset_task(task_index=0)

  def reset_task(self, task_index: Optional[int] = None):
    # TODO: Remove the assertion and support random task sampling when different
    # graph across the batch is supported. (See environment_loop.py L225)
    assert task_index is not None
    config_factory = self._configs[task_index % len(self._configs)]

    self.config = config_factory(
      seed=task_index,
      feature_mode=self._feature_mode,
      graph_param=self._graph_param,
      keep_pristine=self._keep_pristine)
    self._update_config(self.config)

    return self.config

  def reset(self):
    # Reset episode-based internal states
    self.step_count = 0
    self.done = False
    self.step_done = False
    self.completion = None
    self.eligibility = None
    self.task_success = False
    self.subtask_ever_completed = set()

    # Reset & update batch_params
    self._update_batch_params(reset=True)
    assert not self.done, "done cannot be True here"

    observation = self._get_observation()
    return observation


  def _update_batch_params(self, action=None, step=None, reset=False):
    '''Reset & Update mask, completion, and eligibility.
    '''
    # completion
    completed_subtask_names = []
    if reset:
      next_completion = self.initial_completion
    else:
      next_completion = self.completion.copy()
      effect = self.graph.compute_effect(option=action)
      next_completion.update(effect)
      self.reward = 0.
      for subtask_name, value in effect.items():
        if value == True and self.completion[subtask_name] == False:
          completed_subtask_names.append(subtask_name)
    self.completion = next_completion
    
    # reward, task_success, task_over_by_terminal_subtask
    self.reward = 0.
    for subtask_name in completed_subtask_names:
      if subtask_name in self.config.repeatable_subtasks or subtask_name not in self.subtask_ever_completed:
        self.reward += self.config.subtask_reward[subtask_name]
      if subtask_name not in self.subtask_ever_completed:
        self.subtask_ever_completed.add(subtask_name)
      
    self.task_success = self._check_task_success(completed_subtask_names)
    task_over_by_terminal_subtask = self._check_environment_done(completed_subtask_names)
    
    # eligibility
    attributes = dict(next_completion, **self.feature)
    self.eligibility = self._update_eligibility(attributes)
    """
    goal_subtask_name = [name for name, rew in self.config.subtask_reward.items() if rew>0.9]
    for option_name, elig in self.eligibility.items():
      params = goal_subtask_name[0][1:-1].split(',')
      if all([param in option_name for param in params]):
        if elig:
          print(f'[{self.step_count}Goal subtask {goal_subtask_name[0]} became eligible!!')
          import ipdb; ipdb.set_trace()"""

    # 2. time_over
    if step is not None:
      self.step_count += step
    time_over = self.step_count >= self.max_step # Time exceeds episode limit
    agent_over = not any(self.eligibility) # No more action (option) left to execute

    self.done = time_over or agent_over or task_over_by_terminal_subtask
    self.step_done = time_over

    if self._verbose_level > 1 and self.rank == 0:
      print('='*40)
      print('[%d] Available options:'%(self.step_count))
      for subtask_name in self.eligibility:
        if self.eligibility[subtask_name] and self.mask[subtask_name]:
          print(subtask_name)

  def _get_observation(self):
    # Comprehensive version. slower.
    assert len(self.subtask_name) == len(self.completion), "Error: completion and subtask_name does not match"
    completion = np.array([self.completion[name] for name in self.subtask_name]) # no feature included
    eligibility = [1 if option_name in self.eligibility and self.eligibility[option_name] else 0 for option_name in self.option_name]
    eligibility = np.array(eligibility, dtype=np.float32)
    parameter_embeddings = self.feature_mat.copy()
    observation = {
        'mask': np.ones_like(eligibility).astype(np.float32), # XXX: remove mask from the entire code
        'completion': completion.astype(np.float32),
        'eligibility': eligibility.astype(np.float32),
        'parameter_embeddings': parameter_embeddings.astype(np.float32),
        'subtask_param_embeddings': self.subtask_param_embeddings,
        'option_param_embeddings': self.option_param_embeddings,
        'option_success': np.asarray(True),
        'task_success': np.asarray(self.task_success),
    }
    assert self.option_param_embeddings.shape[0] == eligibility.shape[0]
    # Remaining episode step.
    remaining_steps = (self.max_step + 1 - self.step_count)
    observation.update(
        remaining_steps=np.array(remaining_steps, dtype=np.int32),
        termination=np.array(self.done, dtype=np.bool),
        step_done=np.array(self.step_done, dtype=np.float32),
    )
    return observation

  def step(self, action: np.ndarray, p=False):
    assert self.config.subtask_reward is not None, \
        'Subtask reward are not defined. Please reset the task.'
    
    action_names = self._unwrap_action(action)
    assert self.eligibility[action_names], 'ineligible action'
    if self._verbose_level > 0 and self.rank == 0:
      print('[%d] Option name: %s'%(self.step_count, action_names))
    # 1. Reward, step
    step_count = 1  # XXX step: In web-nav, option == single primitive action

    # 2. Get next (comp, elig, mask, done, reward)
    self._update_batch_params(action_names, step_count)
    reward = self.reward
    """
    if reward > 0.25:
      print(f'[{self.step_count}]action={action_names}, reward={reward}')
    if reward > 0.9:
      print('goal achieved!')
      import ipdb; ipdb.set_trace()"""

    # 3. Prepare outputs
    observation = self._get_observation()
    done = self.done
    return observation, reward, done, step_count

