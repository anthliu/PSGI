from typing import Optional, List, Callable, Dict
from psgi.utils.graph_utils import SubtaskGraph
import abc
import numpy as np

from acme import types, specs

from psgi.envs.predicate_graph import _parse_name
from psgi.envs.logic_graph import eval_precision_recall
from psgi.envs import BaseWoBEnv


class BasePredicateEnv(BaseWoBEnv):
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

  def _update_config(self, config):
    self.initial_completion = config.initial_completion
    self.feature = config.feature
    self.feature_mat = config.feature_mat
    assert self.feature_mat.shape[1] == len(config.feature_func_names), 'Shape error in feature.'

    # Set observation specs.
    observation = self.reset()
    self._observation_specs = {
      k: specs.Array(shape=v.shape, dtype=v.dtype, name=k) \
      for k, v in observation.items()
    }

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

  @property
  def parameters(self): # list of (local) parameter
    return self.config.param_set
  @property
  def parameter_name_to_index(self): # index of (local) parameter
    return self.config.parameter_name_to_index
  
  # API for option - eligibility
  @property
  def option_name(self):
    return self.config.options
  @property
  def option_param(self):
    return [_parse_name(name) for name in self.option_name]
  
  # API for subtask - completion
  @property
  def subtask_name(self):
    return self.config.subtasks
  @property
  def subtask_param(self):
    return [_parse_name(name) for name in self.subtask_name]
  
  # GT info
  @property
  def graph(self):
    return self.config.graph
  @property
  def subtask_reward(self):
    return np.array([self.config.subtask_reward[name] for name in self.subtask_name])

  # Essential
  @property
  def action_dim(self):
    return len(self.options)
  @property
  def num_graphs(self):
    return self.config.num_graphs
  @property
  def max_step(self):
    return self.config.max_step
  @property
  def task(self):
    return self.config

  ### Communicating with the agent (actor) --> should be id-based
  @property
  def task_embedding(self): # XXX: TODO
    # TODO: generate ground truth graph information that GRProp can interpret.
    return self.config

  ### For connecting adaptation-test envs ~~ ###
  def set_task(self, task):
    """fixed the current task to the given task. used for aligning the task
      between the adaptation & test environments.
    """
    self.config = task
    self._update_config(self.config)

  def evaluate_inferred_task(self, inferred_task: SubtaskGraph):
    """Measure Precision-recall against environment
    """
    gt_kmap = self.config.kmap_by_option # TODO: this should be over all options (not only feasible)
    kmap = inferred_task.kmap
    precision, recall = eval_precision_recall(gt_kmap, kmap)
    filename = os.path.join(self.dirname, 'graph_PR.txt')
    ep = 1 # TODO: update to iteration.
    with open(filename, 'a') as f:
      string = '{}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\n'.format(ep, precision.mean(), recall.mean(), 0., 0.)
      f.writelines( string )
    return precision, recall

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

  def _check_task_success(self, subtask_names: List[str]):
    """Check task ended with success."""
    return any([name in self.config.success_subtasks for name in subtask_names])

  def _check_environment_done(self, subtask_names: List[str]):
    """Check environment done."""
    return any([name in self.config.terminal_subtasks for name in subtask_names])
    return action in self.config.terminal_options

  def _update_eligibility(self, completion: Dict[str, bool]):
    """Update eligiblity."""
    eligibility = self.graph.compute_eligibility(completion=completion)
    return eligibility

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_specs

  def action_spec(self) -> types.NestedSpec:
    return specs.DiscreteArray(num_values=self.action_dim, dtype=np.int32)

  def reward_spec(self) -> types.NestedSpec:
    return specs.Array(shape=(), dtype=np.float32, name='reward')

  def _unwrap_action(self, action: np.ndarray) -> str:
    return self.option_name[action]

  @property
  def subtask_param_embeddings(self):
    # ndarray : #Subtasks x #parameters x #embedding dim
    return self.config.subtask_param_embeddings
  @property
  def option_param_embeddings(self):
    # ndarray : #Options x #parameters x #embedding dim
    return self.config.option_param_embeddings
  
