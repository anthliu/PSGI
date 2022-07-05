'''Configuration script for ToyDummy environment.'''
from typing import Optional, Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config as base
from psgi.envs.base_config import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


##### Stage 0
SUBTASK_LIST = ['Click Dummy GoPage1', 'Fill First']
LABEL_NAME = SUBTASK_LIST


# Additional Collections.
TERMINAL_SUBTASKS = ['Fill First']


class ToyDummy(WobConfig):
  environment_id = 'dummy'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.max_step = 30
    self.graph = self._construct_task(seed=seed, keep_pristine=keep_pristine)  # pylint: disable=no-member
    if keep_pristine:
      self._assert_subtask_set(self.subtasks, SUBTASK_LIST)

    self.subtask_reward = {subtask.name: subtask.reward for subtask in self.graph.nodes}
    assert len(self.subtasks) == len(self.subtask_reward)
    # TODO: revive later with EXTRA_OPTIONS.
    #self.subtask_reward['NO_OP'] = 0.

    self._construct_mappings()

    # Define special completion mechanism
    self.option_extra_outcome = self._construct_option_outcome()

  def _construct_task(self, seed: int, keep_pristine: bool = False):
    """Implement precondition & subtask reward & terminal subtasks
    """
    g = SubtaskLogicGraph('ToyDummy')

    ##### Stage 0. Distractors
    g.add_base(['Click Dummy GoPage1'])
    g['Fill First'] = g['Click Dummy GoPage1']

    # 4. Terminal subtasks
    self.terminal_subtasks = TERMINAL_SUBTASKS
    return g

  def _construct_option_outcome(self):
    """Implement a special mechanism of completion dynamics (i.e., one-step forward model)
    """
    option_extra_outcome = dict()
    # TODO: handle random perturbation.
    # Toggle
    #option_extra_outcome['Hide Items'] = {'Click Items': False}
    # Multiple choice
    ### Subpages
    #option_extra_outcome['Click Step1'] = {'Click Step0': False}
    #option_extra_outcome['Click Step2'] = {'Click Step0': False, 'Click Step1': False}
    #### Credit/Gift
    #option_extra_outcome['Click Credit'] = {'Click Gift': False}
    #option_extra_outcome['Click Gift'] = {'Click Credit': False}

    return option_extra_outcome
