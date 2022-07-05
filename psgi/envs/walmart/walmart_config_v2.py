'''Configuration script for Walmart environment.'''
from typing import Optional, Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config_v2 as base
from psgi.envs.base_config_v2 import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


##### Stage 0
BASE_SUBTASKS = [
    'Click Help',
    'Click RP',
    'Click Subscribe',
    'Click Zip',
    'Click ContinueBase'
]
_BASE_DISTRACTORS = [
    'Click Help',
    'Click RP',
    'Click Subscribe',
    'Click Zip'
]

##### Stage 1
SHIPPING_REQUIRED = [
    'Fill First', 'Fill Last',
    'Fill Zip', 'Fill Street', 'Fill City', 'Fill State', 'Fill Country'
]
SHIPPING_OPTIONAL = ['Fill Apt']
SHIPPING_SUBTASKS = SHIPPING_REQUIRED + SHIPPING_OPTIONAL

CONTACT_SUBTASKS = [
    'Fill Email',
    'Fill Phone'
]

CONTINUE1_SUBTASKS = [
    'Click ContinueShipping',
    'Click ContinueContact'
]

##### Stage 2
PAYMENT_SUBTASKS = [
    'Click Credit',
    'Click Gift'
]
CREDIT_SUBTASKS = [
    'Fill C_First', 'Fill C_Last',
    'Fill C_NUM', 'Fill C_EXPMM',
    'Fill C_EXPYY', 'Fill C_CVV',
    'Fill C_Phone'
]

CONTINUE2_SUBTASKS = ['Click ContinuePayment']

GIFT_FIELD = ['G_NUM', 'G_PIN']
GIFT_BUTTON = ['G_No PIN']
GIFT_SUBTASK = ['Fill %s' % (field) for field in GIFT_FIELD]
GIFT_SUBTASK = GIFT_SUBTASK + ['Click %s' % (button) for button in GIFT_BUTTON]

##### Stage 3
FINAL_SUBTASKS = [
    'Click Place Order',
    'Click EditShipping'
]

SUBTASK_LIST = BASE_SUBTASKS + SHIPPING_SUBTASKS + CONTACT_SUBTASKS + \
    CONTINUE1_SUBTASKS + CREDIT_SUBTASKS + \
    GIFT_SUBTASK + ['Click G_Apply'] + PAYMENT_SUBTASKS + CONTINUE2_SUBTASKS + \
     FINAL_SUBTASKS
LABEL_NAME = SUBTASK_LIST


# Additional Collections.
FAILURE_SUBTASK = ['Click Help', 'Click RP']
SUCCESS_SUBTASK = ['Click Place Order']
TERMINAL_SUBTASKS = FAILURE_SUBTASK + SUCCESS_SUBTASK


class Walmart2(WobConfig):
  environment_id = 'walmart_v2'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.seed = seed if not keep_pristine else None
    self.max_step = 29  # default (pristine)
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

  def _perturb_subtasks(self, rng):
    # Shipping
    extra_shipping = graph_utils.sample_subtasks(rng, pool=['Fill Full', 'Fill Address', 'Fill SecurityCode'],
                                                    minimum_size=0)
    shipping_subtasks = SHIPPING_SUBTASKS + extra_shipping
    self.max_step += len(extra_shipping)
    if 'Fill Full' in shipping_subtasks:
      shipping_subtasks.remove('Fill First')
      shipping_subtasks.remove('Fill Last')
      self.max_step -= 2

    # Credit
    credit_subtasks = CREDIT_SUBTASKS.copy()
    if rng.random() < 0.4:
      credit_subtasks.remove('Fill C_Phone')
      self.max_step -= 1
    if rng.random() < 0.4:
      credit_subtasks.remove('Fill C_First')
      credit_subtasks.remove('Fill C_Last')
      self.max_step -= 2

    return shipping_subtasks, credit_subtasks

  def _construct_task(self, seed: int, keep_pristine: bool = False):
    """Implement precondition & subtask reward & terminal subtasks
    """
    # Create graph with random perturbation.
    rng = np.random.RandomState(seed)  # pylint: disable=no-member

    g = SubtaskLogicGraph('Walmart')

    if not keep_pristine:
      # Sample base layer distractors & failure nodes.
      distractors = graph_utils.sample_subtasks(rng, pool=base.EXTRA_DISTRACTORS,
                                                minimum_size=0)
      base_failures = graph_utils.sample_subtasks(rng, pool=base.FAILURES_SUBSET1,
                                             minimum_size=0, maximum_size=3)
      base_subtasks = distractors + base_failures + ['Click ContinueBase']
      terminal_subtasks = base_failures.copy()
      self.max_step += len(base_subtasks) - len(BASE_SUBTASKS)
      self._base_distractors = distractors + base_failures
    else:
      base_subtasks = BASE_SUBTASKS
      terminal_subtasks = TERMINAL_SUBTASKS.copy()
      self._base_distractors= _BASE_DISTRACTORS

    ##### Stage 0. Base & Distractors
    g.add_base(base_subtasks)

    if not keep_pristine:
      shipping_subtasks, credit_subtasks = self._perturb_subtasks(rng=rng)
      contact_subtasks = CONTACT_SUBTASKS
    else:
      shipping_subtasks = SHIPPING_SUBTASKS
      contact_subtasks = CONTACT_SUBTASKS
      credit_subtasks = CREDIT_SUBTASKS

    ##### Stage 1. Fill address info
    with g.stage("Address"):
      g.add_one_to_many(source='Click ContinueBase',
                        sinks=shipping_subtasks + contact_subtasks)

      g.add_many_to_one(sources=shipping_subtasks, sink='Click ContinueShipping')
      g.add_many_to_one(sources=contact_subtasks, sink='Click ContinueContact')

    ##### Stage 2. Fill Credit info
    g.add_many_to_many(sources=CONTINUE1_SUBTASKS, sinks=PAYMENT_SUBTASKS)

    # 2-1. Click Credit
    g.add_one_to_many(source='Click Credit', sinks=credit_subtasks)

    # 2-2. Click Gift
    g.add_one_to_many(source='Click Gift', sinks=GIFT_SUBTASK)

    g['Click G_Apply'] = g['Fill G_NUM'] & g['Fill G_PIN']

    # 2-z. Last
    g.add_many_to_one(sources=credit_subtasks, sink='Click ContinuePayment')

    if not keep_pristine:
      # Sample final layer distractors & failure nodes.
      distractors = graph_utils.sample_subtasks(rng, pool=base.EDIT_DISTRACTORS,
                                                minimum_size=0)
      final_failures = graph_utils.sample_subtasks(rng, pool=base.FAILURES_SUBSET2,
                                             minimum_size=0, maximum_size=3)
      final_subtasks = distractors + final_failures + ['Click Place Order']
      terminal_subtasks += final_failures + ['Click Place Order']
      self.max_step += len(final_subtasks) - len(FINAL_SUBTASKS)
    else:
      final_subtasks = FINAL_SUBTASKS

    ##### Stage 3. Last stage
    g.add_many_to_many(sources=CONTINUE2_SUBTASKS, sinks=final_subtasks)

    # 3. Define subtask rewards
    g.add_reward('Click Place Order', 5.)
    if not keep_pristine:
      for failure in base_failures + final_failures:
        g.add_reward(failure, -rng.random())  # assign neg. reward
    else:
      g.add_reward('Click Help', -1.)
      g.add_reward('Click RP', -1.)

    # Apply graph perturbation if necessary.
    # TODO: Remove hardcoded probabilities.
    if not keep_pristine:
      # Skip some bottleneck states
      if rng.random() < 0.3:
        g.remove_node('Click G_No PIN')
      if rng.random() < 0.3:
        rnum = rng.random()
        if rnum < 0.4:
          g.remove_nodes(['Click Gift', 'Fill G_PIN', 'Fill G_NUM', 'Click G_Apply'])
        else:
          g.remove_nodes(['Fill G_PIN', 'Fill G_NUM', 'Click G_Apply'])

      # Skip one page..?
      if rng.random() < 0.3:
        # Skip gift card completely
        g.remove_nodes(["Click Gift", "Fill G_PIN", "Fill G_NUM",
                        "Click G_No PIN", "Click G_Apply"],
                       skip_nonexistent=True)

    # 4. Terminal subtasks
    self.terminal_subtasks = terminal_subtasks
    return g

  def get_previous_subtasks(self, completion: Dict[str, bool]):
    subtasks = []
    if completion['Click ContinueShipping'] and completion['Click ContinueContact']:
      subtasks += self._base_distractors
    if completion['Click ContinuePayment']:
      subtasks += ['Click Gift'] + GIFT_SUBTASK + ['Click G_Apply']
    return subtasks

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
