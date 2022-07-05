'''Configuration script for Dummy environment.'''
from typing import Optional, Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config as base
from psgi.envs.base_config import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


##### Stage 0
BASE_SUBTASKS = [
    'Click Help',
    'Click RP',
    'Click Subscribe',
    'Click Zip',
    'Click Dummy GoPage1'
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

FINISH1_SUBTASKS = [
    'Click FinishShipping',
    'Click FinishContact'
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

FINISH2_SUBTASKS = ['Click FinishPayment']

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
    FINISH1_SUBTASKS + ['Click Dummy GoPage2'] + CREDIT_SUBTASKS + \
    GIFT_SUBTASK + ['Click G_Apply'] + PAYMENT_SUBTASKS + FINISH2_SUBTASKS + \
    ['Click Dummy GoPage3'] + FINAL_SUBTASKS
LABEL_NAME = SUBTASK_LIST


# Additional Collections.
FAILURE_SUBTASK = ['Click Help', 'Click RP']
SUCCESS_SUBTASK = ['Click Place Order']
TERMINAL_SUBTASKS = FAILURE_SUBTASK + SUCCESS_SUBTASK


class Dummy(WobConfig):
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
    # Create graph with random perturbation.
    rng = np.random.RandomState(seed)  # pylint: disable=no-member

    g = SubtaskLogicGraph('Dummy')

    ##### Stage 0. Distractors
    g.add_base(BASE_SUBTASKS)

    if not keep_pristine:
      # bring all the possible subtasks (prune later).
      shipping_subtasks = base.SHIPPING_SUBTASKS
      billing_subtasks = base.BILLING_SUBTASKS
      credit_subtasks = base.CREDIT_SUBTASKS
      terminal_subtasks = TERMINAL_SUBTASKS.copy()
    else:
      shipping_subtasks = SHIPPING_SUBTASKS
      credit_subtasks = CREDIT_SUBTASKS
      terminal_subtasks = TERMINAL_SUBTASKS.copy()

    ##### Stage 1. Fill address info
    with g.stage("Address"):
      g.add_one_to_many(source='Click Dummy GoPage1',
                        sinks=shipping_subtasks + CONTACT_SUBTASKS)

      # Sample preconditions from shipping & contact & shipmethod subtasks.
      if not keep_pristine:
        shipping_required = graph_utils.sample_subtasks(rng, pool=shipping_subtasks,
                                                        minimum_size=1)
        contact_required = graph_utils.sample_subtasks(rng, pool=CONTACT_SUBTASKS,
                                                       minimum_size=1)
      else:  # pristine ver.
        shipping_required = SHIPPING_REQUIRED
        contact_required = CONTACT_SUBTASKS

      g.add_many_to_one(sources=shipping_required, sink='Click FinishShipping')
      g.add_many_to_one(sources=contact_required, sink='Click FinishContact')
      g.add_many_to_one(sources=FINISH1_SUBTASKS, sink="Click Dummy GoPage2")

    ##### Stage 2. Fill Credit info
    g.add_one_to_many(source='Click Dummy GoPage2', sinks=PAYMENT_SUBTASKS)

    # 2-1. Click Credit
    g.add_one_to_many(source='Click Credit', sinks=credit_subtasks)

    # 2-2. Click Gift
    g.add_one_to_many(source='Click Gift', sinks=GIFT_SUBTASK)

    g['Click G_Apply'] = g['Fill G_NUM'] & (g['Fill G_PIN'] | g['Click G_No PIN'])


    # Sample preconditons from billing & credit subtasks.
    if not keep_pristine:
      credit_required = graph_utils.sample_subtasks(rng, pool=credit_subtasks,
                                               minimum_size=1)
    else:  # pristine ver.
      credit_required = credit_subtasks

    # 2-z. Last
    g.add_many_to_one(sources=credit_required, sink='Click FinishPayment')
    g["Click Dummy GoPage3"] = g['Click FinishPayment']

    ##### Stage 3. Last stage
    g.add_one_to_many(source='Click Dummy GoPage3', sinks=FINAL_SUBTASKS)

    # 3. Define subtask rewards
    g.add_reward('Click Place Order', 5.)
    g.add_reward('Click Help', -1.)
    g.add_reward('Click RP', -1.)

    # Apply graph perturbation if necessary.
    # TODO: Remove hardcoded probabilities.
    if not keep_pristine:
      # Skip some bottleneck states
      if rng.random() < 0.3:
        g.remove_node('Click Credit')
      if rng.random() < 0.3:
        g.remove_node('Click Gift')
      if rng.random() < 0.3:
        g.remove_node('Click Subscribe')
      if rng.random() < 0.3:
        g.remove_node('Click Zip')
      if rng.random() < 0.3:
        g.remove_node('Click EditShipping')

      # Skip some nodes (independently)
      for optionals in ['Click Help', 'Click RP']:
        if rng.random() < 0.5:
          g.remove_node(optionals)
          terminal_subtasks.remove(optionals)

      # Skip one page..?
      if rng.random() < 0.3:
        # Skip gift card completely
        g.remove_nodes(["Click Gift", "Fill G_PIN", "Fill G_NUM",
                        "Click G_No PIN", "Click G_Apply"],
                       skip_nonexistent=True)

       #TODO: Add optional task, such as “add promo code”, or “add coupon code”,
       # or “use gift card balance”, etc,  before confirmation

      # Randomly remove some shipping & billing subtasks
      #if rng.random() < 0.4:
      #  shipping = graph_utils.sample_subtasks(rng, pool=shipping_subtasks,
      #                                       minimum_size=1)
      #  g.remove_nodes(shipping)

      #if rng.random() < 0.4:
      #  credit = graph_utils.sample_subtasks(rng, pool=credit_subtasks,
      #                                       minimum_size=1)
      #  g.remove_nodes(credit)

      # Add distractors & failures
      #g, _ = graph_utils.add_sampled_nodes(
      #    graph=g, rng=rng, pool=base.DISTRACTORS,
      #    minimum_size=1, maximum_size=None
      #)

      #g, failures_added = graph_utils.add_sampled_nodes(
      #    graph=g, rng=rng, pool=base.FAILURE_SUBTASKS,
      #    minimum_size=1, maximum_size=4
      #)

      #for failure in failures_added:
      #  if failure in base.FAILURE_SUBTASKS:
      #    g.add_reward(failure, -rng.random())  # assign neg. reward
      #    terminal_subtasks.append(failure)

      # Add promotion, gift cards, etc.
      # TODO: Mix-match

    # 4. Terminal subtasks
    self.terminal_subtasks = terminal_subtasks
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
