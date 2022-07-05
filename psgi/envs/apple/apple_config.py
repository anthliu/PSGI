'''Configuration script for Apple environment.'''
from typing import Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config as base
from psgi.envs.base_config import WobConfig, FAILURE
from psgi.envs.logic_graph import SubtaskLogicGraph


# Stage 0 (base)
BASE_SUBTASKS = [
    'Click Subscribe',
    'Click Items',
    'Click Zip',
    'Click Apple GoPage1'
]

# Stage 1 (contact & shipping addresses)
SHIPPING_REQUIRED = [
    'Fill First', 'Fill Last',
    'Fill Street', 'Fill State', 'Fill City', 'Fill Zip'
]
SHIPPING_OPTIONAL = ['Fill Apt']
SHIPPING_SUBTASKS = SHIPPING_REQUIRED + SHIPPING_OPTIONAL

CONTACT_SUBTASKS = [
    'Fill Email',
    'Fill Phone'
]

SHIPMETHOD_SUBTASK = ['Click ShipMethod']
SELECT_SHIPMETHOD_SUBTASKS = [
    'Click Standard',
    'Click Expedited',
    'Click NextDay'
]

FINISH1_SUBTASKS = [
    'Click FinishShipping',
    'Click FinishContact'
]
CONTINUE_SUBTASK = ['Click Apple GoPage2']  # Continue to payment

# Stage 2 (payments)
BILLING_REQUIRED = [
    'Fill BillFirst', 'Fill BillLast',
    'Fill BillStreet', 'Fill BillState', 'Fill BillCity', 'Fill BillZip'
]
BILLING_OPTIONAL = ['Fill BillApt']
BILLING_SUBTASKS = BILLING_REQUIRED + BILLING_OPTIONAL

SELECT_PAYMENT_SUBTASKS = [
    'Click Credit',
    'Click Gift',
    'Click PayPal'
]

FILL_CREDIT_SUBTASKS = [
    'Fill C_NUM',
    'Fill C_EXPMM',
    'Fill C_EXPYY',
    'Fill C_CVV'
]
FILL_GIFT_SUBTASKS = [
    'Fill G_NUM',
    'Fill G_PIN'
]

FINISH2_SUBTASKS = [
    'Click FinishPayment',
    'Click FinishBilling'
]

CONTINUE2_SUBTASK = ['Click Apple GoPage3']

# Stage 3 (review)
FINAL_SUBTASKS = [
    'Click EditShipMethod', 'Click EditShipping', 'Click EditBilling',
    'Click Place Order', 'Click Request PI', 'Click ReturnCart'
]

SUBTASK_LIST = BASE_SUBTASKS + SHIPPING_SUBTASKS + CONTACT_SUBTASKS + \
    SHIPMETHOD_SUBTASK + SELECT_SHIPMETHOD_SUBTASKS + FINISH1_SUBTASKS + CONTINUE_SUBTASK + \
    BILLING_SUBTASKS + SELECT_PAYMENT_SUBTASKS + FILL_CREDIT_SUBTASKS + \
    FILL_GIFT_SUBTASKS + FINISH2_SUBTASKS + CONTINUE2_SUBTASK + FINAL_SUBTASKS
LABEL_NAME = SUBTASK_LIST


# Additional Collections.
FAILURE_SUBTASKS = ['Click Request PI', 'Click ReturnCart']
TERMINAL_SUBTASKS = FAILURE_SUBTASKS + ['Click Place Order']


class Apple(WobConfig):
  environment_id = 'apple'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.max_step = 36
    self.graph = self._construct_task(seed=seed, keep_pristine=keep_pristine)
    if keep_pristine:
      self._assert_subtask_set(self.subtasks, SUBTASK_LIST)

    # Define subtask rewards.
    self.subtask_reward = {subtask.name: subtask.reward for subtask in self.graph.nodes}
    assert len(self.subtasks) == len(self.subtask_reward)

    # TODO: revive later with EXTRA_OPTIONS.
    # Additional subtask reward for NO_OP.
    #self.subtask_reward['NO_OP'] = 0.

    self._construct_mappings()

    # Define special completion mechanism (e.g. toggling)
    self.option_extra_outcome = self._construct_option_outcome()

  def _construct_task(self, seed: int, keep_pristine: bool = False):
    """Implement precondition & subtask reward & terminal subtasks
    """
    # Create graph with random perturbation.
    rng = np.random.RandomState(seed)  # pylint: disable=no-member

    g = SubtaskLogicGraph('Apple')

    if not keep_pristine:
      # Sample base layer distractors & failure nodes.
      distractors = graph_utils.sample_subtasks(rng, pool=base.EXTRA_DISTRACTORS,
                                                minimum_size=0)
      base_failures = graph_utils.sample_subtasks(rng, pool=base.FAILURES_SUBSET1,
                                             minimum_size=0, maximum_size=3)
      base_subtasks = distractors + base_failures + ['Click Apple GoPage1']
      terminal_subtasks = base_failures.copy()
    else:
      base_subtasks = BASE_SUBTASKS
      terminal_subtasks = TERMINAL_SUBTASKS.copy()

    # Add stage 0 (base) subtasks.
    g.add_base(base_subtasks)

    if not keep_pristine:
      # bring all the possible subtasks (prune later).
      shipping_subtasks = base.SHIPPING_SUBTASKS
      billing_subtasks = base.BILLING_SUBTASKS
      credit_subtasks = base.CREDIT_SUBTASKS
    else:
      shipping_subtasks = SHIPPING_SUBTASKS
      billing_subtasks = BILLING_SUBTASKS
      credit_subtasks = FILL_CREDIT_SUBTASKS

    # Connect stage 0 ==> stage 1 (contact & addresses) subtasks.
    with g.stage("Shipping Address"):
      g.add_one_to_many(
          source='Click Apple GoPage1',
          sinks=shipping_subtasks + CONTACT_SUBTASKS + SHIPMETHOD_SUBTASK
      )

      # Connect 'Click ShipMethod' ==> shipping method subtasks.
      g.add_one_to_many(
          source='Click ShipMethod',
          sinks=SELECT_SHIPMETHOD_SUBTASKS
      )

      # Sample preconditions from shipping & contact & shipmethod subtasks.
      if not keep_pristine:
        shipping_required = graph_utils.sample_subtasks(rng, pool=shipping_subtasks,
                                                        minimum_size=1)
        contact_required = graph_utils.sample_subtasks(rng, pool=CONTACT_SUBTASKS,
                                                       minimum_size=1)
      else:  # pristine ver.
        shipping_required = SHIPPING_REQUIRED
        contact_required = CONTACT_SUBTASKS

      # Connect Shipping & Contact ==> 'Click Apple GoPage2'
      g.add_many_to_one(sources=shipping_required, sink='Click FinishShipping')
      g.add_many_to_one(sources=contact_required, sink='Click FinishContact')
      g.add_many_to_one(sources=FINISH1_SUBTASKS, sink='Click Apple GoPage2')

    # Connect 'Click Apple GoPage2' ==> stage 2 (payment) subtasks.
    g.add_one_to_many(
        source='Click Apple GoPage2',
        sinks=SELECT_PAYMENT_SUBTASKS + billing_subtasks
    )

    # Connect 'Click Credit' ==> credit info subtasks.
    g.add_one_to_many(
        source='Click Credit',
        sinks=credit_subtasks
    )

    # Connect 'Click Gift' ==> gift card info subtasks.
    g.add_one_to_many(
        source='Click Gift',
        sinks=FILL_GIFT_SUBTASKS
    )

    # Sample preconditons from billing & credit subtasks.
    if not keep_pristine:
      billing_required = graph_utils.sample_subtasks(rng, pool=billing_subtasks,
                                                     minimum_size=1)
      credit_required = graph_utils.sample_subtasks(rng, pool=credit_subtasks,
                                                    minimum_size=1)
    else:  # pristine ver.
      billing_required = BILLING_REQUIRED
      credit_required = credit_subtasks

    # Connect credit & billing subtasks ==> 'Click Apple GoPage3'
    g.add_many_to_one(sources=billing_required, sink='Click FinishBilling')
    g.add_many_to_one(sources=credit_required, sink='Click FinishPayment')
    g.add_many_to_one(sources=FINISH2_SUBTASKS, sink='Click Apple GoPage3')

    if not keep_pristine:
      # Sample final layer distractors & failure nodes.
      distractors = graph_utils.sample_subtasks(rng, pool=base.EDIT_DISTRACTORS,
                                                minimum_size=0)
      final_failures = graph_utils.sample_subtasks(rng, pool=base.FAILURES_SUBSET2,
                                             minimum_size=0, maximum_size=3)
      final_subtasks = distractors + final_failures + ['Click Place Order']
      terminal_subtasks += final_failures + ['Click Place Order']
    else:
      final_subtasks = FINAL_SUBTASKS

    # Connect 'Click Apple GoPage3' ==> final subtasks.
    g.add_one_to_many(
        source='Click Apple GoPage3',
        sinks=final_subtasks
    )

    # Define subtask rewards.
    g.add_reward('Click Place Order', 5.)
    if not keep_pristine:
      for failure in base_failures + final_failures:
        g.add_reward(failure, -rng.random())  # assign neg. reward
    else:
      g.add_reward('Click Request PI', -1)
      g.add_reward('Click ReturnCart', -1)

    if not keep_pristine:
      # Skip some optional subtasks
      if rng.random() < 0.3:
        g.remove_node('Click Gift')
      if rng.random() < 0.3:
        g.remove_node('Click PayPal')
      if rng.random() < 0.1:
        g.remove_node('Click Standard')
      if rng.random() < 0.2:
        g.remove_node('Click Expedited')
      if rng.random() < 0.3:
        g.remove_node('Click NextDay')

    # Terminal subtasks
    self.terminal_subtasks = terminal_subtasks
    return g

  def _construct_option_outcome(self):
    """Implement a special mechanism of completion dynamics.
      (i.e., one-step forward model)
    """
    option_extra_outcome = dict()

    # TODO: Add toggling completions & handle random perturbation.
    # Select payment subtasks.
    #option_extra_outcome = env_utils.add_toggle_completion(
    #    toggle_outcomes=option_extra_outcome,
    #    subtasks=SELECT_PAYMENT_SUBTASKS
    #)
    return option_extra_outcome
