'''Configuration script for Apple environment.'''
from typing import Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config_v2 as base
from psgi.envs.base_config_v2 import WobConfig, FAILURE
from psgi.envs.logic_graph import SubtaskLogicGraph


# Stage 0 (base)
BASE_SUBTASKS = [
    'Click Subscribe',
    'Click Items',
    'Click Zip',
    'Click ContinueBase'
]
_BASE_DISTRACTORS = [
    'Click Subscribe',
    'Click Items',
    'Click Zip',
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

CONTINUE1_SUBTASKS = [
    'Click ContinueShipping',
    'Click ContinueContact'
]
#CONTINUE_SUBTASK = ['Click Apple GoPage2']  # Continue to payment

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

CONTINUE2_SUBTASKS = [
    'Click ContinuePayment',
    'Click ContinueBilling'
]

#CONTINUE2_SUBTASK = ['Click Apple GoPage3']

# Stage 3 (review)
FINAL_SUBTASKS = [
    'Click EditShipMethod', 'Click EditShipping', 'Click EditBilling',
    'Click Place Order', 'Click Request PI', 'Click ReturnCart'
]

SUBTASK_LIST = BASE_SUBTASKS + SHIPPING_SUBTASKS + CONTACT_SUBTASKS + \
    SHIPMETHOD_SUBTASK + SELECT_SHIPMETHOD_SUBTASKS + CONTINUE1_SUBTASKS + \
    BILLING_SUBTASKS + SELECT_PAYMENT_SUBTASKS + FILL_CREDIT_SUBTASKS + \
    FILL_GIFT_SUBTASKS + CONTINUE2_SUBTASKS + FINAL_SUBTASKS
LABEL_NAME = SUBTASK_LIST


# Additional Collections.
FAILURE_SUBTASKS = ['Click Request PI', 'Click ReturnCart']
TERMINAL_SUBTASKS = FAILURE_SUBTASKS + ['Click Place Order']


class Apple2(WobConfig):
  environment_id = 'apple_v2'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.seed = seed if not keep_pristine else None
    self.max_step = 37
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

  def _perturb_subtasks(self, rng):
    # Shipping
    shipping_pool = ['Fill Full', 'Fill Address', 'Fill SecurityCode',
                     'Fill Country']
    extra_shipping = graph_utils.sample_subtasks(rng, pool=shipping_pool,
                                                    minimum_size=0)
    shipping_subtasks = SHIPPING_SUBTASKS + extra_shipping
    self.max_step += len(extra_shipping)
    if 'Fill Full' in shipping_subtasks:
      shipping_subtasks.remove('Fill First')
      shipping_subtasks.remove('Fill Last')
      self.max_step -= 2
    if rng.random() < 0.4:
      shipping_subtasks.remove('Fill State')
      self.max_step -= 1
    if rng.random() < 0.4:
      shipping_subtasks.remove('Fill City')
      self.max_step -= 1

    # Billing
    billing_pool = ['Fill BillFull', 'Fill BillAddress', 'Fill BillSecurityCode',
                    'Fill BillCountry']
    extra_billing = graph_utils.sample_subtasks(rng, pool=billing_pool,
                                                    minimum_size=0)
    billing_subtasks = BILLING_SUBTASKS + extra_billing
    self.max_step += len(extra_billing)
    if 'Fill BillFull' in billing_subtasks:
      billing_subtasks.remove('Fill BillFirst')
      billing_subtasks.remove('Fill BillLast')
      self.max_step -= 2
    if rng.random() < 0.4:
      billing_subtasks.remove('Fill BillCity')
      self.max_step -= 1
    if rng.random() < 0.4:
      billing_subtasks.remove('Fill BillState')
      self.max_step -= 1

    # Credit
    credit_pool = ['Fill C_First', 'Fill C_Last', 'Fill C_Phone']
    extra_credit = graph_utils.sample_subtasks(rng, pool=credit_pool,
                                                    minimum_size=0)
    credit_subtasks = FILL_CREDIT_SUBTASKS + extra_credit
    self.max_step += len(extra_credit)
    return shipping_subtasks, billing_subtasks, credit_subtasks

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
      base_subtasks = distractors + base_failures + ['Click ContinueBase']
      terminal_subtasks = base_failures.copy()
      self.max_step += len(base_subtasks) - len(BASE_SUBTASKS)
      self._base_distractors = distractors + base_failures
    else:
      base_subtasks = BASE_SUBTASKS
      terminal_subtasks = TERMINAL_SUBTASKS.copy()
      self._base_distractors= _BASE_DISTRACTORS

    # Add stage 0 (base) subtasks.
    g.add_base(base_subtasks)

    if not keep_pristine:
      # sample random subtasks
      shipping_subtasks, billing_subtasks, credit_subtasks = self._perturb_subtasks(rng=rng)
      contact_subtasks = CONTACT_SUBTASKS
    else:
      shipping_subtasks = SHIPPING_SUBTASKS
      billing_subtasks = BILLING_SUBTASKS
      contact_subtasks = CONTACT_SUBTASKS
      credit_subtasks = FILL_CREDIT_SUBTASKS

    # Connect stage 0 ==> stage 1 (contact & addresses) subtasks.
    with g.stage("Shipping Address"):
      g.add_one_to_many(
          source='Click ContinueBase',
          sinks=shipping_subtasks + contact_subtasks + SHIPMETHOD_SUBTASK
      )

      # Connect 'Click ShipMethod' ==> shipping method subtasks.
      g.add_one_to_many(
          source='Click ShipMethod',
          sinks=SELECT_SHIPMETHOD_SUBTASKS
      )

      # Connect Shipping & Contact ==> 'Click Apple GoPage2'
      g.add_many_to_one(sources=shipping_subtasks, sink='Click ContinueShipping')
      g.add_many_to_one(sources=contact_subtasks, sink='Click ContinueContact')

    # Connect 'Click Apple GoPage2' ==> stage 2 (payment) subtasks.
    g.add_many_to_many(
        sources=CONTINUE1_SUBTASKS,
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

    # Connect credit & billing subtasks ==> 'Click Apple GoPage3'
    g.add_many_to_one(sources=billing_subtasks, sink='Click ContinueBilling')
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

    # Connect 'Click Apple GoPage3' ==> final subtasks.
    g.add_many_to_many(
        sources=CONTINUE2_SUBTASKS,
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
        rnum = rng.random()
        if rnum < 0.4:
          g.remove_nodes(['Click Gift', 'Fill G_PIN', 'Fill G_NUM'])
        else:
          g.remove_nodes(['Fill G_PIN', 'Fill G_NUM'])
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

  def get_previous_subtasks(self, completion: Dict[str, bool]):
    subtasks = []
    if completion['Click ContinueShipping'] and completion['Click ContinueContact']:
      subtasks += self._base_distractors
    if completion['Click ContinuePayment'] and completion['Click ContinueBilling']:
      subtasks += ['Click Gift', 'Fill G_PIN', 'Fill G_NUM', 'Click PayPal']
    return subtasks

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
