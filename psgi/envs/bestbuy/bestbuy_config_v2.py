'''Configuration script for BestBuy environment.'''
from typing import Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config_v2 as base
from psgi.envs.base_config_v2 import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


# Stage 0 (base)
BASE_SUBTASKS = [
    'Click Subscribe',
    'Click Items',
    'Click Help',
    'Click Privacy',
    'Click ContinueBase'
]
_BASE_DISTRACTORS = [
    'Click Subscribe',
    'Click Items',
    'Click Help',
    'Click Privacy',
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
    'Click ContinueShipMethod',
    'Click ContinueContact'
]
#CONTINUE_SUBTASK = ['Click BestBuy GoPage2']  # Continue to payment

# Stage 2 (payments)
BILLING_REQUIRED = [
    'Fill BillFirst', 'Fill BillLast',
    'Fill BillStreet', 'Fill BillState', 'Fill BillCity', 'Fill BillZip'
]
BILLING_OPTIONAL = ['Fill BillApt']
BILLING_SUBTASKS = BILLING_REQUIRED + BILLING_OPTIONAL

PAYMENT_SUBTASKS = [
    'Click Credit',
    'Click Gift',
    'Click Others'
]

CREDIT_SUBTASKS = [
    'Fill C_NUM',
    'Fill C_EXPMM',
    'Fill C_EXPYY',
    'Fill C_CVV'
]
GIFT_SUBTASKS = [
    'Fill G_NUM',
    'Fill G_PIN'
]

CONTINUE2_SUBTASKS = [
    'Click ContinuePayment',
    'Click ContinueBilling'
]

OTHER_PAYMENTS_SUBTASKS = ['Click PayPal']
FILL_COUPON_SUBTASK = ['Fill Coupon']
APPLY_COUPON_SUBTASK = ['Click P_Apply']  # Apply coupon
FINAL_SUBTASKS = [
    'Click Place Order',
    'Click DNS PI'
]

SUBTASK_LIST = BASE_SUBTASKS + SHIPPING_SUBTASKS + CONTACT_SUBTASKS + \
    SHIPMETHOD_SUBTASK + SELECT_SHIPMETHOD_SUBTASKS + CONTINUE1_SUBTASKS + \
    BILLING_SUBTASKS + PAYMENT_SUBTASKS + CREDIT_SUBTASKS + \
    GIFT_SUBTASKS + OTHER_PAYMENTS_SUBTASKS + FILL_COUPON_SUBTASK + \
    APPLY_COUPON_SUBTASK + CONTINUE2_SUBTASKS + FINAL_SUBTASKS
LABEL_NAME = SUBTASK_LIST


# Additional Collections.
FAILURE_SUBTASKS = ['Click Privacy', 'Click DNS PI']
TERMINAL_SUBTASKS = FAILURE_SUBTASKS + ['Click Place Order']


class BestBuy2(WobConfig):
  environment_id = 'bestbuy_v2'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.seed = seed if not keep_pristine else None
    self.max_step = 40
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
    credit_subtasks = CREDIT_SUBTASKS + extra_credit
    self.max_step += len(extra_credit)

    # ShipMethod
    shipmethod_subtasks = graph_utils.sample_subtasks(rng, pool=SELECT_SHIPMETHOD_SUBTASKS,
                                                    minimum_size=1, maximum_size=1)
    return shipping_subtasks, billing_subtasks, credit_subtasks, shipmethod_subtasks

  def _construct_task(self, seed: int, keep_pristine: bool = False):
    """Implement precondition & subtask reward & terminal subtasks
    """
    # Create graph with random perturbation.
    rng = np.random.RandomState(seed)  # pylint: disable=no-member

    g = SubtaskLogicGraph('BestBuy')

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
      shipping_subtasks, billing_subtasks, credit_subtasks, shipmethod_subtasks = \
          self._perturb_subtasks(rng=rng)
      contact_subtasks = CONTACT_SUBTASKS
    else:
      shipping_subtasks = SHIPPING_SUBTASKS
      billing_subtasks = BILLING_SUBTASKS
      contact_subtasks = CONTACT_SUBTASKS
      credit_subtasks = CREDIT_SUBTASKS
      shipmethod_subtasks = ['Click Standard']

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

      # Connect Shipping & Contact & ShipMethod ==> 'Click BestBuy GoPage2'
      g.add_many_to_one(sources=shipping_subtasks, sink='Click ContinueShipping')
      g.add_many_to_one(sources=contact_subtasks, sink='Click ContinueContact')
      g.add_many_to_one(sources=shipmethod_subtasks, sink='Click ContinueShipMethod')

    # Connect 'Click BestBuy GoPage2' ==> stage 2 (payment) subtasks.
    g.add_many_to_many(
        sources=CONTINUE1_SUBTASKS,
        sinks=PAYMENT_SUBTASKS + billing_subtasks + FILL_COUPON_SUBTASK
    )

    # Connect 'Click Credit' ==> credit info subtasks.
    g.add_one_to_many(
        source='Click Credit',
        sinks=credit_subtasks
    )

    # Connect 'Click Others' ==> Click PayPal subtask.
    g['Click PayPal'] = g['Click Others']

    # Connect 'Click Gift' ==> gift card info subtasks.
    g.add_one_to_many(
        source='Click Gift',
        sinks=GIFT_SUBTASKS
    )

    # Connect 'Fill Coupon Code' ==> 'Click Apply Code'
    g['Click P_Apply'] = g['Fill Coupon']

    # Connect billing/credit info subtasks ==> 'Click BestBuy GoPage3'
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

    # Final layer.
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
      g.add_reward('Click Help', -1)
      g.add_reward('Click Privacy', -1)
      g.add_reward('Click DNS PI', -1)

    if not keep_pristine:
      # Skip some optional subtasks
      if rng.random() < 0.3:
        rnum = rng.random()
        if rnum < 0.4:
          g.remove_nodes(['Click Gift', 'Fill G_PIN', 'Fill G_NUM'])
        else:
          g.remove_nodes(['Fill G_PIN', 'Fill G_NUM'])
      if rng.random() < 0.3:
        rnum = rng.random()
        if rnum < 0.4:
          g.remove_nodes(['Click Others', 'Click PayPal'])
        else:
          g.remove_node('Click PayPal')
      if rng.random() < 0.5:
        g.remove_nodes(['Fill Coupon', 'Click P_Apply'])

    # Terminal subtasks
    self.terminal_subtasks = terminal_subtasks
    return g

  def get_previous_subtasks(self, completion: Dict[str, bool]):
    subtasks = []
    if completion['Click ContinueShipping'] and completion['Click ContinueContact']:
      subtasks += self._base_distractors
    if completion['Click ContinuePayment'] and completion['Click ContinueBilling']:
      subtasks += ['Click Gift', 'Fill G_PIN', 'Fill G_NUM', 'Click Others',
                   'Click PayPal', 'Fill Coupon', 'Click P_Apply']
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
    #    subtasks=PAYMENT_SUBTASKS
    #)
    return option_extra_outcome
