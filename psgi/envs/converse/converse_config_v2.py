'''Configuration script for Converse environment.'''
from typing import Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config_v2 as base
from psgi.envs.base_config_v2 import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


# Stage 0 (base)
BASE_SUBTASKS = [
    'Click Privacy',
    'Click Zip',
    'Click ContinueBase'
]
_BASE_DISTRACTORS = [
    'Click Privacy',
    'Click Zip',
]
# Stage 1 (shipping address)
SHIPPING_REQUIRED = [
    'Fill First', 'Fill Last',
    'Fill Street', 'Fill Zip', 'Fill City', 'Fill State'
]
SHIPPING_OPTIONAL = ['Fill Apt']    # XXX optional
SHIPPING_SUBTASKS = SHIPPING_REQUIRED + SHIPPING_OPTIONAL

CLICK_SHIP_METHOD_SUBTASKS = ['Click ShipMethod']

SELECT_SHIP_METHOD_SUBTASKS = [
    'Click Standard',
    'Click Expedited',
    'Click NextDay'
]

CONTACT_SUBTASKS = [
    'Fill Phone',
    'Fill Email',
]

CONTINUE1_SUBTASKS = [
    'Click ContinueShipping',
    'Click ContinueContact'
]

# TODO: add this subtask
SAME_ADDR_SUBTASKS = []
#SAME_ADDR_SUBTASKS = [
#    'Click Same Address',
#    'Unclick Same Address'
#]  # Billing & Shipping are same

#CONTINUE_SUBTASK = ['Click Converse GoPage2']  # Continue to billing & payment

# Stage 2 (billing address & payment)
BILLING_REQUIRED = [
    'Fill BillFirst', 'Fill BillLast',
    'Fill BillStreet', 'Fill BillZip', 'Fill BillCity', 'Fill BillState',
]
BILLING_OPTIONAL = ['Fill BillApt']    # XXX optional
BILLING_SUBTASKS = BILLING_REQUIRED + BILLING_OPTIONAL

PAYMENT_SUBTASKS = [
    'Click Credit',
    'Click Gift',
    'Click PayPal'
]

CREDIT_SUBTASKS = [
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
    'Click ContinueBilling',
    'Click ContinuePayment'
]
#CONTINUE2_SUBTASK = ['Click Converse GoPage3']

# Stage 3 (review)
FINAL_SUBTASKS = [
    'Click Place Order',
    'Click EditShipping',
    'Click EditBilling',
    'Click ToU',
    'Click Feedback'
]

TERMINAL_SUBTASKS = ['Click ToU', 'Click Privacy', 'Click Feedback','Click Place Order']

SUBTASK_LIST = BASE_SUBTASKS + SHIPPING_SUBTASKS + \
    CLICK_SHIP_METHOD_SUBTASKS + SELECT_SHIP_METHOD_SUBTASKS + \
    SAME_ADDR_SUBTASKS + CONTINUE1_SUBTASKS + \
    BILLING_SUBTASKS + CONTACT_SUBTASKS + \
    PAYMENT_SUBTASKS + CREDIT_SUBTASKS + FILL_GIFT_SUBTASKS + \
    CONTINUE2_SUBTASKS + FINAL_SUBTASKS


class Converse2(WobConfig):
  environment_id = 'converse_v2'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.seed = seed if not keep_pristine else None
    self.max_step = 35
    self.graph = self._construct_task(seed=seed, keep_pristine=keep_pristine)
    if keep_pristine:
      self._assert_subtask_set(self.subtasks, SUBTASK_LIST)

    # Define subtask rewards.
    self.subtask_reward = {subtask.name: subtask.reward for subtask in self.graph.nodes}
    assert len(self.subtasks) == len(self.subtask_reward)
    # TODO: revive later with EXTRA_OPTIONS.
    #self.subtask_reward['NO_OP'] = 0.

    self._construct_mappings()

    # Define special completion mechanism
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
    return shipping_subtasks, billing_subtasks, credit_subtasks

  def _construct_task(self, seed: int, keep_pristine: bool = False):
    """Implement precondition & subtask reward & terminal subtasks
    """
    # Create graph with random perturbation.
    rng = np.random.RandomState(seed)  # pylint: disable=no-member

    g = SubtaskLogicGraph('Converse')

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
      self._base_distractors = _BASE_DISTRACTORS

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
      credit_subtasks = CREDIT_SUBTASKS

    # Connect stage 0 ==> stage 1 (shipping) subtasks.
    g.add_one_to_many(
        source='Click ContinueBase',
        sinks=shipping_subtasks + contact_subtasks + CLICK_SHIP_METHOD_SUBTASKS
    )

    # Connect 'Click ShipMethod' ==> shipping methods subtasks.
    g.add_one_to_many(
        source='Click ShipMethod',
        sinks=SELECT_SHIP_METHOD_SUBTASKS
    )

    # Add 'Click Converse GoPage2' subtask.
    g.add_many_to_one(sources=shipping_subtasks, sink='Click ContinueShipping')
    g.add_many_to_one(sources=contact_subtasks, sink='Click ContinueContact')

    # Connect stage 1 ==> stage 2 (billing & payment) subtasks.
    g.add_many_to_many(
        sources=CONTINUE1_SUBTASKS,
        sinks=billing_subtasks + PAYMENT_SUBTASKS
    )

    # Connect 'Click Credit' ==> credit info subtasks.
    g.add_one_to_many(
        source='Click Credit',
        sinks=credit_subtasks
    )

    # Connect 'Click Gift' ==> gift info subtasks.
    g.add_one_to_many(
        source='Click Gift',
        sinks=FILL_GIFT_SUBTASKS
    )

    # Add 'Click Converse GoPage3' subtask.
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

    # Connect stage 2 ==> stage 3 (review) subtasks.
    g.add_many_to_many(
        sources=CONTINUE2_SUBTASKS,
        sinks=final_subtasks
    )

    # Define subtask rewards
    g.add_reward('Click Place Order', 5.)
    if not keep_pristine:
      for failure in base_failures + final_failures:
        g.add_reward(failure, -rng.random())  # assign neg. reward
    else:
      g.add_reward('Click ToU', -1.)
      g.add_reward('Click Privacy', -1.)
      g.add_reward('Click Feedback', -1.)

    # Apply graph perturbation if necessary.
    if not keep_pristine:
      # Skip some optional subtasks
      if rng.random() < 0.5:
        g.remove_node('Click Standard')
      if rng.random() < 0.5:
        g.remove_node('Click Expedited')
      if rng.random() < 0.5:
        g.remove_node('Click NextDay')
      if rng.random() < 0.5:
        g.remove_nodes(['Click Gift', 'Fill G_NUM', 'Fill G_PIN'])
      if rng.random() < 0.5:
        g.remove_node('Click PayPal')

    # Terminal subtasks
    self.terminal_subtasks = terminal_subtasks
    return g

  def get_previous_subtasks(self, completion: Dict[str, bool]):
    subtasks = []
    if completion['Click ContinueShipping'] and  completion['Click ContinueContact']:
      subtasks += self._base_distractors
    if completion['Click ContinueBilling'] and completion['Click ContinuePayment']:
      subtasks += ['Click Gift', 'Click PayPal', 'Fill G_NUM', 'Click G_PIN']
    return subtasks

  def _construct_option_outcome(self):
    """Implement a special mechanism of completion dynamics
      (i.e., one-step forward model)
    """
    option_extra_outcome = dict()

    # TODO: handle random perturbation.
    # Select shipping method subtasks.
    #option_extra_outcome = env_utils.add_toggle_completion(
    #    option_extra_outcome,
    #    subtasks=SELECT_SHIP_METHOD_SUBTASKS
    #)

    ## Select payment subtasks.
    #option_extra_outcome = env_utils.add_toggle_completion(
    #    option_extra_outcome,
    #    subtasks=PAYMENT_SUBTASKS
    #)
    return option_extra_outcome
