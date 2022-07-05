'''Configuration script for Dicks environment.'''
from typing import Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config_v2 as base
from psgi.envs.base_config_v2 import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


# Stage 0 (base)
BASE_SUBTASKS = [
    'Click SP',
    'Click Help',
    'Click Zip',
    'Click Items',
    'Click ContinueBase'
]
_BASE_DISTRACTORS = [
    'Click SP',
    'Click Help',
    'Click Zip',
    'Click Items',
]

# Stage 1 (contact & addresses)
CONTACT_SUBTASKS = [
    'Fill Email',
    'Fill Phone'
]

BILLING_REQUIRED = [
    'Fill BillFirst', 'Fill BillLast',
    'Fill BillStreet', 'Fill BillZip'
]
BILLING_SUBTASKS = BILLING_REQUIRED + ['Fill BillApt']

SHIPPING_REQUIRED = [
    'Fill First', 'Fill Last',
    'Fill Street', 'Fill Zip'
]
SHIPPING_SUBTASKS = SHIPPING_REQUIRED + ['Fill Apt']

CONTINUE1_SUBTASKS = [
    'Click ContinueShipping',
    'Click ContinueBilling',
    'Click ContinueContact'
]

# TODO(CRITICAL): OR nodes can result in weird precondition (seed=406).
#SAME_ADDR_SUBTASKS = [
#    'Click Same as shipping',
#    'Unclick Same as shipping'
#]  # Billing & Shipping are same

#CONTINUE_SUBTASK = ['Click Dicks GoPage2']  # Continue to payment

# Stage 2 (payments)
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

CONTINUE2_SUBTASKS = ['Click ContinuePayment']

FILL_COUPON_SUBTASK = ['Fill Coupon']
APPLY_COUPON_SUBTASK = ['Click P_Apply']  # Apply coupon

FINAL_SUBTASKS = ['Click Place Order']

SUBTASK_LIST = BASE_SUBTASKS + CONTACT_SUBTASKS + BILLING_SUBTASKS + \
    SHIPPING_SUBTASKS + CONTINUE1_SUBTASKS + \
    PAYMENT_SUBTASKS + CREDIT_SUBTASKS + FILL_COUPON_SUBTASK + \
    APPLY_COUPON_SUBTASK + CONTINUE2_SUBTASKS + FINAL_SUBTASKS
LABEL_NAME = SUBTASK_LIST

TERMINAL_SUBTASKS = ['Click Help', 'Click SP', 'Click Place Order']


class Dicks2(WobConfig):
  environment_id = 'dicks_v2'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.seed = seed if not keep_pristine else None
    self.max_step = 28  # default (pristine)
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
                     'Fill City', 'Fill State', 'Fill Country']
    extra_shipping = graph_utils.sample_subtasks(rng, pool=shipping_pool,
                                                    minimum_size=0)
    shipping_subtasks = SHIPPING_SUBTASKS + extra_shipping
    self.max_step += len(extra_shipping)
    if 'Fill Full' in shipping_subtasks:
      shipping_subtasks.remove('Fill First')
      shipping_subtasks.remove('Fill Last')
      self.max_step -= 2

    # Billing
    billing_pool = ['Fill BillFull', 'Fill BillAddress', 'Fill BillSecurityCode',
                     'Fill BillCity', 'Fill BillState', 'Fill BillCountry']
    extra_billing = graph_utils.sample_subtasks(rng, pool=billing_pool,
                                                    minimum_size=0)
    billing_subtasks = BILLING_SUBTASKS + extra_billing
    self.max_step += len(extra_billing)
    if 'Fill BillFull' in billing_subtasks:
      billing_subtasks.remove('Fill BillFirst')
      billing_subtasks.remove('Fill BillLast')
      self.max_step -= 2

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

    g = SubtaskLogicGraph('Dicks')

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
      credit_subtasks = CREDIT_SUBTASKS

    # Connect stage 0 ==> stage 1 (contact & addresses) subtasks.
    with g.stage("Address"):
      g.add_one_to_many(
          source='Click ContinueBase',
          sinks=contact_subtasks + billing_subtasks + shipping_subtasks
      )

      # Connect 'Unclick Same as shipping' ==> shipping address subtasks.
      #g.add_one_to_many(
      #    source='Unclick Same as shipping',
      #    sinks=shipping_subtasks
      #)

      # Add 'Click Dicks GoPage2' subtask.
      g.add_many_to_one(sources=shipping_subtasks, sink='Click ContinueShipping')
      g.add_many_to_one(sources=billing_subtasks, sink='Click ContinueBilling')
      g.add_many_to_one(sources=contact_subtasks, sink='Click ContinueContact')

      #preconditions = g[contact_required[0]]
      #for node in (contact_required + billing_required + shipping_required)[1:]:
      #  preconditions &= g[node]
      #g['Click Dicks GoPage2'] = preconditions

      # TODO: seed=406 causes a problem when it removes some OR relationships.
      # temporarily disable OR relationship.
      #shipping = g[shipping_required[0]]
      #for node in shipping_required[1:]:
      #  shipping &= g[node]
      #same_address = g['Click Same as shipping']
      #g['Click Step1'] = contact_and_billing & (same_address | shipping)

    # Connect 'Click Continue' ==> stage 2 (payment) subtasks.
    g.add_many_to_many(
        sources=CONTINUE1_SUBTASKS,
        sinks=PAYMENT_SUBTASKS + FILL_COUPON_SUBTASK
    )

    # Connect 'Fill Coupon Code' ==> 'Click Apply Code'
    g['Click P_Apply'] = g['Fill Coupon']

    # Connect 'Click Credit' ==> credit info (+ coupon) subtasks.
    g.add_one_to_many(
        source='Click Credit',
        sinks=credit_subtasks
    )

    # Connect credit info subtasks ==> 'Click Dicks GoPage3'
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
      g.add_reward('Click Help', -1.)
      g.add_reward('Click SP', -1.)

    # Apply graph perturbation if necessary.
    if not keep_pristine:
      # Skip some optional subtasks
      if rng.random() < 0.5:
        g.remove_node('Click Gift')
      if rng.random() < 0.5:
        g.remove_node('Click PayPal')
      if rng.random() < 0.5:
        g.remove_nodes(['Fill Coupon', 'Click P_Apply'])

    # Terminal subtasks
    self.terminal_subtasks = terminal_subtasks
    return g

  def get_previous_subtasks(self, completion: Dict[str, bool]):
    subtasks = []
    if completion['Click ContinueShipping'] and completion['Click ContinueBilling'] and \
        completion['Click ContinueContact']:
      subtasks += self._base_distractors
    if completion['Click ContinuePayment']:
      subtasks += ['Click Gift', 'Click PayPal', 'Fill Coupon', 'Click P_Apply']
    return subtasks

  def _construct_option_outcome(self):
    """Implement a special mechanism of completion dynamics.
      (i.e., one-step forward model)
    """
    option_extra_outcome = dict()

    # TODO: handle random perturbation.
    # Same address subtasks.
    #option_extra_outcome = env_utils.add_toggle_completion(
    #    toggle_outcomes=option_extra_outcome,
    #    subtasks=SAME_ADDR_SUBTASKS
    #)

    ## Select payment subtasks.
    #option_extra_outcome = env_utils.add_toggle_completion(
    #    toggle_outcomes=option_extra_outcome,
    #    subtasks=PAYMENT_SUBTASKS
    #)
    return option_extra_outcome
