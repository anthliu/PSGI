'''Configuration script for Dicks environment.'''
from typing import Dict
import numpy as np

from psgi.utils import env_utils, graph_utils
from psgi.envs import base_config as base
from psgi.envs.base_config import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


# Stage 0 (base)
BASE_SUBTASKS = [
    'Click SP',
    'Click Help',
    'Click Zip',
    'Click Items',
    'Click Dicks GoPage1'
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

FINISH1_SUBTASKS = [
    'Click FinishShipping',
    'Click FinishBilling',
    'Click FinishContact'
]

# TODO(CRITICAL): OR nodes can result in weird precondition (seed=406).
#SAME_ADDR_SUBTASKS = [
#    'Click Same as shipping',
#    'Unclick Same as shipping'
#]  # Billing & Shipping are same

CONTINUE_SUBTASK = ['Click Dicks GoPage2']  # Continue to payment

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

FINISH2_SUBTASKS = ['Click FinishPayment']

FILL_COUPON_SUBTASK = ['Fill Coupon']
APPLY_COUPON_SUBTASK = ['Click P_Apply']  # Apply coupon

FINAL_SUBTASKS = ['Click Place Order']

SUBTASK_LIST = BASE_SUBTASKS + CONTACT_SUBTASKS + BILLING_SUBTASKS + \
    SHIPPING_SUBTASKS + FINISH1_SUBTASKS + CONTINUE_SUBTASK + \
    PAYMENT_SUBTASKS + CREDIT_SUBTASKS + FILL_COUPON_SUBTASK + \
    APPLY_COUPON_SUBTASK + ['Click Dicks GoPage3'] + FINISH2_SUBTASKS + FINAL_SUBTASKS
LABEL_NAME = SUBTASK_LIST

TERMINAL_SUBTASKS = ['Click Help', 'Click SP', 'Click Place Order']


class Dicks(WobConfig):
  environment_id = 'dicks'

  def __init__(self, seed: int, keep_pristine: bool = False):
    super().__init__()
    self.num_graphs = 1
    self.max_step = 28
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

    g = SubtaskLogicGraph('Dicks')

    if not keep_pristine:
      # Sample base layer distractors & failure nodes.
      distractors = graph_utils.sample_subtasks(rng, pool=base.EXTRA_DISTRACTORS,
                                                minimum_size=0)
      base_failures = graph_utils.sample_subtasks(rng, pool=base.FAILURES_SUBSET1,
                                             minimum_size=0, maximum_size=3)
      base_subtasks = distractors + base_failures + ['Click Dicks GoPage1']
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
      credit_subtasks = CREDIT_SUBTASKS

    # Connect stage 0 ==> stage 1 (contact & addresses) subtasks.
    with g.stage("Address"):
      g.add_one_to_many(
          source='Click Dicks GoPage1',
          #sinks=CONTACT_SUBTASKS + billing_subtasks + SAME_ADDR_SUBTASKS
          sinks=CONTACT_SUBTASKS + billing_subtasks + shipping_subtasks
      )

      # Connect 'Unclick Same as shipping' ==> shipping address subtasks.
      #g.add_one_to_many(
      #    source='Unclick Same as shipping',
      #    sinks=shipping_subtasks
      #)

      # Sample preconditions from shipping & contact & billing subtasks.
      if not keep_pristine:
        contact_required = graph_utils.sample_subtasks(rng, pool=CONTACT_SUBTASKS,
                                                       minimum_size=1)
        billing_required = graph_utils.sample_subtasks(rng, pool=billing_subtasks,
                                                       minimum_size=1)
        shipping_required = graph_utils.sample_subtasks(rng, pool=shipping_subtasks,
                                                        minimum_size=1)
      else:  # pristine ver.
        contact_required = CONTACT_SUBTASKS
        billing_required = BILLING_REQUIRED
        shipping_required = SHIPPING_REQUIRED

      # Add 'Click Dicks GoPage2' subtask.
      g.add_many_to_one(sources=shipping_required, sink='Click FinishShipping')
      g.add_many_to_one(sources=billing_required, sink='Click FinishBilling')
      g.add_many_to_one(sources=contact_required, sink='Click FinishContact')
      g.add_many_to_one(sources=FINISH1_SUBTASKS, sink='Click Dicks GoPage2')

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
    g.add_one_to_many(
        source='Click Dicks GoPage2',
        sinks=PAYMENT_SUBTASKS
    )

    # Connect 'Click Credit' ==> credit info (+ coupon) subtasks.
    g.add_one_to_many(
        source='Click Credit',
        sinks=credit_subtasks + FILL_COUPON_SUBTASK
    )

    # Connect 'Fill Coupon Code' ==> 'Click Apply Code'
    g['Click P_Apply'] = g['Fill Coupon']

    # Sample preconditons from credit subtasks.
    if not keep_pristine:
      credit_required = graph_utils.sample_subtasks(rng, pool=credit_subtasks,
                                                    minimum_size=1)
    else:  # pristine ver.
      credit_required = credit_subtasks

    # Connect credit info subtasks ==> 'Click Dicks GoPage3'
    g.add_many_to_one(sources=credit_required, sink='Click FinishPayment')
    g['Click Dicks GoPage3'] = g['Click FinishPayment']

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

    # Final layer.
    g.add_one_to_many(
        source='Click Dicks GoPage3',
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
        g.remove_node('Click PayPal')
      if rng.random() < 0.5:
        g.remove_nodes(['Fill Coupon', 'Click P_Apply'])

    # Terminal subtasks
    self.terminal_subtasks = terminal_subtasks
    return g

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
