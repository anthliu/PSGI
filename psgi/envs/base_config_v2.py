'''Configuration script for Walmart environment.'''
import abc
import difflib
from typing import List, Dict, Any, Sequence
import numpy as np


# yapf: disable
BUTTONS = [
    #'Step0', 'Step1', 'Step2', 'Step3', 'Step4',  # replaced with GoPage
    'Place Order',
    'Promo', 'P_Apply', 'G_Apply', # Promo & gift
    'Shipping', 'Billing', 'Payment', 'ShipMethod',
]
CONTINUE = [
    'ContinueShipping', 'ContinueBilling', 'ContinueContact',
    'ContinuePayment', 'ContinueShipMethod', 'ContinueBase'
]

# distractors
EXTRAS = ['Items', 'Zip', 'Subscribe']
EDITS = ['EditShipMethod', 'EditShipping', 'EditBilling', 'EditPayment']
FAILURE = [
    'Feedback', 'Privacy', 'ToU', 'DNS PI', 'Request PI', 'SP', 'RP',
    'ReturnCart', 'Help'
]
BUTTONS += EXTRAS + EDITS + FAILURE + CONTINUE

# TODO: need to bring finite-difference GRPRop to make this work.
MULTIPLE_CHOICES = [
    'Credit', 'Debit', 'PayPal', 'Gift', 'Others',
    'Standard', 'Expedited', 'NextDay',  # shipping method
    'Delivery', 'Pick Up'
]

TOGGLES = [ # often leads to expansion/shinkage of webpage
    'Same as shipping',
    'Show Items',
    'G_No PIN',
]
CLICK_SUBTASKS = ['Click ' + elem for elem in BUTTONS + MULTIPLE_CHOICES + TOGGLES]
UNCLICK_SUBTASKS = ['Unclick ' + elem for elem in TOGGLES]
#######
#######
NAME_PROFILE = ['Full', 'BillFull']

SHIPPING_PROFILE = [  # Shipping
  'First', 'Last',
  'Address', 'Zip',
  'Apt', 'Street', 'City',
  'State', 'Country',
  'SecurityCode'
]
BILLING_PROFILE = [
  'BillFirst', 'BillLast',
  'BillAddress', 'BillZip',
  'BillApt', 'BillStreet', 'BillCity',
  'BillState', 'BillCountry',
  'BillSecurityCode'
]
CONTACT_PROFILE = ['Phone', 'Email']
BILL_CONTACT_PROFILE = ['BillPhone', 'BillEmail']
CREDIT_PROFILE = [
  'C_First', 'C_Last',
  'C_NUM', 'C_EXPMM', 'C_EXPYY', 'C_CVV',
  'C_Phone',
]
DEBIT_PROFILE = [
  'D_First', 'D_Last',
  'D_NUM', 'D_EXPMM', 'D_EXPYY', 'D_CVV',
  'D_Phone',
]
PAYPAL_PROFILE = [
    'PP_ID', 'PP_PW'
]
GIFT_PROFILE = ['G_NUM', 'G_PIN']
PROMO_PROFILE = ['P_NUM', 'P_PIN']
COUPON_PROFILE = ['Coupon']
PROFILES = NAME_PROFILE + SHIPPING_PROFILE + BILLING_PROFILE + CONTACT_PROFILE + \
    BILL_CONTACT_PROFILE + CREDIT_PROFILE + DEBIT_PROFILE + PAYPAL_PROFILE + \
    GIFT_PROFILE + PROMO_PROFILE + COUPON_PROFILE
###
SHIPPING_SUBTASKS = ['Fill ' + ele for ele in SHIPPING_PROFILE]
BILLING_SUBTASKS = ['Fill ' + ele for ele in BILLING_PROFILE]
CONTACT_SUBTASKS = ['Fill ' + elem for elem in CONTACT_PROFILE]
CREDIT_SUBTASKS = ['Fill ' + ele for ele in CREDIT_PROFILE]
FILL_SUBTASKS = ['Fill ' + elem for elem in PROFILES]
####
# TODO: temporarily comment out to match subtask == option
#EXTRA_OPTIONS = ['NO_OP', 'Hide Items']
EXTRA_OPTIONS = []

# List of subtask names
SUBTASK_NAMES = (CLICK_SUBTASKS + UNCLICK_SUBTASKS + FILL_SUBTASKS + EXTRA_OPTIONS)

# List of subtasks
SUBTASK_LIST = []
SUBTASK_POOL_NAME_TO_ID = dict()
SUBTASK_POOL_ID_TO_NAME = []
for idx, subtask_name in enumerate(SUBTASK_NAMES):
  SUBTASK_LIST.append(dict(name=subtask_name, id=idx))
  SUBTASK_POOL_NAME_TO_ID[subtask_name] = idx
  SUBTASK_POOL_ID_TO_NAME.append(subtask_name)

# TODO: Action space can be different from the subtask pool.
# For now, we assume that they are the same (see TODO comments).
#OPTION_NAMES = SUBTASK_NAMES + EXTRA_OPTIONS
OPTION_NAMES = SUBTASK_NAMES
OPTION_NAME_TO_ID = dict()
OPTION_ID_TO_NAME = []
for idx, option_name in enumerate(OPTION_NAMES):
  OPTION_NAME_TO_ID[option_name] = idx
  OPTION_ID_TO_NAME.append(option_name)

# Distractor collections.
DISTRACTORS = ['Click ' + ele for ele in EXTRAS + EDITS]
#FAILURE_SUBTASKS = ['Click ' + ele for ele in FAILURE]
EXTRA_DISTRACTORS = ['Click ' + ele for ele in EXTRAS]  # smaller subset
EDIT_DISTRACTORS = ['Click ' + ele for ele in EDITS]  # smaller subset
FAILURES_SUBSET1 = ['Click ' + ele for ele in ['SP', 'Privacy', 'RP', 'Help']]  # smaller subset
FAILURES_SUBSET2 = ['Click ' + ele for ele in ['ReturnCart', 'Feedback', 'ToU', 'Request PI', 'DNS PI']]  # smaller subset

LABEL_NAME = {subtask['id']: subtask['name'] for subtask in SUBTASK_LIST}
###
# yapf: enable

def check_type(name, typeset):
  for prefix in typeset:
    if prefix in name:
      return True
  return False


class WobConfig(abc.ABC):
  def __init__(self):
    self.subtask_pool_name_to_id = SUBTASK_POOL_NAME_TO_ID

  @abc.abstractmethod
  def _construct_task(self):
    """Construct task."""

  @abc.abstractmethod
  def _construct_option_outcome(self):
    """Define extra option mechanisms."""

  def _construct_mappings(self):
    # Subtask pool index <-> task-specific index
    self._pool_to_index = np.ones(len(SUBTASK_POOL_NAME_TO_ID), dtype=np.int32) * -1
    self._index_to_pool = np.zeros(len(self.subtasks), dtype=np.int32)
    for tind, name in enumerate(self.subtasks):
      assert isinstance(name, str), str(type(name))
      tid = SUBTASK_POOL_NAME_TO_ID[name]
      self._pool_to_index[tid] = tind
      self._index_to_pool[tind] = tid

  def _assert_subtask_set(self, actual, expected):
    actual = list(sorted(actual))
    expected = list(sorted(expected))

    assert actual == expected, (
      'Subtasks from the graph do not match: \n' +
      '\n'.join(difflib.unified_diff(actual, expected)) +
      '\nEND OF DIFF.'
    )

  # Subclasses must implement these properties.
  # TODO: abstract property
  graph: 'SubtaskLogicGraph'
  subtask_reward: Dict[str, float]
  max_step: int

  @property
  def subtasks(self) -> Sequence[str]:
    return tuple(self.graph._nodes.keys())

  @property
  def num_subtask(self):
    return len(self.subtasks)

  def __repr__(self):
    return "<WobConfig[%s, %d subtasks, graph=%s>" % (
      type(self).__name__, len(self.subtasks or []), self.graph
    )
