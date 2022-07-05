'''Configuration script for ToyWoB environment.'''

from psgi.utils import env_utils
from psgi.envs.base_config import WobConfig
from psgi.envs.logic_graph import SubtaskLogicGraph


# Stage 0 (base)
SELECT_PAYMENT_SUBTASKS = [
    'Click Credit',
    'Click Debit',
    'Click PayPal'
]

BASE_SUBTASKS = [
    'Fill First',
    'Fill Last',
    'Fill Email',
    'Fill Street',
    'Fill Zip',
    'Fill Coupon'
] + SELECT_PAYMENT_SUBTASKS

# Stage 1 (payment infos)
FILL_CREDIT_SUBTASKS = [
    'Fill C_First',
    'Fill C_Last',
    'Fill C_CVV',
    'Fill C_NUM',
    'Fill C_EXPMM',
    'Fill C_EXPYY',
]

FILL_DEBIT_SUBTASKS = [
    'Fill D_First',
    'Fill D_Last',
    'Fill D_CVV',
    'Fill D_NUM',
    'Fill D_EXPMM',
    'Fill D_EXPYY',
]

FILL_PAYPAL_SUBTASKS = [
    'Fill PP_ID',
    'Fill PP_PW'
]

TERMINAL_SUBTASK = ['Click Place Order']

SUBTASK_LIST = BASE_SUBTASKS + FILL_CREDIT_SUBTASKS + FILL_DEBIT_SUBTASKS + \
    FILL_PAYPAL_SUBTASKS + TERMINAL_SUBTASK
LABEL_NAME = SUBTASK_LIST

PRISTINE = 0


class ToyWoB(WobConfig):
  def __init__(self, seed: int):
    super().__init__()
    self.num_graphs = 1
    self.max_step = 25
    self.graph = self._construct_task(seed=seed)
    if seed == PRISTINE:
      self._assert_subtask_set(self.subtasks, SUBTASK_LIST)

    # Define subtask rewards.
    self.subtask_reward = {subtask.name: subtask.reward for subtask in self.graph.nodes}
    assert len(self.subtasks) == len(self.subtask_reward)

    # Additional subtask reward for NO_OP.
    self.subtask_reward['NO_OP'] = 0.

    self._construct_mappings()

    # Define special completion mechanism
    self.option_extra_outcome = self._construct_option_outcome()

  def _construct_task(self, seed: int):
    """Implement precondition & subtask reward & terminal subtasks
    """
    g = SubtaskLogicGraph('Toywob')

    # Add stage 0 (base) subtasks.
    g.add_base(BASE_SUBTASKS)

    # Connect 'Click Credit' ==> credit info subtasks.
    g.add_one_to_many(
        source='Click Credit',
        sinks=FILL_CREDIT_SUBTASKS
    )

    # Connect 'Click Debit' ==> debit info subtasks.
    g.add_one_to_many(
        source='Click Debit',
        sinks=FILL_DEBIT_SUBTASKS
    )

    # Connect 'Click PayPal' ==> paypal info subtasks.
    g.add_one_to_many(
        source='Click PayPal',
        sinks=FILL_PAYPAL_SUBTASKS
    )

    # Add 'Click Place Order' subtask, which is eligible when:
    contact = g['Fill First'] & g['Fill Last'] & g['Fill Email']
    address = g['Fill Street'] & g['Fill Zip']
    credit = g['Fill C_First'] & g['Fill C_Last'] & g['Fill C_CVV'] & \
        g['Fill C_NUM'] & g['Fill C_EXPMM'] & g['Fill C_EXPYY']
    debit = g['Fill D_First'] & g['Fill D_Last'] & g['Fill D_CVV'] & \
        g['Fill D_NUM'] & g['Fill D_EXPMM'] & g['Fill D_EXPYY']
    paypal = g['Fill PP_ID'] & g['Fill PP_PW']
    g['Click Place Order'] = contact & address & (credit | debit | paypal)

    # Define subtask rewards
    g.add_reward('Click Place Order', 1.)

    # Terminal subtasks
    self.terminal_subtasks = TERMINAL_SUBTASK
    return g

  def _construct_option_outcome(self):
    """Implement a special mechanism of completion dynamics
      (i.e., one-step forward model)
    """
    option_extra_outcome = dict()

    # Toggle select payment subtasks.
    option_extra_outcome = env_utils.add_toggle_completion(
        toggle_outcomes=option_extra_outcome,
        subtasks=SELECT_PAYMENT_SUBTASKS
    )

    # TODO: Do we want to turn off all fill subtask completions when
    # different payment method is selected?
    return option_extra_outcome
