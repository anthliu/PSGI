import sys
import pytest
import numpy as np
from tqdm import tqdm

from acme import specs

import mockito
from mockito import spy2

import psgi
from acme.utils import paths
from psgi.utils import env_utils
from psgi.utils import log_utils

from psgi import envs
from psgi import environment_loop
from psgi.envs import wrappers
from psgi.utils.graph_utils import transform_obs
from psgi.graph.ilp import ILP
from psgi.agents.grprop import GRPropActor
from psgi.agents.eval_actor import EvalWrapper

## CONSTANTS for testing ##
ENV_NAME = 'mining'
SEED = 1
NUM_ADAPT_STEPS = 100
NUM_TASKS = 20
NUM_TEST_EPI = 4
GRAPH_PARAM = "eval"
ROOT='./psgi/testing/debug_data'
TEMP = 200.
## CONSTANTS for testing ##

@pytest.mark.xfail(reason="data needs to be updated (option_success)")
def test_ilp(spy2):
  # TODO(srsohn): Broken tests;
  # Due to changes in option_success, the reward data needs to be updated.

  # Prepare test phase environment.
  environment = env_utils.create_environment(
      env_id=ENV_NAME,
      batch_size=1,
      graph_param=GRAPH_PARAM,
      use_multi_processing=False,
      seed=SEED,
      gamma=.99
  )
  environment_spec = specs.make_environment_spec(environment)

  # ILP
  ilp = ILP(
    environment_spec=environment_spec,
    num_adapt_steps=NUM_ADAPT_STEPS,
    visualize=False,
    directory=None,
    environment_id=ENV_NAME
  )
  ilp.reset(environment)

  # GRProp
  actor = GRPropActor(
    environment_spec=environment_spec,
    temp=TEMP,
    w_a=3., beta_a=8.,
    ep_or=0.8, temp_or=2.
  )
  assert ENV_NAME == 'mining'
  actor = EvalWrapper(actor)

  # Load
  GT_graphs = np.load(f'{ROOT}/ilp_graph_{ENV_NAME}_{GRAPH_PARAM}_{SEED}_random.npy', allow_pickle=True)
  #GT_results = np.load(f'{ROOT}/test_result_{ENV_NAME}_{GRAPH_PARAM}_{SEED}_random.npy', allow_pickle=True)
  GT_logits, GT_masks, GT_states = np.load(f'{ROOT}/grprop_{ENV_NAME}_eval_{SEED}_random.npy', allow_pickle=True)

  for task_idx in tqdm(range(NUM_TASKS)):
    GT_graph = GT_graphs[task_idx]
    #
    GT_logit = GT_logits[task_idx]
    GT_mask = GT_masks[task_idx]
    GT_state = GT_states[task_idx]

    # Set task
    environment.reset_task(task_index=task_idx)
    ilp.reset(environment)

    # Load adaptation trajectory
    ilp.load(f'{ROOT}/ilp_traj_{ENV_NAME}_{GRAPH_PARAM}_{SEED}_random_{task_idx}.npy')

    # Run ilp & verify
    graph = ilp.infer_task()
    _verify_graph(graph[0], GT_graph)

    # Test GRProp with inferred graph
    actor.observe_task(graph)
    count = 0
    for gt_state, gt_logits in zip(GT_state, GT_logit):
      count += 1
      state_dict = dict(
        completion=gt_state[2].astype(np.float32),
        mask=gt_state[1].astype(np.float32),
        eligibility=gt_state[3].astype(np.float32),
      )
      logits = actor._actor._get_raw_logits_indexed_debug(state_dict)
      if not np.allclose(logits, gt_logits*TEMP, atol=1e-5):
        print(logits)
        print(gt_logits*TEMP)
        # NOTE: this might fail due to a bug in grprop (should be fixed in #31)
        np.testing.assert_allclose(logits, gt_logits, atol=1e-5)


def _verify_graph(graph, GT_graph):
  GT_ANDmat, GT_ORmat, GT_W_a, GT_W_o, GT_subtask_reward, GT_tind_by_layer, _, _, _ = GT_graph
  np.testing.assert_allclose(graph.subtask_reward, GT_subtask_reward, atol=1e-4, err_msg="Reward is wrong!!")
  for wa, gt_wa in zip(graph.W_a, GT_W_a):
    assert np.all(wa == gt_wa.astype(np.float64)), "W_a is wrong!!"
  for wo, gt_wo in zip(graph.W_o, GT_W_o):
    assert np.all(wo == gt_wo.astype(np.float64)), "W_o is wrong!!"
  assert np.all(graph.ANDmat == GT_ANDmat), "AND mat is wrong!"
  assert np.all(graph.ORmat == GT_ORmat), "OR mat is wrong!"
  assert np.all(graph.tind_by_layer == GT_tind_by_layer), "tind_by_layer is wrong!"

if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
