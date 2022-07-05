import random
import sys
import pytest

from psgi.envs.sge import mazeenv
from psgi.envs.wrappers import maze_wrapper
from psgi.envs import sge


class TestMazeEnv:

  def _run_random_episode(self, env, verbose=False):
    # Action spaces are enum (for raw env).
    action_set = env.legal_actions
    print("action_set = ", action_set)

    # interaction with a random agent
    print("Running a random episode...")
    step, done = 0, False
    t = 0
    R = 0
    while not done:
      action = random.choice(list(action_set))
      try:
        s, r, done, info = env.step(action)
      except mazeenv.IneligibleSubtaskError:
        print('Ineligible action performed : ', action)
        break
      if verbose:
        print('Step={:02d}, Action={}, Reward={:.2f}, Done={}'.format(
          t, action, r, done))
      t += 1
      R += r
      if t > 200:
        raise RuntimeError("Episode did not finish...")
    print(f"Episode terminated in {t} steps, Return = {R}")

  def _examine_graph(self, env: mazeenv.MazeEnv):
      """Print brief information about the subtask graph in the env."""
      print(env.graph)
      print(" - tind_by_layer: ", env.graph.tind_by_layer)

  # -------------------------------------------------------------------------

  @pytest.mark.parametrize("game_name, graph_param", [
      ("mining", "train"),
      ("playground", "D1_train"),
  ])
  def test_maze_env_loaded(self, game_name: str, graph_param: str):
    """Tests raw-action mining environment with a predefined subtask config."""
    print("[%s]" % game_name)
    env = mazeenv.MazeEnv(
      game_name=game_name,
      graph_param=graph_param,  # scenario (Difficulty)
    )
    assert game_name.capitalize() in str(env.config)

    s, info = env.reset()
    assert 'observation' in s
    assert 'mask' in s
    assert 'completion' in s
    assert 'eligibility' in s
    #assert 'step' in s
    #assert 'graph' in info  # SubtaskGraph

    # Validate action space and environment space

    # Validate masks for non-active subtasks

    self._examine_graph(env)
    self._run_random_episode(env)


  def test_mining_generate(self):
    env = mazeenv.MazeEnv(game_name='mining', graph_param='train',
                          generate_graphs=True)

    # TODO: Remove info from reset.
    s, info = env.reset()
    print(s)

    assert 'observation' in s
    assert 'mask' in s
    assert 'completion' in s
    assert 'eligibility' in s

    # Access internal data: maze config and SubtaskGraph
    print(env.config)
    assert 'Mining' in str(env.config)

    self._examine_graph(env)
    assert isinstance(env.graph, sge.graph.SubtaskGraphGenerator)

    self._run_random_episode(env)


  def test_maze_env_parameterized(self):
    env = mazeenv.MazeEnv(game_name='playground',
                          generate_graphs=True, graph_param=None)   # generation mode
    self._examine_graph(env)

    # Note that in the psgi version, we allow the environment to be used
    # without an explicit call of reset_graph (as opposed to sge-light one)
    # as we want to have a consistent action and observation space.
    #with pytest.raises(RuntimeError):
    #  env.reset()

    # override pool_size and subtasks.
    env.reset_task(
      subtask_pool_size=14, subtasks=12, num_layers=4,
      subtask_design=[])
    self._examine_graph(env)
    assert env.graph.max_task == 14
    assert env.max_task == env.graph.max_task
    assert env.graph.ntasks == 12
    assert env.graph.nlayer == 4
    # TODO: validate per-layer distractors and subtasks.
    env.reset()
    self._run_random_episode(env)

    # subtasks: As a subset.
    env.reset_task(
      subtask_pool_size=14, subtasks=[0, 2, 4, 6, 8, 10],
      num_layers=2,
      subtask_design=[])
    self._examine_graph(env)
    assert env.graph.max_task == 14
    assert env.max_task == env.graph.max_task
    assert env.graph.ntasks == 6
    assert env.graph.nlayer == 2
    # TODO: validate per-layer distractors and subtasks.
    env.reset()
    self._run_random_episode(env)

    # subtasks: design
    design = [
      {'id': 0, 'layer': 0, 'distractor': 0},
      {'id': 1, 'layer': 1, 'distractor': 1},
      {'id': 2, 'layer': 2, 'distractor': 0},
    ]
    env.reset_task(subtask_pool_size=14, subtasks=14, num_layers=3,
                   subtask_design=design)
    env.reset()


  def test_maze_optionenv_parameterized(self):
    env = mazeenv.MazeOptionEnv(game_name='playground',
                                generate_graphs=True, graph_param=None)   # generation mode
    self._examine_graph(env)
#    with pytest.raises(RuntimeError):
#      env.reset()   # never called task_reset, not ready

    # override pool_size and subtasks.
    env.reset_task(subtask_pool_size=14, subtasks=12, num_layers=4,
                   subtask_design=[])
    self._examine_graph(env)
    assert env.graph.max_task == 14
    assert len(env.legal_actions) == 12
    assert env.max_task == env.graph.max_task
    assert env.graph.ntasks == 12
    assert env.graph.nlayer == 4
    # TODO: validate per-layer distractors and subtasks.
    env.reset()
    self._run_random_episode(env)

    # subtasks: As a subset.
    env.reset_task(subtask_pool_size=14, subtasks=[0, 2, 4, 6, 8, 10],
                   num_layers=2,
                   subtask_design=[])
    self._examine_graph(env)
    assert env.graph.max_task == 14
    assert list(env.legal_actions) == [0, 2, 4, 6, 8, 10]
    assert env.max_task == env.graph.max_task
    assert env.graph.ntasks == 6
    assert env.graph.nlayer == 2
    # TODO: validate per-layer distractors and subtasks.
    env.reset()
    self._run_random_episode(env)

  def test_maze_optionenv_parameterized_design(self):
    env = mazeenv.MazeOptionEnv(game_name='playground',
                                generate_graphs=True, graph_param=None)   # generation mode

    # subtasks: custom design
    design = [
      {'id': 0, 'layer': 0, 'distractor': 0},
      {'id': 1, 'layer': 1, 'distractor': 1},
      {'id': 2, 'layer': 2, 'distractor': 0},
    ]
    env.reset_task(subtask_pool_size=14, subtasks=14, num_layers=3,
                   subtask_design=design)
    env.reset()
    self._examine_graph(env)

    # TODO: validate id and layer assignment.

  def test_maze_optionenv_design_hard(self):
    env = mazeenv.MazeOptionEnv(game_name='playground',
                                generate_graphs=True, graph_param=None)   # generation mode

    def _get_args(design):
      subtask_pool_size = max(t['id'] for t in design) + 1
      num_layers = max(t['layer'] for t in design) + 1
      subtasks = [t['id'] for t in design]
      return dict(subtask_pool_size=subtask_pool_size,
                  num_layers=num_layers, subtasks=subtasks,
                  subtask_design=design)

    # Everything is in the flat layer (layer=0)
    design = [{'id': 1, 'layer': 0, 'distractor': 0}, {'id': 3, 'layer': 0, 'distractor': 0}, {'id': 6, 'layer': 0, 'distractor': 1}, {'id': 5, 'layer': 0, 'distractor': 1}, {'id': 13, 'layer': 0, 'distractor': 1}, {'id': 4, 'layer': 0, 'distractor': 1}, {'id': 11, 'layer': 0, 'distractor': 1}, {'id': 0, 'layer': 0, 'distractor': 0}, {'id': 2, 'layer': 0, 'distractor': 0}, {'id': 7, 'layer': 0, 'distractor': 0}, {'id': 8, 'layer': 0, 'distractor': 0}, {'id': 12, 'layer': 0, 'distractor': 0}, {'id': 9, 'layer': 0, 'distractor': 1}, {'id': 10, 'layer': 0, 'distractor': 0}]
    env.reset_task(**_get_args(design))
    env.reset()

    # Some random cases
    design = [{'id': 3, 'layer': 1, 'distractor': 0}, {'id': 11, 'layer': 1, 'distractor': 1}, {'id': 13, 'layer': 0, 'distractor': 0}, {'id': 0, 'layer': 0, 'distractor': 0}, {'id': 6, 'layer': 0, 'distractor': 0}, {'id': 1, 'layer': 1, 'distractor': 1}, {'id': 5, 'layer': 2, 'distractor': 0}, {'id': 4, 'layer': 1, 'distractor': 0}, {'id': 12, 'layer': 1, 'distractor': 0}, {'id': 2, 'layer': 1, 'distractor': 1}, {'id': 8, 'layer': 1, 'distractor': 1}, {'id': 10, 'layer': 0, 'distractor': 1}, {'id': 9, 'layer': 1, 'distractor': 0}, {'id': 7, 'layer': 1, 'distractor': 1}]
    env.reset_task(**_get_args(design))
    env.reset()

    # action=1 throws an ineligiblesubtask error
    design = [{'id': 3, 'layer': 0, 'distractor': 0}, {'id': 4, 'layer': 2, 'distractor': 0}, {'id': 10, 'layer': 1, 'distractor': 0}, {'id': 12, 'layer': 0, 'distractor': 1}, {'id': 13, 'layer': 0, 'distractor': 1}, {'id': 1, 'layer': 2, 'distractor': 1}, {'id': 0, 'layer': 1, 'distractor': 1}, {'id': 6, 'layer': 2, 'distractor': 0}, {'id': 7, 'layer': 2, 'distractor': 1}, {'id': 11, 'layer': 2, 'distractor': 0}, {'id': 9, 'layer': 2, 'distractor': 0}, {'id': 8, 'layer': 1, 'distractor': 0}, {'id': 2, 'layer': 1, 'distractor': 1}]
    s = env.reset_task(**_get_args(design))
    env.reset()

    # TODO: validate masks and generated environment


  def test_maze_wrapper(self):
    env = mazeenv.MazeOptionEnv(game_name='playground',
                                generate_graphs=True, graph_param=None)
    environment = maze_wrapper.MazeWrapper(env)

    # TODO: Validate these spec values.
    print(f"environment.observation_spec = {environment.observation_spec()}")
    print(f"environment.action_spec = {environment.action_spec()}")
    print(f"environment.discount_spec() = {environment.discount_spec()}")


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
