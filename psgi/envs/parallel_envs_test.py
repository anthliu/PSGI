from typing import Optional
import collections
import sys

import dm_env
from acme import specs
import tensorflow as tf

from psgi import agents
from psgi.system import multiprocessing
from psgi.utils import env_utils

EnvParameter = collections.namedtuple(
    'EnvParameter', ['env_id', 'graph_param', 'num_envs', 'action_dim'])

TEST_PARAMS = (
    EnvParameter('playground', 'D1_train', 4, 16),
    EnvParameter('mining', 'train', 4, 26),
)


class ParallelEnvironmentsTest(tf.test.TestCase):

  def _make_parallel_environments(
      self,
      env_id: str,
      num_envs: int = 1,
      graph_param: Optional[str] = None,
      num_adapt_steps: Optional[int] = None,
      add_fewshot_wrapper: bool = False
  ):
    return env_utils.create_environment(
        env_id=env_id,
        batch_size=num_envs,
        graph_param=graph_param,
        num_adapt_steps=num_adapt_steps,
        add_fewshot_wrapper=add_fewshot_wrapper,
        seed=1
    )

  def test_specs(self):
    for param in TEST_PARAMS:
      env = self._make_parallel_environments(
          env_id=param.env_id,
          num_envs=param.num_envs,
          graph_param=param.graph_param
      )

      env_spec = specs.make_environment_spec(env)
      obs_spec = env_spec.observations

      self.assertEqual(obs_spec['mask'].shape, (param.action_dim,))
      self.assertEqual(obs_spec['completion'].shape, (param.action_dim,))
      self.assertEqual(obs_spec['eligibility'].shape, (param.action_dim,))

  def test_reset(self):
    for param in TEST_PARAMS:
      env = self._make_parallel_environments(
          env_id=param.env_id,
          num_envs=param.num_envs,
          graph_param=param.graph_param
      )

      env.reset_task(task_index=0)
      ts = env.reset()

      self.assertIsInstance(ts, dm_env.TimeStep)
      self.assertTrue(all(ts.first()))
      self.assertFalse(any(ts.mid()))
      self.assertFalse(any(ts.last()))

  def test_step(self):
    for param in TEST_PARAMS:
      env = self._make_parallel_environments(
          env_id=param.env_id,
          num_envs=param.num_envs,
          graph_param=param.graph_param
      )

      env_spec = specs.make_environment_spec(env)

      # Create random actor.
      actor = agents.RandomActor(env_spec)

      env.reset_task(task_index=0)
      ts = env.reset()

      # Step.
      action = actor.select_action(ts.observation)
      ts = env.step(action)

      self.assertIsInstance(ts, dm_env.TimeStep)
      self.assertFalse(any(ts.first()))

      ob = ts.observation
      self.assertEqual(ob['mask'].shape, (param.num_envs, param.action_dim))
      self.assertEqual(ob['completion'].shape, (param.num_envs, param.action_dim))
      self.assertEqual(ob['eligibility'].shape, (param.num_envs, param.action_dim))

if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
