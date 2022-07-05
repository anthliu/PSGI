from typing import Optional, Dict, List

import tree
import numpy as np
import functools
from acme import types

from psgi import envs
from psgi.envs import wrappers
from psgi.envs import parallel_envs
#from psgi.envs import toythor

def sample_web_configs(eval_env: str, seed: int, size: int) -> List[str]:
  # Randomly choose <size> number of websites excluding the one given.
  configs_dict = envs.web_configs_v2_dict if 'v2' in eval_env else envs.web_configs_dict
  candidates = [k for k, v in configs_dict.items() if k != eval_env]
  rng = np.random.RandomState(seed=seed)
  configs = rng.choice(candidates, size=size, replace=False)
  return list(configs)

def create_environment(
    env_id: str,
    graph_param: str,
    seed: int,
    batch_size: int = 1,
    feature_mode: str = 'cluster-s',
    num_adapt_steps: Optional[int] = None,
    add_fewshot_wrapper: bool = False,
    use_multi_processing: bool = True,
    gamma: float = 0.99,
    verbose_level: Optional[int] = 0,
):
  # Create an environment.
  env_args = dict(verbose_level=verbose_level)
  keep_pristine = False if graph_param == 'train' else True
  #keep_pristine = True # TODO: Remove

  environment_cls = envs.BaseWoBEnv
  adapter_cls = wrappers.WoBWrapper
  if env_id == 'walmart':
    env_args.update(
        config_factory=[envs.Walmart],
        keep_pristine=keep_pristine
    )
  elif env_id == 'dicks':
    env_args.update(
      config_factory=[envs.Dicks],
      keep_pristine=keep_pristine
    )
  elif env_id == 'converse':
    env_args.update(
      config_factory=[envs.Converse],
      keep_pristine=keep_pristine
    )
  elif env_id == 'bestbuy':
    env_args.update(
      config_factory=[envs.BestBuy],
      keep_pristine=keep_pristine
    )
  elif env_id == 'apple':
    env_args.update(
      config_factory=[envs.Apple],
      keep_pristine=keep_pristine
    )
  elif env_id == 'amazon':
    env_args.update(
      config_factory=[envs.Amazon],
      keep_pristine=keep_pristine
    )
  elif env_id == 'ebay':
    env_args.update(
      config_factory=[envs.Ebay],
      keep_pristine=keep_pristine
    )
  elif env_id == 'samsung':
    env_args.update(
      config_factory=[envs.Samsung],
      keep_pristine=keep_pristine
    )
  elif env_id == 'ikea':
    env_args.update(
      config_factory=[envs.IKEA],
      keep_pristine=keep_pristine
    )
  elif env_id == 'target':
    env_args.update(
      config_factory=[envs.Target],
      keep_pristine=keep_pristine
    )
  elif env_id == 'walmart_v2':
    env_args.update(
        config_factory=[envs.Walmart2],
        keep_pristine=keep_pristine
    )
  elif env_id == 'dicks_v2':
    env_args.update(
      config_factory=[envs.Dicks2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'converse_v2':
    env_args.update(
      config_factory=[envs.Converse2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'bestbuy_v2':
    env_args.update(
      config_factory=[envs.BestBuy2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'apple_v2':
    env_args.update(
      config_factory=[envs.Apple2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'amazon_v2':
    env_args.update(
      config_factory=[envs.Amazon2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'ebay_v2':
    env_args.update(
      config_factory=[envs.Ebay2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'samsung_v2':
    env_args.update(
      config_factory=[envs.Samsung2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'ikea_v2':
    env_args.update(
      config_factory=[envs.IKEA2],
      keep_pristine=keep_pristine
    )
  elif env_id == 'target_v2':
    env_args.update(
      config_factory=[envs.Target2],
      keep_pristine=keep_pristine
    )

  elif env_id == 'dummy':  # TODO: Remove
    env_args.update(
      config_factory=[envs.Dummy],
      keep_pristine=keep_pristine
    )
  elif env_id == 'toywalmart':  # TODO: Remove
    env_args.update(
      config_factory=[envs.ToyWalmart],
      keep_pristine=keep_pristine
    )
  elif env_id == 'toydummy':  # TODO: Remove
    env_args.update(
      config_factory=[envs.ToyDummy],
      keep_pristine=keep_pristine
    )
  elif env_id == 'webnav':
    assert graph_param in {'train', 'eval'}, \
        f'Unrecognized graph_param argument "{graph_param}".'
    assert seed <= 10, f'Unrecognized seed arguement "{seed}".'
    env_args.update(
        config_factory=envs.WEBNAV_TASKS[f'{graph_param}_{seed}'],
        keep_pristine=keep_pristine
    )
  elif env_id in {'playground', 'mining'}:
    environment_cls = envs.MazeOptionEnv
    env_args.update(
        game_name=env_id,
        graph_param=graph_param,
        seed=seed,
        gamma=gamma,
    )
    adapter_cls = wrappers.MazeWrapper
  elif env_id == 'toythor':
    environment_cls = toythor.make_env
    env_args.update(
        env_name='toythor',
        eval=graph_param == 'eval',
        seed=seed
    )
  elif env_id == 'cooking':
    environment_cls = envs.BasePredicateEnv
    env_args.update(
        config_factory=[envs.Cooking],
        feature_mode=feature_mode,
        graph_param=graph_param,
        keep_pristine=True,
    )
  elif env_id == 'cookclassic':
    environment_cls = envs.BasePredicateEnv
    env_args.update(
        config_factory=[envs.CookingClassic],
        feature_mode=feature_mode,
        graph_param=graph_param,
        keep_pristine=True,
    )
  elif env_id == 'pickplace':
    environment_cls = envs.BasePredicateEnv
    env_args.update(
        config_factory=[envs.PickPlace],
        feature_mode=feature_mode,
        graph_param=graph_param,
        keep_pristine=True,
    )
  elif env_id == 'ETmining':
    environment_cls = envs.BasePredicateEnv
    env_args.update(
        config_factory=[envs.ETMining],
        feature_mode=feature_mode,
        graph_param=graph_param,
        keep_pristine=True,
    )
  elif env_id == 'ai2thor':
    environment_cls = envs.AI2ThorEnv
    env_args.update(
        config_factory=[envs.AI2ThorConfig],
        feature_mode=feature_mode,
        graph_param=graph_param,
        keep_pristine=True,
    )
  else:
    raise NotImplementedError

  if use_multi_processing and batch_size > 1:
    batch_wrapper = parallel_envs.ParallelEnvironments
  else:
    batch_wrapper = wrappers.SerialEnvironments

  if add_fewshot_wrapper:
    assert num_adapt_steps is not None, \
        'Maximum number of adaptation steps must be specified.'
    return batch_wrapper([
        functools.partial(
        lambda rank: wrappers.FewshotWrapper(
            adapter_cls(environment_cls(**{**env_args, 'rank': rank})),
            max_steps=num_adapt_steps),
            rank=rank) for rank in range(batch_size)
    ])

  return batch_wrapper([
      functools.partial(
      lambda rank: adapter_cls(environment_cls(**{**env_args, 'rank': rank})),
          rank=rank) for rank in range(batch_size)
  ])

def add_batch_dim(nest: types.NestedArray) -> types.NestedArray:
  """Adds a batch dimension to each leaf of a nested structure of Array."""
  return tree.map_structure(lambda x: np.expand_dims(x, axis=0), nest)

def add_toggle_completion(
    toggle_outcomes: Dict[str, Dict[str, bool]],
    subtasks: List[str]):
  """Create toggled outcomes/completions mapping
    for each toggle-able subtasks.

    e.g.
      >>> toggle({}, ['Click A', 'Click B', 'Click C'])
      >>> {
            'Click A': {'Click B': False, 'Click C': False},
            'Click B': {'Click A': False, 'Click C': False},
            'Click C': {'Click A': False, 'Click B': False}
          }
  """
  for subtask in subtasks:
    toggle_outcomes.update({
        subtask: {s: False for s in subtasks if s != subtask}
    })
  return toggle_outcomes

def flatten(x):
  return list(itertools.chain(*x))

def remove_duplicates(x):
  assert isinstance(x, list)
  return list(dict.fromkeys(x))
