# Expose base/sge envrionment.
from psgi.envs.base import BaseWoBEnv
from psgi.envs.base_predicate_env import BasePredicateEnv
from psgi.envs.sge.mazeenv import MazeEnv, MazeOptionEnv

from psgi.envs import logic_graph

# Expose environment configurations.
from psgi.envs.walmart.walmart_config import Walmart
from psgi.envs.dicks.dicks_config import Dicks
from psgi.envs.toywob.toywob_config import ToyWoB
from psgi.envs.converse.converse_config import Converse
from psgi.envs.bestbuy.bestbuy_config import BestBuy
from psgi.envs.apple.apple_config import Apple
from psgi.envs.amazon.amazon_config import Amazon
from psgi.envs.ebay.ebay_config import Ebay
from psgi.envs.samsung.samsung_config import Samsung
from psgi.envs.ikea.ikea_config import IKEA
from psgi.envs.target.target_config import Target

# Expose environment version 2 configurations.
from psgi.envs.walmart.walmart_config_v2 import Walmart2
from psgi.envs.dicks.dicks_config_v2 import Dicks2
from psgi.envs.converse.converse_config_v2 import Converse2
from psgi.envs.bestbuy.bestbuy_config_v2 import BestBuy2
from psgi.envs.apple.apple_config_v2 import Apple2
from psgi.envs.amazon.amazon_config_v2 import Amazon2
from psgi.envs.ebay.ebay_config_v2 import Ebay2
from psgi.envs.samsung.samsung_config_v2 import Samsung2
from psgi.envs.ikea.ikea_config_v2 import IKEA2
from psgi.envs.target.target_config_v2 import Target2

# Expose PDDL environment configurations.
from psgi.envs.toy_mining.mining_config import ETMining
from psgi.envs.toy_cooking.cooking_config import Cooking
from psgi.envs.toy_cooking.cooking_classic_config import CookingClassic
from psgi.envs.toy_cooking.pickandplace_config import PickPlace

from psgi.envs.ai2thor.ai2thor_base_config import AI2ThorConfig
from psgi.envs.ai2thor.ai2thor_env import AI2ThorEnv

# XXX TODO: Remove later
from psgi.envs.dummy.dummy_config import Dummy
from psgi.envs.toy_walmart.config import ToyWalmart
from psgi.envs.toy_dummy.config import ToyDummy


web_configs = [
    Walmart, Dicks, Converse, BestBuy, Apple,
    Amazon, Ebay, Samsung, IKEA, Target
]
web_configs_v2 = [
    Walmart2, Dicks2, Converse2, BestBuy2, Apple2,
    Amazon2, Ebay2, Samsung2, IKEA2, Target2
]
web_toy_configs = [Dummy, ToyWalmart, ToyDummy]   # XXX TODO: Remove

web_configs_dict = {c.environment_id: c for c in web_configs}
web_configs_v2_dict = {c.environment_id: c for c in web_configs_v2}

# Train/test splits for web navigation tasks.
WEBNAV_TASKS = {
    # seed 1
    'train_1': [Apple, Converse, BestBuy],
    'eval_1': [Target, IKEA, Ebay, Walmart, Samsung, Amazon, Dicks],
    # seed 2
    'train_2': [Dicks, Samsung, IKEA],
    'eval_2': [Apple, Amazon, BestBuy, Converse, Walmart, Ebay, Target],
    # seed 3
    'train_3': [Amazon, Ebay, Walmart],
    'eval_3': [Dicks, BestBuy, Target, Samsung, IKEA, Converse, Apple],
    # seed 4
    'train_4': [BestBuy, IKEA, Apple],
    'eval_4': [Amazon, Samsung, Dicks, Converse, Target, Walmart, Ebay],
    # seed 5
    'train_5': [Target, Apple, Walmart],
    'eval_5': [BestBuy, Dicks, Converse, Samsung, Amazon, Ebay, IKEA],
    # seed 6
    'train_6': [Ebay, Target, Converse],
    'eval_6': [Apple, Walmart, IKEA, BestBuy, Dicks, Samsung, Amazon],
    # seed 7
    'train_7': [Converse, BestBuy, Samsung],
    'eval_7': [IKEA, Apple, Walmart, Dicks, Target, Amazon, Ebay],
    # seed 8
    'train_8': [Walmart, Dicks, Amazon],
    'eval_8': [Samsung, Target, Converse, IKEA, Ebay, BestBuy, Apple],
    # seed 9
    'train_9': [Samsung, Amazon, Apple],
    'eval_9': [Target, Dicks, Converse, Ebay, Walmart, IKEA, BestBuy],
    # seed 10
    'train_10': [Amazon, IKEA, Converse],
    'eval_10': [Samsung, Ebay, Target, Dicks,  Apple, BestBuy, Walmart],
}


def get_subtask_label(env_id):
  if env_id == 'toywob':
    from psgi.envs.toywob.toywob_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'walmart':
    from psgi.envs.walmart.walmart_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'dicks':
    from psgi.envs.dicks.dicks_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'converse':
    from psgi.envs.converse.converse_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'bestbuy':
    from psgi.envs.bestbuy.bestbuy_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'apple':
    from psgi.envs.apple.apple_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'amazon':
    from psgi.envs.amazon.amazon_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'ebay':
    from psgi.envs.ebay.ebay_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'samsung':
    from psgi.envs.samsung.samsung_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'ikea':
    from psgi.envs.ikea.ikea_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'target':
    from psgi.envs.target.target_config import LABEL_NAME
    return LABEL_NAME
  elif env_id == 'playground':
    # TODO: maybe use subtask name
    return [str(num) for num in range(13)]
  else:
    return ValueError(env_id)
