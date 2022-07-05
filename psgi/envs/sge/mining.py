from enum import Enum, unique
from .utils import AGENT, BLOCK, WATER, EMPTY, TYPE_PICKUP, TYPE_TRANSFORM, KEY, MOVE_ACTS, OID_TO_IID
from .maze_config import MazeConfig

@unique
class Object(Enum):
  WORKSPACE = 0
  FURNACE = 1
  TREE = 2
  STONE = 3
  GRASS = 4
  PIG = 5
  COAL = 6
  IRON = 7
  SILVER = 8
  GOLD = 9
  DIAMOND = 10
  JEWELER = 11
  LUMBERSHOP = 12

class Mining(MazeConfig):
    def __init__(self):
        # map
        self.env_id = 'mining'
        nb_block = [1, 3]
        nb_water = [1, 3]

        # object
        objects = []
        objects.append(dict(
            name='workspace', pickable=False, transformable=True,
            oid=0, outcome=0, unique=True))
        objects.append(dict(
            name='furnace', pickable=False, transformable=True,
            oid=1, outcome=1, unique=True))

        objects.append(
            dict(name='tree', pickable=True, transformable=False,
                 oid=2, max=3))
        objects.append(
            dict(name='stone', pickable=True, transformable=False,
                 oid=3, max=3))
        objects.append(
            dict(name='grass', pickable=True, transformable=False,
                 oid=4, max=2))
        objects.append(
            dict(name='pig', pickable=True, transformable=False,
                 oid=5, max=1))

        objects.append(
            dict(name='coal', pickable=True, transformable=False,
                 oid=6, max=1))
        objects.append(
            dict(name='iron', pickable=True, transformable=False,
                 oid=7, max=1))
        objects.append(
            dict(name='silver', pickable=True, transformable=False,
                 oid=8, max=1))
        objects.append(
            dict(name='gold', pickable=True, transformable=False,
                 oid=9, max=1))
        objects.append(
            dict(name='diamond', pickable=True, transformable=False,
                 oid=10, max=3))
        objects.append(dict(
            name='jeweler', pickable=False, transformable=True, oid=11,
            outcome=11, unique=True))
        objects.append(dict(
            name='lumbershop', pickable=False, transformable=True, oid=12,
            outcome=12, unique=True))

        for obj in objects:
            obj['imgname'] = obj['name']+'.png'

        # operation: pickup (type=0) or transform (type=1)
        operation_list = {
            KEY.PICKUP: dict(name='pickup', oper_type=TYPE_PICKUP, key='p'),
            KEY.USE_1: dict(name='use_1', oper_type=TYPE_TRANSFORM, key='1'),
            KEY.USE_2: dict(name='use_2', oper_type=TYPE_TRANSFORM, key='2'),
            KEY.USE_3: dict(name='use_3', oper_type=TYPE_TRANSFORM, key='3'),
            KEY.USE_4: dict(name='use_4', oper_type=TYPE_TRANSFORM, key='4'),
            KEY.USE_5: dict(name='use_5', oper_type=TYPE_TRANSFORM, key='5'),
        }
        # item = agent+block+water+objects
        item_image_name_by_iid = dict()
        item_image_name_by_iid[AGENT] = 'agent.png'
        item_image_name_by_iid[BLOCK] = 'mountain.png'
        item_image_name_by_iid[WATER] = 'water.png'

        item_name_to_iid = dict()
        item_name_to_iid['agent'] = 0
        item_name_to_iid['block'] = 1
        item_name_to_iid['water'] = 2
        for obj in objects:
            iid = OID_TO_IID(obj['oid'])
            item_name_to_iid[obj['name']] = iid
            item_image_name_by_iid[iid] = obj['imgname']

        # subtask
        subtasks = []
        subtasks.append(dict(name='Cut wood', param=(KEY.PICKUP, Object.TREE.value)))
        subtasks.append(dict(name="Get stone", param=(KEY.PICKUP, Object.STONE.value)))
        subtasks.append(
            dict(name="Get string", param=(KEY.PICKUP, Object.GRASS.value)))  # 2
        #
        subtasks.append(
            dict(name="Make firewood", param=(KEY.USE_1, Object.LUMBERSHOP.value)))  # 3
        subtasks.append(dict(name="Make stick", param=(KEY.USE_2, Object.LUMBERSHOP.value)))
        subtasks.append(dict(name="Make arrow", param=(KEY.USE_3, Object.LUMBERSHOP.value)))
        subtasks.append(dict(name="Make bow", param=(KEY.USE_4, Object.LUMBERSHOP.value)))
        #
        subtasks.append(
            dict(name="Make stone pickaxe", param=(KEY.USE_1, Object.WORKSPACE.value)))  # 7
        subtasks.append(
            dict(name="Hit pig", param=(KEY.PICKUP, Object.PIG.value)))
        #
        subtasks.append(
            dict(name="Get coal", param=(KEY.PICKUP, Object.COAL.value)))  # 9
        subtasks.append(
            dict(name="Get iron ore", param=(KEY.PICKUP, Object.IRON.value)))
        subtasks.append(
            dict(name="Get silver ore", param=(KEY.PICKUP, Object.SILVER.value)))
        #
        subtasks.append(
            dict(name="Light furnace", param=(KEY.USE_1, Object.FURNACE.value)))  # 12
        #
        subtasks.append(
            dict(name="Smelt iron", param=(KEY.USE_2, Object.FURNACE.value)))  # 13
        subtasks.append(
            dict(name="Smelt silver", param=(KEY.USE_3, Object.FURNACE.value)))
        subtasks.append(
            dict(name="Bake pork", param=(KEY.USE_4, Object.FURNACE.value)))
        #
        subtasks.append(
            dict(name="Make iron pickaxe", param=(KEY.USE_2, Object.WORKSPACE.value)))  # 16
        subtasks.append(
            dict(name="Make silverware", param=(KEY.USE_3, Object.WORKSPACE.value)))
        #
        subtasks.append(
            dict(name="Get gold ore", param=(KEY.PICKUP, Object.GOLD.value)))  # 18
        subtasks.append(
            dict(name="Get diamond ore", param=(KEY.PICKUP, Object.DIAMOND.value)))
        #
        subtasks.append(
            dict(name="Smelt gold", param=(KEY.USE_5, Object.FURNACE.value)))  # 20
        subtasks.append(
            dict(name="Craft earrings", param=(KEY.USE_1, Object.JEWELER.value)))
        subtasks.append(
            dict(name="Craft rings", param=(KEY.USE_2, Object.JEWELER.value)))
        #
        subtasks.append(
            dict(name="Make goldware", param=(KEY.USE_4, Object.WORKSPACE.value)))  # 23
        subtasks.append(
            dict(name="Make bracelet", param=(KEY.USE_5, Object.WORKSPACE.value)))
        subtasks.append(
            dict(name="Craft necklace", param=(KEY.USE_3, Object.JEWELER.value)))
        #
        subtask_param_to_id = dict()
        subtask_param_list = []
        for i in range(len(subtasks)):
            subtask = subtasks[i]
            par = subtask['param']
            subtask_param_list.append(par)
            subtask_param_to_id[par] = i
        nb_obj_type = len(objects)
        nb_operation_type = len(operation_list)

        self.operation_list = operation_list
        self.legal_actions = MOVE_ACTS | {
            KEY.PICKUP, KEY.USE_1, KEY.USE_2, KEY.USE_3, KEY.USE_4, KEY.USE_5}

        self.nb_operation_type = nb_operation_type

        self.object_param_list = objects
        self.nb_obj_type = nb_obj_type
        self.item_name_to_iid = item_name_to_iid
        self.item_image_name_by_iid = item_image_name_by_iid
        self.nb_block = nb_block
        self.nb_water = nb_water
        self.subtasks = subtasks
        self.subtask_param_list = subtask_param_list
        self.subtask_param_to_id = subtask_param_to_id

        self.nb_subtask_type = len(subtasks)
        self.rendering_scale = 96
        self.width = 10
        self.height = 10
        self.feat_dim = 3*len(subtasks)+1
        self.ranksep = "0.1"
