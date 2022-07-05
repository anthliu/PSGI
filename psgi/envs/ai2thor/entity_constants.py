import csv
import pathlib
import numpy as np

ENTITY_DATA = 'data/ai2thor_v1.csv'
def parse_floorplan_entities(data_f):
  floorplans = {}# floorplan -> set of objects
  fp_feature_positives = {}# floorplan -> feature_name -> set of objects
  with open(data_f) as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
      _, obj_id = row['Name'].split('_')
      # assert row['Type'] == obj_type # XXX GlassBottle == Bottle? Check this later
      obj_type = row['Type']
      if obj_type in ['StoveKnob']:
        continue# hide these entities
      obj_name = obj_id + '_' + obj_type# use Id_Type format
      # special cases
      if obj_type in {'StoveBurner', 'Floor', 'Pan', 'Pot'}:
        obj_name = obj_type
      floorplan = int(row['Floorplan'])
      floorplans.setdefault(floorplan, set()).add(obj_name)
      for feature_name, value in row.items():
        fp_feature_positives.setdefault(floorplan, {})
        f_name = 'f_' + feature_name
        fp_feature_positives[floorplan].setdefault(f_name, set())
        if value == 'True':
          fp_feature_positives[floorplan][f_name].add(obj_name)

      # manually add features for food objects (some missing)
      if obj_type in ['Bread', 'Egg']:
        fp_feature_positives[floorplan].setdefault('f_Cookable', set()).add(obj_name)

      fp_feature_positives[floorplan].setdefault(f'f_{obj_type}', set()).add(obj_name)

  return_floorplan = {floor: list(sorted(obj_set)) for floor, obj_set in floorplans.items()}
  return_feature_positives = {fp: {feature_name: list(sorted(obj_set)) for feature_name, obj_set in feats.items()} for fp, feats in fp_feature_positives.items()}
  return return_floorplan, return_feature_positives

FP_TO_ENTITY, FP_TO_FEATURES_POSITIVES = parse_floorplan_entities(pathlib.Path(__file__).parent / ENTITY_DATA)




OBJ_REC_COMP = {
    "Apple": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Dresser"],
    "AppleSliced": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Dresser"],
    "Book": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "Bed", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer"],
    "Bottle": ["Fridge", "Box", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "GarbageCan"],
    "Bowl": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"],
    "Bread": ["Microwave", "Fridge", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Plate"],
    "BreadSliced": ["Microwave", "Fridge", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "GarbageCan", "Toaster", "Plate"],
    "ButterKnife": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Cup", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "Desk", "CounterTop", "Drawer"],
    "CellPhone": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "Bed", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Safe"],
    "CreditCard": ["Sofa", "ArmChair", "Box", "Ottoman", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "Shelf"],
    "Cup": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"],
    "DishSponge": ["Pot", "Pan", "Bowl", "Plate", "Box", "Toilet", "Cart", "Cart", "BathtubBasin", "Bathtub", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"],
    "Egg": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"],
    "EggCracked": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"],
    "Fork": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Cup", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"],
    "Kettle": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Sink", "SinkBasin", "Cabinet", "StoveBurner", "Shelf"],
    "Knife": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"],
    "Ladle": ["Pot", "Pan", "Bowl", "Plate", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"],
    "Lettuce": ["Pot", "Pan", "Bowl", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"],
    "LettuceSliced": ["Pot", "Pan", "Bowl", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"],
    "Mug": ["SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"],
    "Pan": ["DiningTable", "CounterTop", "TVStand", "CoffeeTable", "SideTable", "Sink", "SinkBasin", "Cabinet", "StoveBurner", "Fridge"],
    "Pen": ["Mug", "Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"],
    "Pencil": ["Mug", "Box", "Dresser", "Desk", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"],
    "PepperShaker": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer", "Cabinet", "Shelf"],
    "Plate": ["Microwave", "Fridge", "Dresser", "Desk", "Sink", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"],
    "Pot": ["StoveBurner", "Fridge", "Sink", "SinkBasin", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf"],
    "Potato": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"],
    "PotatoSliced": ["Pot", "Pan", "Bowl", "Microwave", "Fridge", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "GarbageCan"],
    "SaltShaker": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer", "Cabinet", "Shelf"],
    "SoapBottle": ["Dresser", "Desk", "Toilet", "Cart", "Bathtub", "Sink", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"],
    "Spatula": ["Pot", "Pan", "Bowl", "Plate", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"],
    "Spoon": ["Pot", "Pan", "Bowl", "Mug", "Plate", "Cup", "Sink", "SinkBasin", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Drawer"],
    "SprayBottle": ["Dresser", "Desk", "Toilet", "Cart", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Drawer", "GarbageCan"],
    "Statue": ["Box", "Dresser", "Desk", "Cart", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Safe"],
    "Tomato": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Sink", "SinkBasin", "Pot", "Bowl", "Fridge", "GarbageCan", "Plate"],
    "TomatoSliced": ["DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Sink", "SinkBasin", "Pot", "Bowl", "Fridge", "GarbageCan", "Plate"],
    "Vase": ["Box", "Dresser", "Desk", "Cart", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "Safe"],
    "WineBottle": ["Fridge", "Dresser", "Desk", "Cabinet", "DiningTable", "TVStand", "CoffeeTable", "SideTable", "CounterTop", "Shelf", "GarbageCan"]
}
REC_OBJ_COMP = {}
for obj, receps in OBJ_REC_COMP.items():
  for recep in receps:
    REC_OBJ_COMP.setdefault(recep, set()).add(obj)

def find_compat_clusters(obj_to_recep):
  c_objs = []
  c_recep_sets = []
  for obj, receps in obj_to_recep.items():
    receps = set(receps)
    found_cluster = -1
    for i, c_receps in enumerate(c_recep_sets):
      if receps == c_receps:
        found_cluster = i
        break
    if found_cluster < 0:
      c_objs.append([obj])
      c_recep_sets.append(receps)
    else:
      c_objs[found_cluster].append(obj)
  return c_objs, c_recep_sets

COMPAT_CLUSTERS = list(zip(*find_compat_clusters(OBJ_REC_COMP)))

def construct_compat_features(fp_to_entity, fp_to_feature_positives, compat_clusters):
  comp_features = {}
  for group, (c_objs, c_receps) in enumerate(compat_clusters):
    feat_name = f'f_PlaceGroup{group}'
    positive_ents = set(c_objs)
    positive_ents.update(c_receps)
    comp_features[feat_name] = positive_ents
  fp_to_compat = {}
  for floorplan, entities in fp_to_entity.items():
    fp_to_compat[floorplan] = {}
    for entity in entities:
      for feat_name, ents in comp_features.items():
        obj_type = entity.split('_')[-1]# XXX hack, obj name stored as id_type
        if obj_type in ents:
          fp_to_compat[floorplan].setdefault(feat_name, []).append(entity)

  return list(sorted(comp_features.keys())), fp_to_compat

COMPAT_FEATURES, FP_TO_COMPAT_FEATURES = construct_compat_features(FP_TO_ENTITY, FP_TO_FEATURES_POSITIVES, COMPAT_CLUSTERS)

### Substitute for test objects in floors 21-30
TEST_FLOORS = list(range(21, 31))
TEST_SUB_OBJS = {
    'Potato': ['Yam', 'Turnip'],
    'Bread': ['Baguette', 'Croissant'],
    'Apple': ['Orange', 'Pear', 'Peach'],
    'Lettuce': ['Broccoli', 'Cabbage'],
    'Tomato': ['Broccoli', 'Cabbage'],
    'Egg': ['Beef', 'Chicken']
}
obj_rng = np.random.default_rng(0)
for floorplan in TEST_FLOORS:
  remap = {}
  for entity in FP_TO_ENTITY[floorplan]:
    if '_' in entity:
      obj_id, obj_type = entity.split('_')# XXX hack, obj name stored as id_type
      obj_id = obj_id + '_'
    else:
      obj_type = entity
      obj_id = ''
    if obj_type in TEST_SUB_OBJS:
      new_type = obj_rng.choice(TEST_SUB_OBJS[obj_type])
      remap[entity] = obj_id + new_type

  for i, entity in enumerate(FP_TO_ENTITY[floorplan]):
    FP_TO_ENTITY[floorplan][i] = remap.get(entity, entity)
  for f_name, entities in FP_TO_FEATURES_POSITIVES[floorplan].items():
    for i, entity in enumerate(entities):
      FP_TO_FEATURES_POSITIVES[floorplan][f_name][i] = remap.get(entity, entity)
  for f_name, entities in FP_TO_COMPAT_FEATURES[floorplan].items():
    for i, entity in enumerate(entities):
      FP_TO_COMPAT_FEATURES[floorplan][f_name][i] = remap.get(entity, entity)

### Add more food objects to each floor
#### Helper, first calculate ent_type->feature
def _ent_type_to_feature(fp_to_feat):
  result = {}
  for floorplan, feats in fp_to_feat.items():
    for feat_name, ents in feats.items():
      for ent in ents:
        obj_type = ent.split('_')[-1]
        result.setdefault(obj_type, set()).add(feat_name)
  return result
_ENT_TO_FEAT = _ent_type_to_feature(FP_TO_FEATURES_POSITIVES)
_ENT_TO_COMPAT = _ent_type_to_feature(FP_TO_COMPAT_FEATURES)
####
TRAIN_DUPE_OBJS = {
    'Pork': 'Egg',
    'Onion': 'Potato',
    'Bagel': 'Bread',
    'Carrot': 'Potato'
}
dupe_objs = list(TRAIN_DUPE_OBJS.keys())
dupe_rng = np.random.default_rng(0)
dupe_id = 1
for floorplan in range(1, 31):
  add_objs = dupe_rng.choice(dupe_objs, 2)
  for add_obj in add_objs:
    name = f'd{dupe_id}_{add_obj}'
    dupe_id += 1

    FP_TO_ENTITY[floorplan].append(name)

    ancestor = TRAIN_DUPE_OBJS[add_obj]
    for feat in _ENT_TO_FEAT[ancestor]:
      FP_TO_FEATURES_POSITIVES[floorplan].setdefault(feat, []).append(name)
    for feat in _ENT_TO_COMPAT[ancestor]:
      FP_TO_COMPAT_FEATURES[floorplan].setdefault(feat, []).append(name)

if __name__ == '__main__':
  from pprint import pprint
  #d = list(zip(*find_compat_clusters(OBJ_REC_COMP)))
  #d = list(zip(*find_compat_clusters(REC_OBJ_COMP)))
  #pprint(FP_TO_COMPAT_FEATURES[28])
  pprint(FP_TO_FEATURES_POSITIVES[23])
  #pprint(_ENT_TO_FEAT)

  '''
  Find entities without glove embeddings
  from torchtext.vocab import GloVe
  from sklearn.cluster import KMeans
  from sklearn.exceptions import ConvergenceWarning

  GLOVE_DIM = 50
  MAX_PRED_DIM = 4
  EMBEDDING_GLOVE = GloVe(name='6B', dim=GLOVE_DIM)
  all_ents = set()
  for ents in FP_TO_ENTITY.values():
    for ent in ents:
      all_ents.add(ent.split('_')[-1].lower())
  for ent in all_ents:
    if ent not in EMBEDDING_GLOVE.stoi:
      print(ent)
  '''
