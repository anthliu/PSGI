import os
import shutil
import csv
from acme.utils import paths
from acme.utils import tree_utils
from psgi import loggers
from collections import defaultdict

def list_of_dict_to_dict_of_list(list_of_dict):
  keys = list_of_dict[0].keys()
  for dic in list_of_dict:
    assert isinstance(dic, dict)
    assert all([key in dic for key in keys])
  # See https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
  dict_of_list = {k: [dic[k] for dic in list_of_dict] for k in keys}
  return dict_of_list

def dump_csv(filename, data):
  with open(filename, mode='w') as f:
    keys = data.keys()
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    data = tree_utils.unstack_sequence_fields(data, len(data['test_score']))
    writer.writerows(data)

def create_loggers(
    logdir: str,
    label: str,
    save_data: bool = False,
):
  """Create loggers for the environment loop and the agent.
  """
  if save_data and os.path.exists(logdir):
    shutil.rmtree(logdir)
    os.makedirs(logdir)

  environment_logger = loggers.make_default_logger(
      logdir=logdir,
      label=label,
      save_data=save_data
  )
  agent_logger = loggers.make_default_logger(
      logdir=logdir,
      label='agent',
      save_data=save_data,
      time_delta=10.0
  )
  return environment_logger, agent_logger

class TimeProfiler(object):
  """Maintain moving statistics of time profile."""
  def __init__(self, prefix=''):
    self._prefix = 'time_profile/' + prefix
    self._buffer_dict = defaultdict(float)
    self._buffer_count = 0.
    self._safe_lock = defaultdict(bool)
    #
    self._prev_time_stamp = None

  def stamp(self, time_stamp, name=None):
    if name is not None:
      if self._prev_time_stamp is None:
        print("Error in Timeprofiler!!! You should call stamp() without 'name' argument after 'period_over()!!")
        assert False
      self._buffer_dict[name] += time_stamp - self._prev_time_stamp
    '''if self._prev_time_stamp is None: pass
    else:
      print(name, time_stamp - self._prev_time_stamp)'''
    self._prev_time_stamp = time_stamp
  
  def period_over(self):
    self._prev_time_stamp = None
    self._buffer_count += 1.
  
  def print(self):
    if self._prev_time_stamp is not None:
      print("Error in Timeprofiler!!! You should finish profiling full period before logging!")
      assert False
    for k, v in self._buffer_dict.items():
      print('%s = %.3f (sec)'%(self._prefix + k, v / self._buffer_count))
  
  def log_summary(self, summary_logger):
    if self._prev_time_stamp is not None:
      print("Error in Timeprofiler!!! You should finish profiling full period before logging!")
      assert False
    for k, v in self._buffer_dict.items():
      summary_logger.logkv(self._prefix + k, v / self._buffer_count)
