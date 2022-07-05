import os
import csv

from acme.utils.loggers import base
from acme.utils import tree_utils
from psgi.utils import log_utils


class CSVDumper(base.Logger):
  """Logger wrapper that intercepts and dumps
    average meta-eval performances to csv.
  """

  _open = open

  def __init__(
      self,
      to: base.Logger,
      logdir: str = 'logs',
      label: str = ''
  ):
    """Initializes the logger.
    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
    """
    self._file_path = os.path.join(logdir, f'{label}_logs.csv')
    self._to = to

  def write(self, data: base.LoggingData):
    if isinstance(data, list): # dump the whole csv at once
      with self._open(self._file_path, mode='w') as f:
        keys = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
      log_dict = log_utils.list_of_dict_to_dict_of_list(data)
      keyset = ['adaptation/steps', 'test/cumulative_reward', 'test/success']
      self.print_on_screen(log_dict, keyset=keyset)
    else:
      self._to.write(data)
      self.print_on_screen(data)

  def print_on_screen(self, log_dict, keyset=None):
    assert isinstance(log_dict, dict)
    vector_summary = {k: v for k, v in log_dict.items() if isinstance(v, list)}
    scalar_summary = {k: v for k, v in log_dict.items() if isinstance(v, float) or isinstance(v, int)}
    for key, vector in vector_summary.items():
      if keyset is None or key in keyset:
        if len(vector) <= 5:
          print(f'{key}: [' + ', '.join([f'{val:.3g}' for val in vector]) + ']')
        else:
          print(f'{key}: [{vector[0]:.3g}, {vector[1]:.3g}, ..., {vector[-1]:.3g}]')
    print('\t'.join(['%s: %.03g' % (key, value) for (key, value) in scalar_summary.items() if keyset is None or key in keyset]))
