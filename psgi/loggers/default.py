# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default logger."""

from acme.utils.loggers import base
from acme.utils.loggers import aggregators
from acme.utils.loggers import filters
from acme.utils.loggers import terminal

from psgi.loggers import csv
from psgi.loggers import tf_summary
from psgi.loggers import wrapper


def make_default_logger(
    logdir: str,
    label: str,
    save_data: bool = True,
    time_delta: float = 0.0,
) -> base.Logger:
  """Make a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Ignored.
    time_delta: Time (in seconds) between logging events.

  Returns:
    A logger (pipe) object that responds to logger.write(some_dict).
  """
  loggers = []

  # TODO: temporarily disable terminal logger for environment.
  if 'agent' in label:
    loggers.append(terminal.TerminalLogger(label=label, time_delta=time_delta))

  if save_data:
    loggers.append(csv.CSVLogger(logdir=logdir, label=label))
    loggers.append(tf_summary.TFSummaryLogger(logdir=logdir, label=label))

  logger = aggregators.Dispatcher(loggers)
  logger = filters.NoneFilter(logger)
  logger = filters.TimeFilter(logger, time_delta)

  if save_data:
    logger = wrapper.CSVDumper(logger, label=label, logdir=logdir)

  return logger
