# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Loggers for experiments."""

from typing import Any, Callable, Dict, Mapping, Optional

import logging
import time
import wandb

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal


class WeightsAndBiasesLogger(base.Logger):

  def __init__(
      self,
      logger: wandb.sdk.wandb_run.Run,
      label: str = '',
      time_delta: float = 0.0,
  ):
    """Initializes the Weights And Biases wrapper for Acme.

    Args:
      logger: Weights & Biases logger instances
      label: label string to use when logging.
      serialize_fn: function to call which transforms values into a str.
      time_delta: How often (in seconds) to write values. This can be used to
        minimize terminal spam, but is 0 by default---ie everything is written.
    """
    self._label = label
    self._time = time.time()
    self._time_delta = time_delta
    self._logger = logger

  def write(self, data: base.LoggingData):
      """Write to weights and biases."""
      now = time.time()
      if (now - self._time) > self._time_delta:
        data = base.to_numpy(data)  # type: ignore
        if self._label:
            stats = {f"{self._label}/{k}": v for k, v in data.items()}
        else:
            stats = data
        self._logger.log(stats)  # type: ignore
        self._time = now

  def close(self):
      pass

def make_logger(
    label: str,
    wandb_logger: wandb.sdk.wandb_run.Run,
    steps_key: str = 'steps',
    save_data: bool = False,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
) -> base.Logger:
  """Makes a default Acme logger.

  Args:
    label: Name to give to the logger.
    wandb_logger: Weights and Biases logger instance.
    save_data: Whether to persist data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Ignored.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  del steps_key
  if not print_fn:
    print_fn = logging.info

  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)
  wandb_logger = WeightsAndBiasesLogger(
    logger=wandb_logger,
    label=label)

  loggers = [terminal_logger, wandb_logger]

  if save_data:
    loggers.append(csv.CSVLogger(label=label))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)

  return logger


def make_experiment_logger_factory(
        wandb_kwargs = Dict[str, Any]
  ) -> Callable[[str, Optional[str], int], base.Logger]:
  """Makes an Acme logger factory.

  Args:
    wandb_kwargs: Dictionary of keywork arguments for wandb.init().

  Returns:
    A logger factory function.
  """

  # In the distributed setting, it is better to initialize the logger once and pickle,
  # than to initialize the W&B logging in each process.
  wandb_logger = wandb.init(
    **wandb_kwargs,
  )

  def make_experiment_logger(label: str,
                             steps_key: Optional[str] = None,
                             task_instance: int = 0) -> base.Logger:
    del task_instance
    if steps_key is None:
      steps_key = f'{label}_steps'
    return make_logger(label=label, steps_key=steps_key,
      wandb_logger=wandb_logger,
    )
  return make_experiment_logger
