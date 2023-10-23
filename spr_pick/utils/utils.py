"""Generic utilities for use with package.
"""

from __future__ import print_function, division

import numpy as np
import torch
import re
import os
import time

from typing import Any, List
from torch import Tensor
from contextlib import contextmanager
from collections import OrderedDict


def insize_from_outsize(layers, outsize):
    """ calculates in input size of a convolution stack given the layers and output size """
    for layer in layers[::-1]:
        if hasattr(layer, 'kernel_size'):
            kernel_size = layer.kernel_size # assume square
            if type(kernel_size) is tuple:
                kernel_size = kernel_size[0]
        else:
            kernel_size = 1
        if hasattr(layer, 'stride'):
            stride = layer.stride
            if type(stride) is tuple:
                stride = stride[0]
        else:
            stride = 1
        if hasattr(layer, 'padding'):
            pad = layer.padding
            if type(pad) is tuple:
                pad = pad[0]
        else:
            pad = 0
        if hasattr(layer, 'dilation'):
            dilation = layer.dilation
            if type(dilation) is tuple:
                dilation = dilation[0]
        else:
            dilation = 1

        outsize = (outsize-1)*stride + 1 + (kernel_size-1)*dilation - 2*pad 
    return outsize


def compute_ramped_lrate(
    i: int,
    iteration_count: int,
    ramp_up_fraction: float,
    ramp_down_fraction: float,
    learning_rate: float,
) -> float:
    if ramp_up_fraction > 0.0:
        ramp_up_end_iter = iteration_count * ramp_up_fraction
        if i <= ramp_up_end_iter:
            t = (i / ramp_up_fraction) / iteration_count
            learning_rate = learning_rate * (0.5 - np.cos(t * np.pi) / 2)

    if ramp_down_fraction > 0.0:
        ramp_down_start_iter = iteration_count * (1 - ramp_down_fraction)
        if i >= ramp_down_start_iter:
            t = ((i - ramp_down_start_iter) / ramp_down_fraction) / iteration_count
            learning_rate = learning_rate * (0.5 + np.cos(t * np.pi) / 2) ** 2

    return learning_rate


def list_constants(clazz: Any, private: bool = False) -> List[Any]:
    """Fetch all values from variables formatted as constants in a class.

    Args:
        clazz (Any): Class to fetch constants from.

    Returns:
        List[Any]: List of values.
    """
    variables = [i for i in dir(clazz) if not callable(i)]
    regex = re.compile(r"^{}[A-Z0-9_]*$".format("" if private else "[A-Z]"))
    names = list(filter(regex.match, variables))
    values = [clazz.__dict__[name] for name in names]
    return values


@contextmanager
def cd(newdir: str):
    """Context manager for managing changes of directory where when the context is left
    the original directory is restored.

    Args:
        newdir (str): New directory to enter
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


class TrackedTime:
    """Class for tracking an ongoing total time. Every update tracks the previous
    time for future updates.
    """

    def __init__(self):
        self.total = 0
        self.last_time = None

    def update(self):
        """Update the total time with the time since the last tracked time.
        """
        current_time = time.time()
        if self.last_time is not None:
            self.total += current_time - self.last_time
        self.last_time = current_time

    def forget(self):
        """Clear the last tracked time.
        """
        self.last_time = None


def seconds_to_dhms(seconds: float, trim: bool = True) -> str:
    """Convert time in seconds to a string of seconds, minutes, hours, days.

    Args:
        seconds (float): Time to convert.
        trim (bool, optional): Whether to remove leading time units if not needed.

    Returns:
        str: String representation of time.
    """
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // (60 * 60) % 24
    d = seconds // (60 * 60 * 24)
    times = [(d, "d"), (h, "h"), (m, "m"), (s, "s")]
    time_str = ""
    for t, char in times:
        if trim and t < 1:
            continue
        trim = False
        time_str += "{:02}{}".format(int(t), char)
    return time_str


class Metric:
    """ Only works if batch is in first dim.
    """

    def __init__(self, batched: bool = True, collapse: bool = True):
        self.reset()
        self.batched = batched
        self.collapse = collapse

    def add(self, value: Tensor):
        n = value.shape[0] if self.batched else 1
        if self.collapse:
            data_start = 1 if self.batched else 0
            mean_dims = list(range(data_start, len(value.shape)))
            if len(mean_dims) > 0:
                value = torch.mean(value, dim=mean_dims)
        if self.batched:
            value = torch.sum(value, dim=0)
        if self.total is None:
            self.total = value
        else:
            self.total += value
        self.n += n

    def __add__(self, value: Tensor):
        self.add(value)
        return self

    def accumulated(self, reset: bool = False):
        if self.n == 0:
            return None
        acc = self.total / self.n
        if reset:
            self.reset()
        return acc

    def reset(self):
        self.total = None
        self.n = 0

    def empty(self) -> bool:
        return self.n == 0


class MetricDict(OrderedDict):
    def __missing__(self, key):
        self[key] = value = Metric()
        return value


def separator(cols=100) -> str:
    return "#" * cols
