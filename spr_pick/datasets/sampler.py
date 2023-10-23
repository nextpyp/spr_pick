from __future__ import annotations

"""Contains custom sampler to allow repeated use of the same data with fair extraction.
"""


import torch

from torch.utils.data import Sampler, Dataset
from typing import Generator, List, Dict
import timeit
import numpy as np 

def enumerate_pu_coordinates(Y):
    """
    Given a list of 2d arrays containing labels, enumerate the positive and unlabeled(all) coordinates as (image,coordinate) pairs.
    """

    P_size = int(sum(y.sum() for y in Y)) # number of positive coordinates
    size = sum(y.size for y in Y)

    P = np.zeros(P_size, dtype=[('image', np.uint32), ('coord', np.uint32)])
    U = np.zeros(size, dtype=[('image', np.uint32), ('coord', np.uint32)])

    i = 0 # P index
    j = 0 # U index
    max_unravel_coord_0 = 0
    max_unravel_coord_1 = 0
    for image in range(len(Y)):
        # print('image', Y[image].shape)
        r, c = Y[image].shape
        y = Y[image].ravel()
        for coord in range(len(y)):
            unraveled_coord = np.unravel_index(coord, Y[image].shape)
            if unraveled_coord[0] > 72 and unraveled_coord[0] < c - 140 and unraveled_coord[1] > 72 and unraveled_coord[1] < r - 140:
                if y[coord]:
                    P[i] = (image, coord)
                    i += 1
                 
                if unraveled_coord[0] > max_unravel_coord_0:
                    max_unravel_coord_0 = unraveled_coord[0]
                if unraveled_coord[1] > max_unravel_coord_1:
                    max_unravel_coord_1 = unraveled_coord[1]
                U[j] = (image, coord)
                j += 1
    U = U[:j]
    P = P[:i]
    # print('U', U)
    # print('P', P)
    # print('U', U)
    # print('j', j)
    # print('positive U', U[:j])
    # print('zero U', U[j:])

    return P, U

class ShuffledSampler(Sampler):
    def __init__(self, x, random=np.random):
        self.x = x
        self.random = random
        self.i = len(self.x)

    def __len__(self):
        return len(self.x)

    def __next__(self):
        if self.i >= len(self.x):
            self.random.shuffle(self.x)
            self.i = 0
        sample = self.x[self.i]
        self.i += 1
        return sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        return self

class StratifiedCoordinateSampler(Sampler):
    def __init__(self, labels, balance=0.5, size=None, random=np.random, split='pn'):

        groups = []
        weights = np.zeros(len(labels)*2)
        proportions = np.zeros((len(labels), 2))
        i = 0
        for group in labels:
            
            P,U = enumerate_pu_coordinates(group)
            P = ShuffledSampler(P, random=random)
            U = ShuffledSampler(U, random=random)
            groups.append(P)
            groups.append(U)

            proportions[i//2,0] = (len(U) - len(P))/len(U)
            proportions[i//2,1] = len(P)/len(U)

            p = balance
            if balance is None:
                p = proportions[i//2,1]
            weights[i] = p/len(labels)
            weights[i+1] = (1-p)/len(labels)
            i += 2

        if size is None:
            sizes = np.array([len(g) for g in groups])
            size = int(np.round(np.min(sizes/weights)))

        self.groups = groups
        self.weights = weights
        self.proportions = proportions
        self.size = size

        self.history = np.zeros_like(self.weights)
        self.random = random

    def __len__(self):
        return self.size

    def __next__(self):
        n = self.history.sum()
        weights = self.weights
        if n > 0:
            weights = weights - self.history/n
            weights[weights < 0] = 0
            n = weights.sum()
            if n > 0:
                weights /= n
            else:
                weights = np.ones_like(weights)/len(weights)

        i = self.random.choice(len(weights), p=weights)
        self.history[i] += 1
        if np.all(self.history/self.history.sum() == self.weights):
            self.history[:] = 0

        g = self.groups[i]
        sample = next(g)

        i = i//2
        j,c = sample

        # code as integer
        # unfortunate hack required because pytorch converts index to integer...
        h = i*2**56 + j*2**32 + c
        return h
        #return i//2, sample

    # for python 2.7 compatability
    next = __next__

    def __iter__(self):
        for _ in range(self.size):
            yield next(self)


class FixedLengthSampler(Sampler):
    """Sample in either sequential or a random order for the given number of samples. If the
    number of requested samples execeds the dataset, the dataset will loop. Unlike standard
    sampling with replacement this means a sample will only ever be used once more than any
    other sample.

    There is no option for fully random selection with replacement, use PyTorch's
    `RandomSampler` if this behaviour is desired.

    Args:
        data_source (Dataset): Dataset to load samples from.
        num_samples (int, optional): The number of samples to be returned by the dataset.
            Defaults to None; this is equivalent to the length of the dataset.
        shuffled (bool, optional): Whether to randomise order. Defaults to False.
    """

    def __init__(
        self, data_source: Dataset, num_samples: int = None, shuffled: bool = False,
    ):
        self.data_source = data_source
        self._num_samples = num_samples
        self.shuffled = shuffled
        self._next_iter = None
        self._last_iter = None

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        else:
            return self._num_samples

    def sampler(self) -> Generator[int, None, None]:
        """Iterator handling both shuffled and non-shuffled behaviour.

        Yields:
            Generator[int, None, None]: Next index to sample.
        """
        remaining = self.num_samples
        if self.shuffled:
            while remaining > 0:
                n = min(remaining, len(self.data_source))
                for idx in torch.randperm(len(self.data_source))[0:n]:
                    yield int(idx)
                remaining -= n
        else:
            current_idx = None
            while remaining > 0:
                if current_idx is None or current_idx >= len(self.data_source):
                    current_idx = 0
                yield current_idx
                current_idx += 1
                remaining -= 1

    def __iter__(self) -> Generator[int, None, None]:
        # print('start')
        start = timeit.default_timer()
        if self._next_iter is None:
            sample_order = list(self.sampler())
            self._last_iter = SamplingOrder(sample_order)
            stop = timeit.default_timer()
            # print('Time: ', stop - start) 
            return self._last_iter

        else:
            stop = timeit.default_timer()
            # print('Time: ', stop - start) 
            return self._next_iter


    def __len__(self) -> int:
        return self.num_samples

    def for_next_iter(self, iter_order: SamplingOrder):
        self._next_iter = iter_order
        self._last_iter = iter_order

    def last_iter(self) -> Generator[int, None, None]:
        return self._last_iter


class SamplingOrder:
    def __init__(self, order: List[int], index: int = 0):
        self.order = order
        self.index = index

    def __iter__(self) -> Generator[int, None, None]:
        return self

    def __len__(self) -> int:
        return len(self.order)

    def __next__(self) -> int:
        if self.index < len(self.order):
            value = self.order[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration()

    def state_dict(self) -> Dict:
        state_dict = {"order": self.order, "index": self.index}
        return state_dict

    @staticmethod
    def from_state_dict(state_dict: Dict) -> SamplingOrder:
        return SamplingOrder(state_dict["order"], state_dict["index"])
