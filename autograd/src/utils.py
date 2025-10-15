import numpy as np
import torch

from autograd.lib._C import Tensor as CTensor
from autograd.src import PyTensor


def from_numpy(data: np.ndarray) -> CTensor:
    return PyTensor(data).to_c()


def from_tensor(data: torch.Tensor) -> CTensor:
    return PyTensor(data.numpy()).to_c()
