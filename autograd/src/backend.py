from enum import auto, Enum, StrEnum
from typing import Callable

import numpy as np


class ComputationalBackends(StrEnum):
    NUMPY_CPU = "numpy"
    CPP_CPU = "cpp"
    CUDA = "cuda"


class Ops(Enum):
    # initializers
    zeros = auto()
    zeros_like = auto()
    ones = auto()
    ones_like = auto()
    # sum
    sum = auto()
    sum_backward = auto()
    # add
    add = auto()
    add_backward = auto()
    add_inplace = auto()
    # mm
    mm = auto()
    mm_backward = auto()
    # activations
    sigmoid = auto()
    sigmoid_backward = auto()
    leaky_relu = auto()
    leaky_relu_backward = auto()


CB = ComputationalBackends

DISPATCH_TABLE = {}

def register_table(op: Ops, backend: ComputationalBackends) -> None:
    def inner(fn):
        DISPATCH_TABLE.setdefault(op, {})[backend] = fn
        return fn
    return inner

def dispatch(op: Ops, to_backend: ComputationalBackends) -> Callable:
    return DISPATCH_TABLE[op][to_backend]


# ----------------------------- basic numpy impls ---------------------------------
# base
@register_table(Ops.zeros, CB.NUMPY_CPU)
def numpy_zeros_impl(shape: list[int]) -> np.ndarray:
    return np.zeros(shape=shape)

@register_table(Ops.zeros_like, CB.NUMPY_CPU)
def numpy_zeros_like_impl(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)

@register_table(Ops.ones, CB.NUMPY_CPU)
def numpy_ones_impl(shape: list[int]) -> np.ndarray:
    return np.ones(shape=shape)

@register_table(Ops.ones_like, CB.NUMPY_CPU)
def numpy_ones_like_impl(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

# sum
@register_table(Ops.sum, CB.NUMPY_CPU)
def numpy_sum_impl(x: np.ndarray) -> np.ndarray:
    return np.sum(x)

@register_table(Ops.sum_backward, CB.NUMPY_CPU)
def numpy_sum_backward_impl(grad, x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

# add
@register_table(Ops.add, CB.NUMPY_CPU)
def numpy_add_impl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

@register_table(Ops.add_inplace, CB.NUMPY_CPU)
def numpy_add_inplace(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a += b

@register_table(Ops.add_backward, CB.NUMPY_CPU)
def numpy_add_backward_impl(grad: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ...  # TODO

# mm
@register_table(Ops.mm, CB.NUMPY_CPU)
def numpy_mm_impl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)

@register_table(Ops.mm_backward, CB.NUMPY_CPU)
def numpy_mm_backward_impl(grad: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.matmul(grad, b.T), np.matmul(a.T, grad))


# ------------------------------ C++ backend impls --------------------------------
from autograd.lib import _C

# initializers
@register_table(Ops.zeros, CB.CPP_CPU)
def c_zeros_impl(shape: list[int]) -> _C.Tensor:
    return _C.zeros(shape)

@register_table(Ops.zeros_like, CB.CPP_CPU)
def c_zeros_like_impl(x: _C.Tensor) -> _C.Tensor:
    return _C.zeros_like(x)

@register_table(Ops.ones, CB.CPP_CPU)
def c_ones_impl(shape: list[int]) -> _C.Tensor:
    return _C.ones(shape)

@register_table(Ops.ones_like, CB.CPP_CPU)
def c_ones_like_impl(x: _C.Tensor) -> _C.Tensor:
    return _C.ones_like(x)

# sum
@register_table(Ops.sum, CB.CPP_CPU)
def c_sum_impl(x: _C.Tensor) -> _C.Tensor:
    return _C.sum(x)

@register_table(Ops.sum_backward, CB.CPP_CPU)
def c_sum_backward_impl(grad: _C.Tensor, x: _C.Tensor) -> _C.Tensor:
    return _C.ones_like(x)

# add
@register_table(Ops.add, CB.CPP_CPU)
def c_add_impl(a: _C.Tensor, b: _C.Tensor) -> _C.Tensor:
    return _C.add(a, b)

@register_table(Ops.add_inplace, CB.CPP_CPU)
def c_add_inplace_impl(a: _C.Tensor, b: _C.Tensor) -> _C.Tensor:
    return _C.add_inplace(a, b)

@register_table(Ops.add_backward, CB.CPP_CPU)
def c_add_backward_impl(grad: _C.Tensor, a:_C.Tensor, b:_C.Tensor) -> tuple[_C.Tensor, ...]:
    return _C.add_backward(grad, a, b)

# mm
@register_table(Ops.mm, CB.CPP_CPU)
def c_mm_impl(a: _C.Tensor, b: _C.Tensor) -> _C.Tensor:
    return _C.mm(a, b)

@register_table(Ops.mm_backward, CB.CPP_CPU)
def c_mm_backward(grad: _C.Tensor, a: _C.Tensor, b: _C.Tensor) -> _C.Tensor:
    return _C.mm_backward(grad, a, b)
