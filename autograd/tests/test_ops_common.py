from itertools import product

import numpy as np
import pytest

from autograd.lib import _C
from autograd.src import PyTensor

ATOL = 1e-2


# Testing broadcast & reduce utilities
@pytest.mark.parametrize(
    argnames="s0,s1,expected",
    argvalues=[
        ([2, 5], [5], ([2, 5], [0])),
        ([120, 24], [120, 1], ([120, 24], [1])),
        ([13, 24, 89], [13, 24, 1], ([13, 24, 89], [2])),
        ([13, 24, 89], [13, 1, 89], ([13, 24, 89], [1])),
    ],
)
def test_ops_common(s0: list[int], s1: list[int], expected):
    res1 = _C._broadcast_shapes(s0, s1)
    res2 = _C._dims_to_reduce(s0, s1)
    r1, r2 = expected
    assert res1 == r1
    assert res2 == r2


@pytest.mark.parametrize(
    argnames="m,b",
    argvalues=product(
        [23, 37],
        [2, 5],
    )
)
def test_broadcast_1d_to_2d(m, b):
    a_ = np.random.randn(m)

    a = PyTensor(a_).to_c()
    a_bcast = a.broadcast_to([b, *a.shape])

    assert a_bcast.numel == (b * m)
    assert not a_bcast.is_contiguous
    assert a_bcast.stride == [0, 1]

    a2 = a_bcast.contiguous()

    assert a2.numel == (b * m)
    assert a2.is_contiguous
    assert a2.stride == [m, 1]
    _C._flush_temp()



@pytest.mark.parametrize(
    argnames="m,n,b",
    argvalues=product(
        [23, 37],
        [19, 72],
        [2, 5],
    )
)
def test_broadcast_2d_to_3d(m, n, b):
    a_ = np.random.randn(m, n)

    a = PyTensor(a_).to_c()
    a_bcast = a.broadcast_to([b, *a.shape])

    assert a_bcast.numel == (b * m * n)
    assert not a_bcast.is_contiguous
    assert a_bcast.stride == [0, n, 1]

    a2 = a_bcast.contiguous()

    assert a2.numel == (b * m * n)
    assert a2.is_contiguous
    assert a2.stride == [m*n, n, 1]
    _C._flush_temp()


# testing Dropout # TODO


# testing conv2d
def test_conv_2d():
    ...