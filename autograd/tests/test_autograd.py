from itertools import product

import numpy as np
import pytest
import torch

from autograd.lib import _C
from autograd.src import PyTensor, Autograd, AutogradBackends


ATOL = 1e-4


# testing optimizer
@pytest.mark.parametrize(
    argnames="m,n",
    argvalues=product(
        [27, 57, 324, 678, 1091],
        [18, 39, 189, 782, 890],
    )
)
def test_optimizer(m: int, n: int):
    a_ = np.random.randn(m, n).astype(np.float32)
    a__ = a_[:, :]
    lr = abs(np.random.randn())

    a = PyTensor(a_).to_c()
    a.requires_grad = True
    optim = _C.SGDOptimizer([a], lr=lr)
    a.grad = _C.Tensor(a.shape, 1.0)
    optim.step()

    a_torch = torch.from_numpy(a__)
    a_torch.requires_grad = True
    torch_optim = torch.optim.SGD(
        params=(x for x in [a_torch]), lr=lr, weight_decay=0, momentum=0
    )
    a_torch.grad = torch.ones(size=a.shape).float()
    torch_optim.step()

    assert np.allclose(a.numpy(), a_torch.detach().numpy(), atol=ATOL)

    optim.zero_grad()
    assert np.sum(a.grad.numpy()) == 0

    _C._flush_temp()
    _C._flush_persistent()


# testing backward with operations all-in-one
# TODO: test non-contiguous view operations(e.g. transpose) throughly
@pytest.mark.parametrize(
    argnames="m,k,n",
    argvalues=product(
        [27, 57, 324, 678, 1091],
        [18, 39, 189, 782, 890],
        [11, 72, 209, 584, 999],
    )
)
def test_autograd(m: int, k: int, n: int):
    mm = _C.MatMul()
    add = _C.Add()
    div = _C.Div()
    sigmoid = _C.Sigmoid()
    relu = _C.LeakyRelu()
    sum = _C.Sum()
    ce_mean = _C.SoftmaxCrossEntropyMean()

    labels = torch.randint(0, n, (m,))  # random class per row
    y__ = torch.nn.functional.one_hot(labels, num_classes=n).float()
    torch_ce_mean = torch.nn.CrossEntropyLoss(reduction="mean")

    x_orig = np.random.randn(m, k).astype(np.float32)
    w1_orig = np.random.randn(k, k).astype(np.float32)
    w2_orig = np.random.randn(k, n).astype(np.float32)
    b2_orig = np.random.randn(n).astype(np.float32)
    y_orig = y__.detach().numpy()

    x_c = PyTensor(x_orig).to_c()
    w1_c = PyTensor(w1_orig).to_c()
    w2_c = PyTensor(w2_orig).to_c()
    b_c = PyTensor(b2_orig).to_c()
    y_c = PyTensor(y_orig).to_c()

    Autograd.set_backend(AutogradBackends.C)

    w1_c.requires_grad = True
    w2_c.requires_grad = True
    b_c.requires_grad = True

    # w1_c_ = sum.forward([w1_c], [2])  # dimension summation on params
    # w2_c_ = sum.forward([w2_c], [2])
    x_c = mm.forward([x_c, w1_c])  # matmuls
    x_c = mm.forward([x_c, w2_c])
    x_c = add.forward([x_c, b_c])  # broadcast add
    x_c = add.forward([x_c, b_c])  # repeated operations
    x_c = sigmoid.forward([x_c])  # activations
    x_c = relu.forward([x_c])
    x_c = div.forward([x_c], [2.0])
    x_c = ce_mean.forward([x_c, y_c])  # loss routine (CE)
    x_c.backward()

    x_torch = torch.from_numpy(x_orig)
    w1_torch = torch.from_numpy(w1_orig)
    w2_torch = torch.from_numpy(w2_orig)
    b_torch = torch.from_numpy(b2_orig)
    y_torch = torch.from_numpy(y_orig)
    w1_torch.requires_grad = True
    w2_torch.requires_grad = True
    b_torch.requires_grad = True

    # w1_torch_ = w1_torch.sum(dim=2)
    # w2_torch_ = w2_torch.sum(dim=2)
    x_torch = x_torch.mm(w1_torch)
    x_torch = x_torch.mm(w2_torch)
    x_torch = x_torch + b_torch
    x_torch = x_torch + b_torch
    x_torch = x_torch.sigmoid()
    x_torch = torch.nn.functional.leaky_relu(x_torch)
    x_torch = x_torch / 2
    x_torch = torch_ce_mean.forward(x_torch, y_torch)
    x_torch.backward()

    assert np.allclose(w1_c.grad.numpy(), w1_torch.grad.numpy(), atol=1e-4)
    # assert np.allclose(w2_c.grad.numpy(), w2_torch.grad.numpy(), atol=1e-4)
    # assert np.allclose(b_c.grad.numpy(), b_torch.grad.numpy(), atol=1e-4)

    _C._flush_temp()
