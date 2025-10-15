from itertools import product

import numpy as np
import pytest
import torch

from autograd.lib import _C
from autograd.src import PyTensor

ATOL = 1e-2


# testing sum
@pytest.mark.parametrize(
    argnames="m,n",
    argvalues=product(
        [120, 898],
        [23, 18],
    ),
)
def test_sum1(m: int, n: int):
    a_ = torch.randn(m, n, requires_grad=True)

    # Reference Torch API
    c_ = a_.sum(dim=0)

    a_np = a_.detach().numpy()
    a = PyTensor(a_np).to_c()
    a.requires_grad = True

    c = _C.sum(a, [0])

    assert np.allclose(c_.detach().numpy(), c.numpy(), atol=ATOL)

    grad_ = torch.randn_like(c_)

    c_.backward(gradient=grad_)

    grad = PyTensor(grad_.detach().numpy()).to_c()
    a_grad, = _C.sum_backward(grad, a)
    
    assert np.allclose(a_.grad.detach().numpy(), a_grad.numpy(), atol=ATOL)
    _C._flush_temp()


# testing add
@pytest.mark.parametrize(
    argnames="m,n",
    argvalues=product(
        [120, 898],
        [23, 18],
    ),
)
def test_add(m: int, n: int):
    a_ = torch.randn(m, n, requires_grad=True)
    b_ = torch.randn(m, n, requires_grad=True)

    # Reference Torch API
    c_ = a_ + b_

    a_np = a_.detach().numpy()
    b_np = b_.detach().numpy()
    a = PyTensor(a_np).to_c()
    b = PyTensor(b_np).to_c()
    a.requires_grad = True
    b.requires_grad = True

    c = _C.add(a, b)

    assert np.allclose(c_.detach().numpy(), c.numpy(), atol=ATOL)

    grad_ = torch.randn_like(c_)

    c_.backward(gradient=grad_)

    grad = PyTensor(grad_.detach().numpy()).to_c()
    a_grad, b_grad = _C.add_backward(grad, a, b)

    assert np.allclose(a_.grad.detach().numpy(), a_grad.numpy(), atol=ATOL)
    assert np.allclose(b_.grad.detach().numpy(), b_grad.numpy(), atol=ATOL)
    _C._flush_temp()


# testing add_broadcast
@pytest.mark.parametrize(
    argnames="m,n",
    argvalues=product(
        [120, 898],
        [23, 18],
    ),
)
def test_add_broadcast(m: int, n: int):
    a_ = torch.randn(m, n, requires_grad=True)
    b_ = torch.randn(n, requires_grad=True)

    # Reference Torch API
    c_ = a_ + b_

    a_np = a_.detach().numpy()
    b_np = b_.detach().numpy()
    a = PyTensor(a_np).to_c()
    b = PyTensor(b_np).to_c()
    a.requires_grad = True
    b.requires_grad = True

    c = _C.add(a, b)

    assert np.allclose(c_.detach().numpy(), c.numpy(), atol=ATOL)

    grad_ = torch.randn_like(c_)

    c_.backward(gradient=grad_)

    grad = PyTensor(grad_.detach().numpy()).to_c()
    a_grad, b_grad = _C.add_backward(grad, a, b)

    assert np.allclose(a_.grad.detach().numpy(), a_grad.numpy(), atol=ATOL)
    assert np.allclose(b_.grad.detach().numpy(), b_grad.numpy(), atol=ATOL)
    _C._flush_temp()


# testing mm
@pytest.mark.parametrize(
    argnames="m,k,n",
    argvalues=product(
        [27, 57, 324, 678, 1091],
        [18, 39, 189, 782, 890],
        [11, 72, 209, 584, 999],
    )
)
def test_mm(m: int, k: int, n: int):
    a_ = torch.randn(m, k, requires_grad=True)
    b_ = torch.randn(k, n, requires_grad=True)
    
    # Reference Torch API
    c_ = a_.mm(b_)
    
    a_np = a_.detach().numpy()
    b_np = b_.detach().numpy()
    a = PyTensor(a_np).to_c()
    b = PyTensor(b_np).to_c()
    a.requires_grad = True
    b.requires_grad = True
    
    c = _C.mm(a, b)

    assert np.allclose(c_.detach().numpy(), c.numpy(), atol=ATOL)
    
    # autodiff with torch
    grad_ = torch.randn_like(c_)
    c_.backward(grad_)
    
    # manual diff with ops::op::mm~
    grad = PyTensor(grad_.numpy()).to_c()
    a_grad, b_grad = _C.mm_backward(grad, a, b)

    assert np.allclose(a_.grad.detach().numpy(), a_grad.numpy(), atol=ATOL)
    assert np.allclose(b_.grad.detach().numpy(), b_grad.numpy(), atol=ATOL)
    _C._flush_temp()


# testing sigmoid
@pytest.mark.parametrize(
    argnames="m, k",
    argvalues=product(
        [27, 57, 324, 678, 1091],
        [18, 39, 189, 782, 890],
    )
)
def test_sigmoid(m: int, k: int):
    a_ = torch.randn(m, k, requires_grad=True)
    
    # Reference Torch API
    c_ = a_.sigmoid()
    
    a_np = a_.detach().numpy()
    a = PyTensor(a_np).to_c()

    a.requires_grad = True

    c = _C.sigmoid(a)

    assert np.allclose(c_.detach().numpy(), c.numpy(), atol=ATOL)
    
    # autodiff with torch
    grad_ = torch.randn_like(c_)
    c_.backward(grad_)
    
    # manual diff with ops::op::mm~
    grad = PyTensor(grad_.numpy()).to_c()
    a_grad, = _C.sigmoid_backward(grad, a)

    assert np.allclose(a_.grad.detach().numpy(), a_grad.numpy(), atol=ATOL)
    _C._flush_temp()


# testing leaky-relu
@pytest.mark.parametrize(
    argnames="m, k",
    argvalues=product(
        [27, 57, 324, 678, 1091],
        [18, 39, 189, 782, 890],
    )
)
def test_leaky_relu(m: int, k: int):
    a_ = torch.randn(m, k, requires_grad=True)
    
    # Reference Torch API
    c_ = torch.nn.functional.leaky_relu(a_, negative_slope=0.01)
    
    a_np = a_.detach().numpy()
    a = PyTensor(a_np).to_c()

    a.requires_grad = True

    c = _C.leaky_relu(a)

    assert np.allclose(c_.detach().numpy(), c.numpy(), atol=ATOL)
    
    # autodiff with torch
    grad_ = torch.randn_like(c_)
    c_.backward(grad_)
    
    # manual diff with ops::op::mm~
    grad = PyTensor(grad_.numpy()).to_c()
    a_grad, = _C.leaky_relu_backward(grad, a)

    assert np.allclose(a_.grad.detach().numpy(), a_grad.numpy(), atol=ATOL)
    _C._flush_temp()


# testing Softmax-CrossEntropy-Mean
@pytest.mark.parametrize(
    argnames="m, k",
    argvalues=product(
        [27, 57, 324, 678, 1091],
        [18, 39, 189, 782, 890],
    )
)
def test_softmax_ce_mean(m: int, k: int):
    a_ = torch.randn(m, k, requires_grad=True)

    labels = torch.randint(0, k, (m,))  # random class per row
    y_ = torch.nn.functional.one_hot(labels, num_classes=k).float()

    # Reference Torch API
    crit = torch.nn.CrossEntropyLoss(reduction="mean")

    a_np = a_.detach().numpy()
    y_np = y_.detach().numpy()
    a = PyTensor(a_np).to_c()
    y = PyTensor(y_np).to_c()

    a.requires_grad = True

    c_ = crit.forward(a_, y_)
    sm_ = torch.softmax(a_, dim=-1)

    c, sm = _C.ce_softmax_mean(a, y)

    assert np.allclose(c_.detach().numpy(), c.numpy(), atol=ATOL)
    assert np.allclose(sm_.detach().numpy(), sm.numpy(), atol=ATOL)

    # autodiff with torch
    grad_ = np.random.random()  # random number
    c_.backward(torch.tensor(grad_))

    # manual diff with ops::op::mm~
    grad = _C.Tensor([1], grad_)
    a_grad, = _C.ce_softmax_mean_backward(grad, sm, y)

    assert np.allclose(a_.grad.detach().numpy(), a_grad.numpy(), atol=ATOL)
    _C._flush_temp()
