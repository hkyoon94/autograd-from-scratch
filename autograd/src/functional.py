from autograd.src.backend import Ops, dispatch
from autograd.src.core import Function, PyTensor
from autograd.lib import _C


class Sum(Function):
    _c = _C.Sum()

    def __init__(self):
        self.x_cache: PyTensor = None
        self._parents = []

    def forward(self, x: PyTensor) -> PyTensor:
        op = dispatch(Ops.sum, to_backend=x.comp_backend)
        return PyTensor(op(x.data), backend=x.comp_backend)

    def backward(self, grad: PyTensor) -> tuple[PyTensor, ...]:
        op = dispatch(Ops.sum_backward, grad.comp_backend)
        return (
            PyTensor(op(grad.data, self._parents[0].data), backend=grad.comp_backend),
        )


class MatMul(Function):
    _c = _C.MatMul()

    def __init__(self):
        self.a_cache: PyTensor = None
        self.b_cache: PyTensor = None
        self._parents = []

    def forward(self, a: PyTensor, b: PyTensor) -> PyTensor:
        op = dispatch(Ops.mm, to_backend=a.comp_backend)
        return PyTensor(op(a.data, b.data), backend=a.comp_backend)

    def backward(self, grad: PyTensor) -> tuple[PyTensor, ...]:
        op = dispatch(Ops.mm_backward, to_backend=grad.comp_backend)
        return (
            PyTensor(r, backend=grad.comp_backend)
            for r in op(grad.data, *self._parents)
        )



def setup_functions(backend: str) -> None:
    from autograd.src.constants import AutogradBackends
    from autograd.src.core import AutogradEngine

    global view, div, sum, add, mm, sigmoid, relu, softmax_ce_mean

    if backend == AutogradBackends.PYTHON:
        sum = Sum()
        mm = MatMul()
        if AutogradEngine.verbose:
            print("Functions now run in python backend")

    elif backend == AutogradBackends.C:
        view = _C.View()
        div = _C.Div()
        sum = _C.Sum()
        add = _C.Add()
        mm = _C.MatMul()
        sigmoid = _C.Sigmoid()
        relu = _C.LeakyRelu()
        softmax_ce_mean = _C.SoftmaxCrossEntropyMean()
        if AutogradEngine.verbose:
            print("Functions now run in c backend")
