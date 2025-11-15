from .lib._C import Tensor, randn
from .src import Autograd, AutogradBackends, PyTensor, SGDOptimizer
from .src import functional as Functional
from .src.utils import from_numpy, from_tensor

Autograd.set_backend(AutogradBackends.C)


__all__ = [
    "Autograd",
    "AutogradBackends",
    "PyTensor",
    "Tensor",
    "Functional",
    "SGDOptimizer",
    "from_numpy",
    "from_tensor",
    "randn",
]
