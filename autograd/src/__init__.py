from autograd.src.core import (
    AutogradEngine as Autograd,
    AutogradBackends,
    PyTensor,
)
from autograd.src.backend import ComputationalBackends
from autograd.src.constants import AutogradBackends

import autograd.src.functional as F
F.setup_functions(AutogradBackends.PYTHON)

from autograd.lib._C import Tensor, SGDOptimizer

Autograd.set_verbose(False)
Autograd.set_backend("py")  # or 'c'

__all__ = [
    "Autograd",
    "AutogradBackends",
    "ComputationalBackends",
    "PyTensor",
    "Tensor",
    "SGDOptimizer",
]
