from __future__ import annotations

import abc
import functools
import re
from array import array
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, TypeVar

import networkx as nx
import numpy as np
from IPython.display import SVG

from autograd.lib import _C
from autograd.src.backend import ComputationalBackends, Ops, dispatch
from autograd.src.constants import AutogradBackends
from autograd.src.dynamic import (
    ActivePatch,
    ActivePatches,
    DynamicFunctionPatcher,
    FunctionSpec,
)

NodeLike = TypeVar("NodeLike")
CB = ComputationalBackends


class PyTensor:
    def __init__(
        self,
        data: np.ndarray,
        comp_backend: ComputationalBackends = ComputationalBackends.NUMPY_CPU,
        requires_grad: bool = False,
    ):
        self.data = data
        self.comp_backend: ComputationalBackends = comp_backend
        self.requires_grad = requires_grad
        self.grad: PyTensor = None

        self.grad_fn: Function = None
        self._name: str = ""

    def to_c(self) -> _C.Tensor:
        """ In this case, the C++ backend does not owns the data.
            The lifetime of this Tensor's buffer memory is handled by Python's GC.
        """
        shape = list(self.data.shape)
        arr = array('f')
        arr.frombytes(self.data.astype(np.float32).tobytes())
        return _C.Tensor(arr, shape)

    def backward(self) -> None:
        if self.grad_fn is None:
            raise ValueError("'grad_fn' is None, can't call Autograd engine.")
        AutogradEngine._backward(self)


class Function(abc.ABC):
    _c: _C.Function

    def __init__(self):
        self._name: str = ""
        self._parents: list[PyTensor] = []

    def __init_subclass__(cls):
        super().__init_subclass__()
        AutogradEngine._hook(cls.forward, autograd_backend=AutogradBackends.PYTHON)

    def __call__(self, *args, **kwargs) -> PyTensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> PyTensor:
        raise NotImplementedError

    def backward(self, *args: PyTensor) -> tuple[PyTensor, ...]:
        raise NotImplementedError
    
    def c_forward(self, *args: PyTensor, **kwargs) -> PyTensor:
        # TODO: kwargs input 개선!!
        # routes arguments into C++backend
        if kwargs:
            return self._c.forward(list(args), kwargs["dims"])
        else:
            return self._c.forward(list(args))


@dataclass(slots=True)
class FunctionHook:
    target: FunctionSpec
    target_orig: FunctionSpec | None = None  # Not None, if chained-hook

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionHook):
            return NotImplemented
        return self.target == other.target  # i.e., is same addr

    def __hash__(self) -> int:
        return self.target.__hash__()

    def __repr__(self) -> str:
        return (
            f"Hook Fn     : {self.target.name}\n"
            f"     address: {self.target.addr}\n")


class FunctionHookRegistry(Dict[str, FunctionHook]):
    def __init__(self, name: str):
        super().__init__(self)
        self.__custom_name__ = name

    def __repr__(self) -> str:
        s = f"FunctionHookRegistry for '{self.__custom_name__}':\n"
        s += f"# registered: {len(self)}\n"
        s += "-" * 50 + "\n"
        s += "\n".join((hook.__repr__() for hook in self.values()))
        s += "\n" + "-" * 50
        return s
    

class ComputationalGraph:
    def __init__(self, graph: nx.DiGraph | None = None):
        # self.nodes: list[NodeLike] = []
        self._G = graph

    @property
    def nodes(self) -> list:
        return self._G.nodes
    
    @property
    def edges(self) -> list:
        return self._G.edges
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def add_edge(self, c: NodeLike, p: NodeLike, name="") -> None:
        # forward oriented DiGraph, so parent -> node when visualizing
        self._G.add_edge(p._name, c._name, name=name)

    def fuse(self) -> ComputationalGraph:
        from autograd.src.fusion import GraphFuser

        G = self._G.copy()
        G = GraphFuser.fuse(G)
        new_graph = ComputationalGraph(G)
        print(f"Number of nodes: {self.num_nodes} -> {new_graph.num_nodes}")
        print(f"Number of edges: {self.num_edges} -> {new_graph.num_edges}")
        print(
            "Forward graph optimization ratio: "
            f"{self.num_nodes / new_graph.num_nodes * 100:.2f}%"
        )
        print(
            "Backward graph optimization ratio: "
            f"{self.num_edges / new_graph.num_edges * 100:.2f}%"
        )
        return ComputationalGraph(G)

    def clear(self) -> None:
        if self._G is not None:
            self._G.clear()

    def show_init(self) -> None:
        self._G = nx.DiGraph()

    def draw(self) -> SVG:
        """ Using Graphviz 'dot' engine to draw a computation graph. """
        # TODO: support 'save_graph' path argument

        G: nx.DiGraph = self._G
        for n in G.nodes:
            n: str
            # stripping ptr ids for fused nodes
            name = "+".join(re.sub(r"_[0-9]+$", "", n_) for n_ in n.split("+"))
            if n.startswith("Param_"):  # Param leaf Tensors
                shape, fill = "ellipse", "#A5D6A7"
            elif n.startswith("Non-Param_"):  # Non-param leaf Tensors
                shape, fill = "ellipse", "#BDBDBD"
            else:  # Functions
                shape, fill = "box", "#90CAF9"
            G.nodes[n]["label"] = name
            G.nodes[n]["shape"] = shape
            G.nodes[n]["style"] = "filled"
            G.nodes[n]["fillcolor"] = fill
            G.nodes[n]["color"] = "black"
            G.nodes[n]["penwidth"] = "1.0"

        for u, v, data in G.edges(data=True):
            shape = data.get("name")
            if shape:
                G.edges[u, v]["label"] = str(shape)
            G.edges[u, v]["color"] = "#555555"
            G.edges[u, v]["fontsize"] = "10"
            G.edges[u, v]["fontcolor"] = "gray"
            G.edges[u, v]["penwidth"] = "1.2"

        A = nx.drawing.nx_agraph.to_agraph(G)
        A.graph_attr.update(
            rankdir="TB",
            splines="true",
            overlap="false",
            concentrate="true",
            ranksep="0.2",
            nodesep="0.1"  # base: 0.5
        )
        A.node_attr.update(fontsize="9", fontname="Helvetica")
        A.edge_attr.update(arrowsize="0.7", arrowhead="normal")
        A.layout(prog="dot")
        return SVG(A.draw(format="svg"))


class AutogradEngine:
    backend: AutogradBackends = None  # First initialized with 'py'
    verbose: bool = False
    _on: bool = False
    _track_graph: bool = False
    _dynamic_patcher: DynamicFunctionPatcher = DynamicFunctionPatcher()
    _active_patches: list[ActivePatch] = ActivePatches()
    _hooks_registry: dict[str, FunctionHook] = FunctionHookRegistry(name="Autograd")
    _graph = ComputationalGraph()
    _graphs: list[ComputationalGraph] = []
    _counter: Counter = Counter()

    @classmethod
    def _hook(cls, fwd: Callable, autograd_backend: str):
        fwd_spec = cls._dynamic_patcher.capture(fwd)
        hook = FunctionHook(
            target=fwd_spec,
            target_orig=fwd_spec,
        )
        cls._hooks_registry[hook.target.name] = hook

    @classmethod
    def _apply_autograd(cls, hook: FunctionHook) -> Callable:
        fwd: Callable = hook.target.fn_unbound

        if cls.backend == AutogradBackends.PYTHON:
            # In this case, fwd == 'Function.forward' (original)
            @functools.wraps(fwd)
            def autograd_hook(*args, **kwargs) -> PyTensor:
                out: PyTensor = fwd(*args, **kwargs)
                new_fn: Function = args[0].__class__()
                # cache input tensors for backward
                new_fn._parents = args[1:]
                if cls._track_graph:  # parse graph if visual mode
                    cls._add_to_graph(new_fn, *args[1:])
                out.grad_fn = new_fn
                return out
        
        elif cls.backend == AutogradBackends.C:
            # in this case, replacement proceeds:
            #   'MyFunction.forward' -> 'MyFunction.c_forward'
            # In order to route ops args into C++ backend.
            autograd_hook = Function.c_forward

        # #! Be careful when modifying wrapper's __qualname__:
        #   -> Then the patched spec won't able to find correct owner when desired.
        #   -> Only use when there are no such situations.
        autograd_hook.__qualname__ += f"_with_{cls.backend}_autograd"

        return autograd_hook

    @classmethod
    def _set_node_name(cls, node: NodeLike) -> None:
        if isinstance(node, PyTensor):
            name = "Param" if node.requires_grad else "Non-param"
        else:
            name = node.__class__.__name__
        cls._counter[name] += 1
        node._name = f"{name}_{cls._counter[name]}"

    @classmethod
    def _add_to_graph(cls, fn: Function, *arg_tensors: PyTensor) -> None:
        cls._set_node_name(fn)
        for tensor in arg_tensors:
            if tensor.grad_fn is None:  # is leaf tensor, add to graph for convenience
                cls._set_node_name(tensor)
                cls._graph.add_edge(c=fn, p=tensor)  # i.e., leaf tensor -> fn
            else:  # is non-leaf tensor
                cls._graph.add_edge(c=fn, p=tensor.grad_fn)

    @classmethod
    def _backward(cls, root: PyTensor) -> None:
        init_grad = dispatch(Ops.ones, to_backend=root.comp_backend)(shape=[1])
        grad = PyTensor(data=init_grad, backend=root.comp_backend)
        
        stack = [(grad, root.grad_fn)]
        while stack:
            grad, fn = stack.pop()  # fn: node, grad: incoming grad to node
            # compute grad output of current node 
            grads = fn.backward(grad)
            # for each grad output and current node's parent tensors
            for grad, parent_tensor in zip(grads, fn._parents):
                # parent is non-leaf tensor
                if parent_tensor.grad_fn:
                    stack.append((grad, parent_tensor.grad_fn))
                    assert not parent_tensor.requires_grad
                # parent is leaf tensor and also 'is-param'
                if parent_tensor.requires_grad:
                    if parent_tensor.grad is None:  # if grad not initialized
                        zeros_op = dispatch(Ops.zeros_like, to_backend=grad.comp_backend)
                        parent_tensor.grad = PyTensor(
                            zeros_op(parent_tensor.data),
                            comp_backend=grad.comp_backend,
                        )  # initialize and accumulate
                        grad_acc_op = dispatch(Ops.add_inplace, to_backend=grad.comp_backend)
                        grad_acc_op(parent_tensor.grad.data, grad.data)
        cls._graph.clear()
        cls._counter.clear()

    @classmethod
    def set_backend(cls, backend: str = AutogradBackends.PYTHON) -> None:
        from autograd.src.functional import setup_functions

        if cls.backend != backend:
            cls.backend = backend
            setup_functions(backend)
            # Reboot engine
            cls.off()
            cls.on()

        if cls.verbose:
            print(f"Set Autograd backend as '{backend}'.")

    @classmethod
    def on(cls) -> None:
        if cls._on:
            return
        for hook in cls._hooks_registry.values():
            wrapped = cls._apply_autograd(hook)
            try:
                ap = cls._dynamic_patcher.patch(hook.target, wrapped)
                cls._active_patches.append(ap)
            except Exception as exc:
                print(exc)
        cls._on = True

    @classmethod
    def off(cls) -> None:
        if not cls._on:
            return
        cls._dynamic_patcher.unpatch(cls._active_patches)
        cls._active_patches.clear()
        cls._graph.clear()
        cls._on = False

    @classmethod
    def set_track_graph(cls, flag: bool) -> None:
        cls._track_graph = flag
        _C.AutogradEngine.track_graph(flag)
        if flag:
            cls._graph.show_init()
        else:
            if cls._graph is not None:
                cls._graph.clear()

    @classmethod
    def get_computation_graph(cls, root: PyTensor | _C.Tensor = None) -> ComputationalGraph:
        if cls.backend == "c":
            for n1, n2, edge_name in _C.AutogradEngine.get_graph().edges:
                cls._graph._G.add_edge(n1, n2, name=edge_name)
            # TODO: use input 'root' tensor for graph root
        return cls._graph

    @classmethod
    def finish_graph(cls) -> None:
        curr_graph = ComputationalGraph()
        curr_graph._G = deepcopy(cls.get_computation_graph()._G)
        cls._graphs.append(curr_graph)
        cls._graph.clear()
        _C.AutogradEngine.clear_graph()
    
    @classmethod
    def clear_graph(cls) -> None:
        cls._graph.clear()

    @classmethod
    def draw_computation_graph(cls, root: PyTensor | _C.Tensor = None):
        # TODO: use input 'root' tensor for graph root
        return cls.get_computation_graph(root).draw()

    @classmethod
    def set_verbose(cls, flag: bool) -> None:
        cls.verbose = flag  # TODO: apply logger & log_level
        cls._dynamic_patcher.verbose = cls.verbose

    class no_grad:
        def __enter__(self) -> None:
            AutogradEngine.off()
            _C.AutogradEngine.on(False)

        def __exit__(self, *args: object) -> None:
            AutogradEngine.on()
            _C.AutogradEngine.on(True)
