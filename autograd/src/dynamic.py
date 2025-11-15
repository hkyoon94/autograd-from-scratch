from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class CustomDunderAttrs(StrEnum):
    __patched__ = "__patched__"


def addr(fn: Callable) -> str:
    return hex(id(fn))


def get_unbound(fn: Callable) -> Callable:
    if inspect.ismethod(fn):
        return fn.__func__
    if isinstance(fn, (classmethod, staticmethod)):
        return fn.__func__
    return fn


def get_module_qualname(fn: Callable) -> str:
    return f"{fn.__module__}/{fn.__qualname__}"


def format_patch(fn_old: Callable, fn_new: Callable) -> str:
    desc = f"{fn_old.__qualname__} -> {fn_new.__qualname__}\n"
    old_addr = f"    FROM: {addr(fn_old)}(unbound: {addr(get_unbound(fn_old))})\n"
    new_addr = f"    TO  : {addr(fn_new)}(unbound: {addr(get_unbound(fn_new))})"
    return desc + old_addr + new_addr


@dataclass
class FunctionSpec:
    fn_unbound: Callable
    name: str
    addr: str

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FunctionSpec):
            return self.addr == other.addr
        if callable(other):
            return self.addr == addr(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return (
            f"FunctionSpec __qualname__: {self.name}\n"
            f"             address     : {self.addr}"
        )

    def find_owner(self) -> Any:
        mod = sys.modules[self.fn_unbound.__module__]
        parts = self.fn_unbound.__qualname__.split(".")
        owner = mod
        for p in parts[:-1]:
            owner = getattr(owner, p)
        return owner

    @classmethod
    def create(cls, fn: Callable):
        fn_unbound = get_unbound(fn)
        return cls(
            fn_unbound=fn_unbound,
            name=get_module_qualname(fn),
            addr=hex(id(fn_unbound)),
        )
    

class FunctionSpecs(set):
    def __repr__(self) -> str:
        return "\n".join(str(fs) for fs in self)


@dataclass
class ActivePatch:
    """ Represents an active patch.
        Stores the original / new function that was patched and
        an unwrapper function that restores the original definition.
    """
    owner: Any
    name: str
    orig_spec: FunctionSpec
    new_spec: FunctionSpec
    old: Callable
    new: Callable

    @classmethod
    def create(
        cls,
        owner: Any,
        name: str,
        orig_spec: FunctionSpec,
        new_spec: FunctionSpec,
        old: Callable,
        new: Callable,
    ) -> ActivePatch:
        old_unbound = get_unbound(old)
        if orig_spec.addr != addr(old_unbound):
            raise FunctionPatchFailError(
                f"ERROR: unbound(self.old): {old_unbound} "
                f"does not match 'orig_spec.addr': {orig_spec.addr}"
            )
        return cls(owner, name, orig_spec, new_spec, old, new)

    def unpatch(self) -> None:
        setattr(self.owner, self.name, self.old)

    def __repr__(self) -> str:
        return format_patch(self.old, self.new)
    

class ActivePatches(list):
    def __repr__(self) -> str:
        return "\n".join(str(ap) for ap in self)


class DynamicFunctionPatcher:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @staticmethod
    def _resolve_descriptors(
        owner: Any, orig_spec: FunctionSpec, fn_new: Callable
    ) -> Callable | classmethod | staticmethod:
        fn_orig_withdesc = owner.__dict__[orig_spec.fn_unbound.__name__]
        if isinstance(fn_orig_withdesc, classmethod):
            return fn_orig_withdesc, classmethod(fn_new)
        elif isinstance(fn_orig_withdesc, staticmethod):
            return fn_orig_withdesc, staticmethod(fn_new)
        return fn_orig_withdesc, fn_new

    def capture(self, fn: Callable) -> FunctionSpec:
        return FunctionSpec.create(fn=fn)

    def patch(
        self, spec: FunctionSpec, fn_new: Callable, overwrite_existing_patch: bool = True
    ) -> ActivePatch:
        if getattr(
            spec.fn_unbound, CustomDunderAttrs.__patched__, False
        ) and not overwrite_existing_patch:
            raise FunctionPatchFailError(
                f"ERROR: Function {spec.fn_unbound} already been patched. "
                "Set 'overwrite_existing_patch=True' to overwrite."
            )

        fn_new = get_unbound(fn_new)
        if addr(fn_new) == spec.addr:
            print(f"WARNING: Received identical functions, {spec.fn_unbound} -> {fn_new}.")

        owner = spec.find_owner()
        if owner is None:
            # NOTE: proceeds to fallback, no patching & no unwrapping.
            raise FunctionPatchFailError(
                f"ERROR: Finding owner of function "
                f"'{spec.fn_unbound}' couldn't be resolved"
            )

        # Resolving original descriptors
        fn_orig_withdesc, fn_new_withdesc = self._resolve_descriptors(owner, spec, fn_new)

        try:  # Patching fn_new
            setattr(owner, spec.fn_unbound.__name__, fn_new_withdesc)
        except Exception:
            # NOTE: built-in type's dunder methods cannot be patched.
            raise FunctionPatchFailError(
                f"Cannot patch '{fn_new_withdesc}' into '{owner}' of type '{type(owner)}'"
            )

        # Applying __patched__ flag to unbounded original fn
        setattr(spec.fn_unbound, CustomDunderAttrs.__patched__, True)
        if self.verbose:
            print(f"Patching successful: {format_patch(fn_orig_withdesc, fn_new_withdesc)}")
        try:
            new_spec = FunctionSpec.create(fn=fn_new_withdesc)
            return ActivePatch.create(
                owner=owner,
                name=spec.fn_unbound.__name__,
                orig_spec=spec,
                new_spec=new_spec,
                old=fn_orig_withdesc,
                new=fn_new_withdesc,
            )
        except FunctionPatchFailError:
            raise

    def unpatch(self, active_patches: list[ActivePatch] | ActivePatch) -> None:
        if not isinstance(active_patches, list):
            active_patches = [active_patches]
        for ap in reversed(active_patches):
            try:
                ap.unpatch()
                if self.verbose:
                    print(f"Roll-back successful: {format_patch(ap.new, ap.old)}")
            except Exception:
                raise FunctionPatchFailError(f"ERROR: Cannot rollback {ap.new} -> {ap.old}")
        active_patches.clear()


class FunctionPatchFailError(Exception):
    """ Raised when a function cannot be patched.
        This exception indicates that the patching operation
        failed due to missing owner resolution, attempting to
        patch a built-in type, or other assignment issues.
    """

    def __init__(self, message: str = ""):
        super().__init__(message)


if __name__ == "__main__":
    """ This script describes the usage of DynamicFunctionPatcher and how it works """

    import functools
    import inspect
    from typing import Callable

    from autograd.src.dynamic import (
        ActivePatch,
        FunctionPatchFailError,
        FunctionSpec,
        DynamicFunctionPatcher,
    )

    patcher = DynamicFunctionPatcher(verbose=True)
    registry: list[FunctionSpec] = []
    active_patches: list[ActivePatch] = []

    def capture(f):
        spec = patcher.capture(f)
        if spec is not None:
            registry.append(spec)
        return f

    def decorate(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            print("This function is modified:")
            return f(*args, **kwargs)
        return wrapped

    class A:  # For testing bound method
        @capture
        def bound_method(self, a):
            return a

    class B:  # For testing static method
        @capture
        @staticmethod
        def static_method(a):
            return a

    class C:  # For testing class method
        # @capture
        @classmethod
        def class_method(cls, a):
            return a
        
    class D:  # For testing dunder method
        @capture
        def __call__(self, a):
            return a

    @capture  # For testing plain module function
    def f(a):
        return a

    a_instance = A()
    b_instance = B()
    c_instance = C()
    d_instance = D()
    capture(a_instance.bound_method)  # double captured
    capture(b_instance.static_method)  # double captured
    capture(c_instance.class_method)
    capture(d_instance.__call__)  # double captured

    print("Captured Functions: ")
    for reg in registry:
        print(f"{reg}")

    print(f(" 'f1'"))
    print(A().bound_method(" bound method 'A().f2'"))
    print(B.static_method(" static method 'B.f2'"))
    print(C.class_method(" class method 'C.f2'"))
    print(D()(" dunder method 'D.__call__()'"))
    print(a_instance.bound_method(" 'a_instance.f2'"))
    print(b_instance.static_method(" 'b_instance.f2'"))
    print(c_instance.class_method(" 'c_instance.f2'"))
    print(d_instance(" 'd_instance.__call__()'"))

    for spec in registry:
        try:
            wrapped = decorate(spec.fn_unbound)
            ap = patcher.patch(spec, wrapped)
            active_patches.append(ap)
        except FunctionPatchFailError as exc:
            print(exc)
    for spec in registry:
        print(f"{spec.fn_unbound}.__patched__: {spec.fn_unbound.__patched__}")

    print(f(" 'f1'"))
    print(A().bound_method(" bound method 'A().f2'"))
    print(B.static_method(" static method 'B.f2'"))
    print(C.class_method(" class method 'C.f2'"))
    print(D()(" dunder method 'D.__call__()'"))
    print(a_instance.bound_method(" 'a_instance.f2'"))
    print(b_instance.static_method(" 'b_instance.f2'"))
    print(c_instance.class_method(" 'c_instance.f2'"))
    print(d_instance(" 'd_instance.__call__()'"))

    patcher.unpatch(active_patches)

    print(f(" 'f1'"))
    print(A().bound_method(" bound method 'A().f2'"))
    print(B.static_method(" static method 'B.f2'"))
    print(C.class_method(" class method 'C.f2'"))
    print(D()(" dunder method 'D.__call__()'"))
    print(a_instance.bound_method(" 'a_instance.f2'"))
    print(b_instance.static_method(" 'b_instance.f2'"))
    print(c_instance.class_method(" 'c_instance.f2'"))
    print(d_instance(" 'd_instance.__call__()'"))
