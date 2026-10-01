"""`--adapter module:Class` resolution.

`check_spec` is a string-only check safe to run in the evaluator process.
`load_adapter` imports contestant code and must only run in the child.
"""

import importlib
import inspect
import sys
from pathlib import Path


class LoaderError(Exception):
    pass


def check_spec(spec: str, allow_private: bool = False) -> tuple:
    module_name, sep, class_name = spec.partition(":")
    if not sep or not module_name or not class_name:
        raise LoaderError(f"adapter must be 'module:Class', got {spec!r}")
    parts = module_name.split(".")
    if not all(p.isidentifier() for p in parts) or not class_name.isidentifier():
        raise LoaderError(f"invalid adapter spec {spec!r}")
    if "private" in parts and not allow_private:
        raise LoaderError("adapters under private/ are not allowed")
    return module_name, class_name


def load_adapter(spec: str, bench_dir, allow_private: bool = False) -> type:
    module_name, class_name = check_spec(spec, allow_private)
    bench_dir = Path(bench_dir).resolve()
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise LoaderError(f"cannot import {module_name!r}: {exc}") from exc

    module_file = Path(getattr(module, "__file__", "") or "").resolve()
    if not allow_private and (bench_dir / "private") in module_file.parents:
        raise LoaderError("adapters under private/ are not allowed")

    cls = getattr(module, class_name, None)
    if cls is None:
        raise LoaderError(f"{module_name!r} has no {class_name!r}")
    if not inspect.isclass(cls):
        raise LoaderError(f"{spec!r} is not a class")
    if cls.__module__ != module.__name__:
        raise LoaderError(
            f"{spec!r} is defined in {cls.__module__!r}, not {module_name!r}; "
            "name the module that defines your class")
    if inspect.isabstract(cls):
        raise LoaderError(f"{spec!r} is abstract; implement solve()")
    if not callable(getattr(cls, "solve", None)):
        raise LoaderError(f"{spec!r} has no solve() method")
    return cls
