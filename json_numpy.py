from __future__ import annotations

__version__ = "2.0.0"
__all__ = ["default", "object_hook", "dumps", "loads", "dump", "load", "patch"]

import json
from base64 import b64decode, b64encode
from typing import Any

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr


def default(obj: object) -> dict[str, Any]:
    if isinstance(obj, (np.ndarray, np.generic)):
        return {
            "__numpy__": b64encode(
                obj.data if obj.flags.c_contiguous else obj.tobytes()
            ).decode(),
            "dtype": dtype_to_descr(obj.dtype),
            "shape": obj.shape,
        }
    msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(msg)


def object_hook(dct: dict[Any, Any]) -> dict[Any, Any] | np.ndarray | np.generic:
    if "__numpy__" in dct:
        np_obj = np.frombuffer(
            b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"])
        )
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct


_dumps = json.dumps
_loads = json.loads
_dump = json.dump
_load = json.load


def dumps(*args: Any, **kwargs: Any) -> str:
    kwargs.setdefault("default", default)
    return _dumps(*args, **kwargs)


def loads(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("object_hook", object_hook)
    return _loads(*args, **kwargs)


def dump(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("default", default)
    return _dump(*args, **kwargs)


def load(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("object_hook", object_hook)
    return _load(*args, **kwargs)


def patch() -> None:
    """Monkey patch json module to support encoding/decoding NumPy arrays/scalars."""
    json.dumps = dumps
    json.loads = loads
    json.dump = dump
    json.load = load
