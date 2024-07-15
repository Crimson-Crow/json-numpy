from __future__ import annotations

__version__ = "2.1.0"
__all__ = ["default", "object_hook", "dumps", "loads", "dump", "load", "patch"]

import json
from base64 import b64decode, b64encode
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from numpy import frombuffer, generic, ndarray
from numpy.lib.format import descr_to_dtype, dtype_to_descr

if TYPE_CHECKING:  # pragma: no cover
    from _typeshed import SupportsRead


def default(
    o: Any, *, fallback_default: Callable[[Any], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Encodes numpy objects to a JSON-serializable dictionary.

    Args:
        o (object): The object to encode.
        fallback_default (Callable[[Any], dict[str, Any]] | None): A fallback encoder function to handle objects that are not numpy objects.

    Returns:
        dict[str, Any]: The JSON-serializable dictionary representation of the numpy object, or the result of the fallback encoder if present and if the object is not a numpy object.

    Raises:
        TypeError: If the object is not JSON serializable.
    """
    if isinstance(o, (ndarray, generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }

    if fallback_default is not None:
        return fallback_default(o)

    msg = f"Object of type {o.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)


def object_hook(dct: dict) -> dict | ndarray | generic:
    """Custom object hook function for decoding JSON objects into numpy arrays.

    Args:
        dct (dict): The dictionary to decode.

    Returns:
        dict | np.ndarray | np.generic: The decoded numpy object or the original dictionary.
    """
    if "__numpy__" in dct:
        np_obj = frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct


_default = default
_hook = object_hook
_dumps = json.dumps
_loads = json.loads
_dump = json.dump
_load = json.load


def _patch_encoder(
    *args: Any,
    default: Callable[[Any], Any] | None = None,
    user_cls: type[json.JSONEncoder] | None = None,
    **kwargs: Any,
) -> json.JSONEncoder:
    """Ensures cooperation with the provided `default` and/or `cls` by manipulating the JSONEncoder."""
    if user_cls is None:
        user_cls = json.JSONEncoder
    elif default is None:
        encoder = user_cls(*args, **kwargs)
        encoder.default = partial(_default, fallback_default=encoder.default)  # type: ignore[method-assign]
        return encoder
    return user_cls(
        *args, default=partial(_default, fallback_default=default), **kwargs
    )


def dumps(*args: Any, cls: type[json.JSONEncoder] | None = None, **kwargs: Any) -> str:
    kwargs["user_cls"] = cls
    return _dumps(*args, cls=_patch_encoder, **kwargs)  # type: ignore[arg-type]


def loads(
    *args: Any, object_hook: Callable[[dict], Any] | None = None, **kwargs: Any
) -> Any:
    return _loads(
        *args,
        object_hook=_hook
        if object_hook is None
        else lambda dct: _hook(object_hook(dct)),
        **kwargs,
    )


def dump(*args: Any, cls: type[json.JSONEncoder] | None = None, **kwargs: Any) -> None:
    kwargs["user_cls"] = cls
    return _dump(*args, cls=_patch_encoder, **kwargs)  # type: ignore[arg-type]


def load(fp: SupportsRead[str | bytes], **kwargs: Any) -> Any:
    return loads(fp.read(), **kwargs)


def patch() -> None:
    """Monkey patch json module to support encoding/decoding NumPy arrays/scalars."""
    json.dumps = dumps
    json.loads = loads
    json.dump = dump
    json.load = load
