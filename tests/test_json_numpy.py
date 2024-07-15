from __future__ import annotations

import json
import unittest
import uuid
from io import StringIO
from typing import Any, TypeVar, cast

import numpy as np
from numpy.testing import assert_array_equal, assert_equal

import json_numpy

T = TypeVar("T")


class NumpyJsonSerializationTest(unittest.TestCase):
    def setUp(self) -> None:
        json_numpy.patch()

    @staticmethod
    def dumps_loads(x: T) -> T:
        return cast(T, json.loads(json.dumps(x)))

    @staticmethod
    def assert_equal_with_type(
        actual: np.ndarray | np.generic | list[Any] | dict[Any, Any],
        desired: np.ndarray | np.generic | list[Any] | dict[Any, Any],
        sort_key: Any = None,
    ) -> None:
        if isinstance(desired, np.ndarray) and isinstance(actual, np.ndarray):
            assert_array_equal(actual, desired)
            assert_equal(actual.dtype, desired.dtype)
        elif isinstance(desired, list) and isinstance(actual, list):
            assert_array_equal(actual, desired)
            assert_array_equal([type(e) for e in actual], [type(e) for e in desired])
        elif isinstance(desired, dict) and isinstance(actual, dict):
            assert_array_equal(
                sorted(actual.values(), key=sort_key),
                sorted(desired.values(), key=sort_key),
            )
            assert_array_equal(
                [type(e) for e in sorted(actual.values(), key=sort_key)],
                [type(e) for e in sorted(desired.values(), key=sort_key)],
            )
            assert_array_equal(sorted(actual.keys()), sorted(desired.keys()))
            assert_array_equal(
                [type(e) for e in sorted(actual.keys())],
                [type(e) for e in sorted(desired.keys())],
            )
        else:
            assert_equal(actual, desired)
            assert_equal(type(actual), type(desired))

    def test_dump_load(self) -> None:
        x = [np.float32(np.random.rand()) for _ in range(5)]
        buff = StringIO()
        json.dump(x, buff)
        buff.seek(0)
        self.assert_equal_with_type(json.load(buff), x)

    def test_typeerror_on_cannot_encode(self) -> None:
        self.assertRaises(TypeError, json.dumps, b"abc")

    def test_numpy_scalar_bool(self) -> None:
        for b in (True, False):
            x = np.bool_(b)
            self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_scalar_float(self) -> None:
        x = np.float32(np.random.rand())
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_scalar_complex(self) -> None:
        x = np.complex64(np.random.rand() + 1j * np.random.rand())
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_bool(self) -> None:
        x = [np.bool_(True), np.bool_(False)]  # noqa: FBT003
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_float(self) -> None:
        x = [np.float32(np.random.rand()) for _ in range(5)]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_complex(self) -> None:
        x = [np.complex64(np.random.rand() + 1j * np.random.rand()) for _ in range(5)]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_float_complex(self) -> None:
        x = [np.float32(np.random.rand()) for _ in range(5)] + [
            np.complex128(np.random.rand() + 1j * np.random.rand()) for _ in range(5)
        ]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_mixed(self) -> None:
        x = [1.0, np.float32(3.5), np.complex128(4.25), "foo"]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_dict_numpy_float(self) -> None:
        x = {"foo": np.float32(1.0), "bar": np.float32(2.0)}
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_dict_numpy_complex(self) -> None:
        x = {"foo": np.complex128(1.0 + 1.0j), "bar": np.complex128(2.0 + 2.0j)}
        self.assert_equal_with_type(self.dumps_loads(x), x, sort_key=np.linalg.norm)

    def test_numpy_array_float(self) -> None:
        x = np.random.rand(5).astype(np.float32)
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_complex(self) -> None:
        x = (np.random.rand(5) + 1j * np.random.rand(5)).astype(np.complex128)
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_float_2d(self) -> None:
        x = np.random.rand(5, 5).astype(np.float32)
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_bytes(self) -> None:
        x = np.array([b"abc", b"cba"])
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_mixed(self) -> None:
        x = np.array(
            [(1, 2, b"a", [1.0, 2.0])],
            np.dtype(
                [
                    ("arg0", np.uint32),
                    ("arg1", np.uint32),
                    ("arg2", "S1"),
                    ("arg3", np.float32, (2,)),
                ]
            ),
        )
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_non_contiguous(self) -> None:
        x = np.ones((10, 10), np.uint32)[0:5, 0:5]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_structured_array(self) -> None:
        structured_dtype = np.dtype([("a", float), ("b", int)])

        x = np.empty((10,), dtype=structured_dtype)
        x["a"] = np.arange(10)
        x["b"] = np.arange(10)

        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_shaped_structured_array(self) -> None:
        shaped_structured_dtype = np.dtype([("a", float, 3), ("b", int)])

        x = np.empty((10,), dtype=shaped_structured_dtype)
        x["a"] = np.arange(30).reshape(10, 3)
        x["b"] = np.arange(10)

        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_nested_structured_array(self) -> None:
        structured_dtype = np.dtype([("a", float), ("b", int)])
        nested_dtype = np.dtype([("foo", structured_dtype), ("bar", structured_dtype)])

        x = np.empty((10,), dtype=nested_dtype)
        x["foo"]["a"] = np.arange(10)
        x["foo"]["b"] = np.arange(10)
        x["bar"]["a"] = np.arange(10) + 10
        x["bar"]["b"] = np.arange(10) + 10

        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_dumps_no_cls_with_default(self) -> None:
        sentinel = {"sentinel": str(uuid.uuid4())}

        def user_default(_: Any) -> dict[str, Any]:
            return sentinel

        dumped = json.dumps(b"unserializable", default=user_default)
        self.assert_equal_with_type(json.loads(dumped), sentinel)

        x = np.random.rand(5).astype(np.float32)
        dumped = json.dumps(x, default=user_default)
        self.assert_equal_with_type(json.loads(dumped), x)

    def test_dumps_with_cls_with_default(self) -> None:
        # The `default` kwargs always overrides the `default` method of the `cls`.
        sentinel = {"sentinel": [37, 42]}

        class Encoder(json.JSONEncoder):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                del kwargs["separators"]
                super().__init__(*args, separators=(",  ", ":  "), **kwargs)

            def default(self, _: Any) -> None:  # pragma: no cover
                raise RuntimeError("Should never be called")  # noqa: TRY003, EM101

        def user_default(_: Any) -> dict[str, Any]:
            return sentinel

        dumped = json.dumps(b"unserializable", default=user_default, cls=Encoder)
        self.assertEqual(dumped, '{"sentinel":  [37,  42]}')

        x = np.random.rand(5).astype(np.float32)
        dumped = json.dumps(x, default=user_default, cls=Encoder)
        self.assert_equal_with_type(json.loads(dumped), x)

    def test_dumps_with_cls_no_default(self) -> None:
        sentinel = {"sentinel": [37, 42]}

        class Encoder(json.JSONEncoder):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                del kwargs["separators"]
                super().__init__(*args, separators=(",  ", ":  "), **kwargs)

            def default(self, _: Any) -> dict[str, Any]:
                return sentinel

        dumped = json.dumps(b"unserializable", cls=Encoder)
        self.assertEqual(dumped, '{"sentinel":  [37,  42]}')

        x = np.random.rand(5).astype(np.float32)
        dumped = json.dumps(x, cls=Encoder)
        self.assert_equal_with_type(json.loads(dumped), x)
