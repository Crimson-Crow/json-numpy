import json
import unittest

import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from json_numpy import patch


class NumpyJsonSerializationTest(unittest.TestCase):

    def setUp(self):
        patch()

    @staticmethod
    def dumps_loads(x):
        return json.loads(json.dumps(x))

    @staticmethod
    def assert_equal_with_type(actual, desired, sort_key=None):
        if isinstance(desired, np.ndarray):
            assert_array_equal(actual, desired)
            assert_equal(actual.dtype, desired.dtype)
        elif isinstance(desired, list):
            assert_array_equal(actual, desired)
            assert_array_equal([type(e) for e in actual],
                               [type(e) for e in desired])
        elif isinstance(desired, dict):
            assert_array_equal(sorted(actual.values(), key=sort_key),
                               sorted(desired.values(), key=sort_key))
            assert_array_equal([type(e) for e in sorted(actual.values(), key=sort_key)],
                               [type(e) for e in sorted(desired.values(), key=sort_key)])
            assert_array_equal(sorted(actual.keys()), sorted(desired.keys()))
            assert_array_equal([type(e) for e in sorted(actual.keys())],
                               [type(e) for e in sorted(desired.keys())])
        else:
            assert_equal(actual, desired)
            assert_equal(type(actual), type(desired))

    def test_numpy_scalar_bool(self):
        for b in (True, False):
            x = np.bool_(b)
            self.assert_equal_with_type(self.dumps_loads(x), x)

    @unittest.skip("np.str_ is a subclass of str and get automatically serialized as such")
    def test_numpy_scalar_str(self):
        x = np.str_("abc")
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_scalar_float(self):
        x = np.float32(np.random.rand())
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_scalar_complex(self):
        x = np.complex64(np.random.rand() + 1j * np.random.rand())
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_bool(self):
        x = [np.bool_(True), np.bool_(False)]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    @unittest.skip("np.str_ is a subclass of str and get automatically serialized as such")
    def test_list_numpy_scalar_str(self):
        x = [np.str_("abc"), np.str_("cba")]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_float(self):
        x = [np.float32(np.random.rand()) for _ in range(5)]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_complex(self):
        x = [np.complex64(np.random.rand() + 1j * np.random.rand()) for _ in range(5)]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_numpy_scalar_float_complex(self):
        x = [np.float32(np.random.rand()) for i in range(5)] + \
            [np.complex128(np.random.rand() + 1j * np.random.rand()) for i in range(5)]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_list_mixed(self):
        x = [1.0, np.float32(3.5), np.complex128(4.25), 'foo']
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_dict_numpy_float(self):
        x = {'foo': np.float32(1.0), 'bar': np.float32(2.0)}
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_dict_numpy_complex(self):
        x = {'foo': np.complex128(1.0 + 1.0j), 'bar': np.complex128(2.0 + 2.0j)}
        self.assert_equal_with_type(self.dumps_loads(x), x, sort_key=np.linalg.norm)

    def test_numpy_array_float(self):
        x = np.random.rand(5).astype(np.float32)
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_complex(self):
        x = (np.random.rand(5) + 1j * np.random.rand(5)).astype(np.complex128)
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_float_2d(self):
        x = np.random.rand(5, 5).astype(np.float32)
        self.assert_equal_with_type(self.dumps_loads(x), x)

    # @unittest.skip("np.str_ is a subclass of str and get serialized automatically as such")
    def test_numpy_array_bytes(self):
        x = np.array([b'abc', b'cba'])
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_mixed(self):
        x = np.array([(1, 2, b'a', [1.0, 2.0])],
                     np.dtype([('arg0', np.uint32),
                               ('arg1', np.uint32),
                               ('arg2', 'S1'),
                               ('arg3', np.float32, (2,))]))
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_array_non_contiguous(self):
        x = np.ones((10, 10), np.uint32)[0:5, 0:5]
        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_structured_array(self):
        structured_dtype = np.dtype([("a", float), ("b", int)])

        x = np.empty((10,), dtype=structured_dtype)
        x["a"] = np.arange(10)
        x["b"] = np.arange(10)

        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_shaped_structured_array(self):
        shaped_structured_dtype = np.dtype([("a", float, 3), ("b", int)])

        x = np.empty((10,), dtype=shaped_structured_dtype)
        x["a"] = np.arange(30).reshape(10, 3)
        x["b"] = np.arange(10)

        self.assert_equal_with_type(self.dumps_loads(x), x)

    def test_numpy_nested_structured_array(self):
        structured_dtype = np.dtype([("a", float), ("b", int)])
        nested_dtype = np.dtype([("foo", structured_dtype), ("bar", structured_dtype)])

        x = np.empty((10,), dtype=nested_dtype)
        x["foo"]["a"] = np.arange(10)
        x["foo"]["b"] = np.arange(10)
        x["bar"]["a"] = np.arange(10) + 10
        x["bar"]["b"] = np.arange(10) + 10

        self.assert_equal_with_type(self.dumps_loads(x), x)


if __name__ == '__main__':
    unittest.main()
