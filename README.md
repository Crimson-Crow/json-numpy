json-numpy
==========

[![PyPI](https://img.shields.io/pypi/v/json-numpy)](https://pypi.org/project/json-numpy/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/json-numpy)](https://pypi.org/project/json-numpy/)
[![GitHub](https://img.shields.io/github/license/Crimson-Crow/json-numpy)]((https://github.com/Crimson-Crow/json-numpy/blob/main/LICENSE.txt))
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)

Description
-----------

`json-numpy` provides lossless and quick JSON encoding/decoding for [NumPy](http://www.numpy.org/) arrays and scalars.

`json-numpy` follows [Semantic Versioning](https://semver.org/).

Installation
------------

`json-numpy` can be installed using [`pip`](https://pip.pypa.io/en/stable/):

    $ pip install json-numpy

Usage
-----

For a quick start, `json_numpy` can be used as a simple drop-in replacement of the built-in `json` module. \
The `dump()`, `load()`, `dumps()` and `loads()` methods are implemented by wrapping the original methods and replacing the default encoder and decoder. \
More information on the usage can be found in the `json` module's [documentation](https://docs.python.org/3/library/json.html).

```python
import numpy as np
import json_numpy

arr = np.array([0, 1, 2])
encoded_arr_str = json_numpy.dumps(arr)
decoded_arr = json_numpy.loads(encoded_arr_str)
```

Another way of using `json_numpy` is to explicitly use the provided encoder and decoder functions in conjunction with the `json` module.

```python
import json
import numpy as np
from json_numpy import default, object_hook

arr = np.array([0, 1, 2])
encoded_arr_str = json.dumps(arr, default=default)
decoded_arr = json.loads(encoded_arr_str, object_hook=object_hook)
```

Finally, the last way of using `json_numpy` is by monkey patching the `json` module after importing it first:

```python
import json
import numpy as np
import json_numpy

json_numpy.patch()

arr = np.array([0, 1, 2])
encoded_arr_str = json.dumps(arr)
decoded_arr = json.loads(encoded_arr_str)
```

This method can be used to change the behavior of a module depending on the `json` module without editing its code.

Tests
-----

The simplest way to run tests is:

    $ python -m unittest

As a more robust alternative, you can install [`tox`](https://tox.readthedocs.io/en/latest/install.html) to automatically test across the supported python versions, then run:

    $ tox

Issue tracker
-------------

Please report any bugs or enhancement ideas using the [issue tracker](https://github.com/Crimson-Crow/json-numpy/issues).

License
-------

`json-numpy` is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT) (see [LICENSE.txt](https://github.com/Crimson-Crow/json-numpy/blob/main/LICENSE.txt) for more information).
