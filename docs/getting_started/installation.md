# Installation

## Requirements

OMP4Py currently targets Python 3.13 or newer. For real multithreaded speedups,
use a free-threaded Python build such as `3.13t` or newer.

## Install From Git

Install the project directly from the repository:

```bash
pip install git+https://github.com/citiususc/omp4py.git
```

## Install For Local Development

From a local checkout, install the runtime and compile optional native modules:

```bash
uv sync --reinstall-package omp4py
```

Build a wheel with:

```bash
uv build
```

## Pure Python Runtime

To force the pure Python runtime, set `OMP4PY_PURE` before importing OMP4Py:

```python
import os

os.environ["OMP4PY_PURE"] = "True"

from omp4py import *
```

## Useful Environment Variables

`OMP4PY_PURE`

: Use the pure Python runtime even when compiled modules are available.

`OMP4PY_CACHE_DIR`

: Override the directory used for preprocessed and compiled artifacts.

`OMP4PY_IGNORE_CACHE`

: Regenerate cached preprocessing and compilation outputs.

`OMP4PY_DUMP`

: Dump transformed source code for inspection.

`OMP4PY_DEBUG`

: Enable additional debugging behavior in the preprocessing pipeline.
