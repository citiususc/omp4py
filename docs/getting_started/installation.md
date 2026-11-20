# Installation

## Requirements

- **Python 3.13 or newer** (3.13, 3.14 and 3.15 are listed as supported).
- A **free-threaded** build (`3.13t`, `3.14t`, ...) to run threads in parallel. A normal
  (GIL) build works too, but CPU-bound code will not speed up; see
  {doc}`/user_guide/performance`.
- For the compiled modes only: a C compiler, Cython and setuptools.

## Install with pip

```{note}
**Coming soon.** OMP4Py is not published on PyPI yet. Until then, install it from the Git
repository as shown below.
```

When it is published, the command will be:

```bash
pip install omp4py
```

## Install from Git

This is the way to install OMP4Py today:

```bash
pip install git+https://github.com/citiususc/omp4py.git
```

To also get the compilation support (Cython and setuptools):

```bash
pip install "omp4py[compile] @ git+https://github.com/citiususc/omp4py.git"
```

If you install from source, the native runtime is built during installation when your
interpreter and compiler allow it; otherwise OMP4Py falls back to its pure Python runtime
({doc}`concepts`).

## Using uv

[uv](https://docs.astral.sh/uv/) can create the environment and select a free-threaded
interpreter for you:

```bash
uv venv -p 3.14t
uv pip install git+https://github.com/citiususc/omp4py.git
uv run python my_program.py
```

## Verify the installation

```bash
python -c "from omp4py import *; omp_set_num_threads(2); print(omp_get_max_threads())"
```

Check whether your interpreter runs threads in parallel:

```bash
python -c "import sys; print('GIL enabled:', sys._is_gil_enabled())"
```

`GIL enabled: False` means a free-threaded build.

## Contributing to OMP4Py?

Development setup (cloning, `uv` dependency groups, building the native runtime) is described in
the {doc}`Contributor Guide </contributor/index>`.
