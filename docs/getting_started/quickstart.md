# Quick Start

The main entry point is {func}`omp4py.omp`. Decorate the Python function that
contains directives, then use directive strings in structured blocks.

## Computing Pi

```python
from omp4py import *

@omp
def pi(n: int) -> float:
    width = 1.0 / n
    pi_value = 0.0

    with omp("parallel for reduction(+:pi_value)"):
        for i in range(n):
            x = (i + 0.5) * width
            pi_value += 4.0 / (1.0 + x * x)

    return pi_value * width

print(pi(10_000_000))
```

## Set The Number Of Threads

Use the runtime API before entering a parallel region:

```python
from omp4py import *

omp_set_num_threads(4)

@omp
def show_threads():
    with omp("parallel"):
        print(omp_get_thread_num(), "of", omp_get_num_threads())

show_threads()
```

## Use Package-Level Preprocessing

Inside a package `__init__.py` file, call `omp(pkg=__package__)` to enable
lazy preprocessing for modules imported from that package:

```python
from omp4py import omp

omp(pkg=__package__)
```

## Run With uv

```bash
uv run -p 3.13t python path/to/program.py
```
