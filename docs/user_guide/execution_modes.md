# Execution Modes

OMP4Py can run the same program in four modes. They differ in *what is native code*: the OMP4Py
runtime (schedulers, barriers, locks) and/or your own functions.

```{mermaid}
flowchart TB
    subgraph pure["1 - Pure"]
        P1["Runtime: Python"] --- P2["Your code: Python"]
    end
    subgraph hybrid["2 - Hybrid (default)"]
        H1["Runtime: native"] --- H2["Your code: Python"]
    end
    subgraph comp["3 - Compiled"]
        C1["Runtime: native"] --- C2["Your code: Cython (untyped)"]
    end
    subgraph typed["4 - Compiled with types"]
        T1["Runtime: native"] --- T2["Your code: Cython + type annotations"]
    end
    pure --> hybrid --> comp --> typed
```

| Mode | How to select it | Needs |
| --- | --- | --- |
| **Pure** | `OMP4PY_PURE=1` | Only Python |
| **Hybrid** (default) | `from omp4py import *` | The native runtime (installed with the package when it can be built) |
| **Compiled** | `@omp(compile=True)` or `omp(pkg=..., compile=True)` | Cython, setuptools and a C compiler |
| **Compiled with types** | Compiled + type annotations | The same |

## 1 · Pure mode

Everything runs as ordinary Python, including the runtime. No compiler is needed and every line
can be debugged with standard tools.

```python
import os
os.environ["OMP4PY_PURE"] = "True"      # before importing omp4py

from omp4py import *
```

Use it to develop, debug, measure coverage, or on platforms without the native runtime.

## 2 · Hybrid mode (default)

```python
from omp4py import *
```

The runtime is native code: thread synchronisation uses atomic operations instead of locks,
which helps most in fine-grained tasks and frequent synchronisation. Your functions still run in
the interpreter.

## 3 · Compiled mode

```python
@omp(compile=True)
def kernel(n): ...
```

Your code is also translated to C by Cython, so it no longer runs in the interpreter. By default
Cython is conservative and treats variables as generic Python objects, so the improvement is
moderate. See {doc}`compiled_mode`.

## 4 · Compiled with types

Add type annotations (`int`, `float`) to your variables and Cython generates native arithmetic:
this is where the large speedups appear. See {doc}`compiled_mode`.

## Mixing modes

- **Hybrid and compiled functions can live in the same program**: both use the native runtime.
- **Do not mix the pure runtime with the native one** in one program: each runtime keeps its own
  state and they do not share it, which can give unexpected results.
- Options are per decorator, so you can compile only the hot kernel and keep the rest as
  plain Python. Because the new decorator has its own name, it needs an `alias`
  ({doc}`decorator/directive`):

```python
from omp4py import omp

fast = omp(alias="fast", compile=True)

@fast
def kernel(a, b):
    with fast("parallel for"):
        for i in range(len(a)):
            a[i] += b[i]

@omp
def orchestration():
    ...                       # plain Python
```

## Choosing a mode

```{list-table}
:header-rows: 1

* - Situation
  - Mode
* - Developing, debugging, writing tests
  - Pure
* - Code full of Python objects (dictionaries, graphs, strings)
  - Hybrid, or Compiled
* - Numeric loops over `int`/`float`
  - Compiled with types
* - Work dominated by a library that already runs native code
  - Hybrid
```
