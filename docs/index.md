# OMP4Py

## OpenMP for Python

```{image} _static/logo.png
:alt: OMP4Py logo
:width: 160px
:align: left
```

**OMP4Py** brings the OpenMP programming model to Python. Write parallel loops, sections and
synchronisation with the familiar OpenMP directives, directly in your Python code, and run
them with real threads on a free-threaded interpreter.

```python
from omp4py import *

@omp
def pi(n):
    w = 1.0 / n
    total = 0.0
    with omp("parallel for reduction(+:total) firstprivate(w)"):
        for i in range(n):
            x = (i + 0.5) * w
            total += 4.0 / (1.0 + x * x)
    return total * w

print(pi(10_000_000))
```

```{button-ref} getting_started/index
:color: primary
:shadow:

Get started
```
```{button-ref} user_guide/index
:color: secondary
:outline:

User guide
```

## Why OMP4Py

````{grid} 1 2 2 3
:gutter: 3

```{grid-item-card} Standard OpenMP syntax
Directives and clauses follow the OpenMP names you already know: `parallel`, `for`,
`schedule`, `reduction`, `critical`, ...
```

```{grid-item-card} Four execution modes
Pure Python, hybrid (native runtime), compiled and compiled with types. Same source, from
easy debugging to near-C speed.
```

```{grid-item-card} Full Python
Any Python object or library works inside a parallel region. No restriction to a numeric subset
as in JIT-based tools.
```

```{grid-item-card} Real threads
Built for free-threaded Python (3.13t and later), so threads run Python code truly in
parallel.
```

```{grid-item-card} Extensible
Add your own directives, transformations and AST passes with devices, subparsers and
modifiers, without touching the core.
```

```{grid-item-card} Works with MPI
Combine it with `mpi4py` for hybrid intra-node and inter-node parallelism.
```
````

## How it works, in one picture

```{mermaid}
flowchart LR
    A["Your Python code<br/>with omp(...) directives"] --> B["@omp preprocessor<br/>(parse · transform)"]
    B --> C["Plain Python<br/>calling the OMP4Py runtime"]
    B -. "compile=True" .-> D["Native code<br/>(Cython)"]
    C --> E["Team of threads"]
    D --> E
```

## Find your way

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} New here?
:link: getting_started/index
:link-type: doc

Install it and run your first parallel program in minutes.
```

```{grid-item-card} Learn the tool
:link: user_guide/index
:link-type: doc

The `omp` decorator, directives, clauses, runtime routines, modes and performance.
```

```{grid-item-card} See it in action
:link: tutorials/index
:link-type: doc

Worked examples: pi, matrix multiplication, histograms, NumPy.
```

```{grid-item-card} Extend or contribute
:link: contributor/index
:link-type: doc

Architecture, the runtime, and the extension API (devices, subparsers, modifiers).
```
````

```{toctree}
:hidden:
:caption: Getting Started
:maxdepth: 2

getting_started/index
getting_started/installation
getting_started/quickstart
getting_started/concepts
```

```{toctree}
:hidden:
:caption: User Guide
:maxdepth: 2

user_guide/index
user_guide/decorator/index
user_guide/directives
user_guide/combined_directives
user_guide/clauses
user_guide/runtime_api
user_guide/environment
user_guide/execution_modes
user_guide/compiled_mode
user_guide/performance
user_guide/debugging
```

```{toctree}
:hidden:
:caption: Tutorials and Examples
:maxdepth: 2

tutorials/index
tutorials/pi
tutorials/matmul
tutorials/histogram
tutorials/mandelbrot
tutorials/collatz
tutorials/numpy
```

```{toctree}
:hidden:
:caption: Contributor Guide
:maxdepth: 2

contributor/index
contributor/architecture
contributor/runtime
contributor/loaders
contributor/adding_directive
contributor/extensions/index
contributor/code_quality
contributor/internals
```

```{toctree}
:hidden:
:caption: Compatibility
:maxdepth: 2

compatibility/coverage
compatibility/differences
```

```{toctree}
:hidden:
:caption: More
:maxdepth: 1

benchmarks
faq
contributing
changelog
license
```

## Citing OMP4Py

If you use OMP4Py in academic work, please cite:

- César Piñeiro and Juan C. Pichel. [Unlocking Python Multithreading
  Capabilities using OpenMP-Based Programming with OMP4Py](
  https://doi.org/10.1109/CGO68049.2026.11395224). IEEE/ACM International
  Symposium on Code Generation and Optimization (CGO), 2026.
- César Piñeiro and Juan C. Pichel. [OMP4Py: A pure Python implementation
  of OpenMP](https://doi.org/10.1016/j.future.2025.108035). Future Generation
  Computer Systems, Vol. 175, 2026.
