# Core Concepts

This page introduces the handful of ideas you need before using OMP4Py. Everything
else in the documentation builds on them.

## Directives are strings

In C, OpenMP directives are `#pragma omp ...` lines. Python has no pragmas, so OMP4Py
uses the {doc}`omp </user_guide/decorator/index>` function and puts the directive in a string:

```python
with omp("parallel for reduction(+:total) num_threads(4)"):
    for i in range(n):
        total += i
```

The directive string has the same syntax as in C/C++: a *construct*
(`parallel`, `for`, `critical`, ...) followed by *clauses*
(`reduction(+:total)`, `num_threads(4)`, ...). Constructs can be combined
(`parallel for`, `parallel sections`).

Two forms exist, depending on whether the directive owns a block of code:

| Form | Used for | Example |
| --- | --- | --- |
| `with omp("..."):` | Constructs that apply to a *structured block* | `parallel`, `for`, `sections`, `single`, `critical`, `master` |
| `omp("...")` | Stand-alone directives with no block | `barrier`, `threadprivate(x)` |

At run time the call `omp("...")` does nothing at all; it only has meaning for the
preprocessor. That is why your code is still valid Python (and can be
debugged and type-checked) before it is transformed.

## The `@omp` decorator

Directives are only recognised inside code that the preprocessor has seen. You mark
that code with the decorator form of `omp`:

```python
@omp
def work(): ...

@omp(compile=True)      # decorator with options
class Solver: ...
```

When the decorated function is defined, OMP4Py reads its source, rewrites every
directive into ordinary Python plus calls to the OMP4Py runtime, and replaces the
function with the transformed version. Other ways to select code (whole
packages, files) are described in {doc}`/user_guide/decorator/index`.

## How a program is processed

```{mermaid}
flowchart LR
    A["Python source with<br/>omp(...) directives"] --> B["Parse to AST<br/>and find directives"]
    B --> C["Parse each directive<br/>string (constructs + clauses)"]
    C --> M1["Modifiers<br/>(before)"]
    M1 --> D["Transformer: generates<br/>the parallel code"]
    D --> M2["Modifiers<br/>(after)"]
    M2 --> F{"compile=True?"}
    F -- no --> G["Plain Python<br/>+ omp4py.runtime calls"]
    F -- yes --> H["Compilation<br/>(Cython)"]
    G --> I["Run: threads execute<br/>the parallel regions"]
    H --> I
```

The result for a `parallel for` over `range(n)` is regular Python code in which the
loop bounds are obtained from the runtime scheduler. You can inspect it with the
`dump` option (see {doc}`/user_guide/debugging`).

## The execution model

OMP4Py follows OpenMP's *fork-join* model:

```{mermaid}
sequenceDiagram
    participant M as Master thread
    participant T1 as Thread 1
    participant T2 as Thread 2
    M->>M: sequential code
    M->>T1: fork (parallel region)
    M->>T2: fork (parallel region)
    par team executes the block
        M->>M: block (thread 0)
    and
        T1->>T1: block (thread 1)
    and
        T2->>T2: block (thread 2)
    end
    T1-->>M: join (implicit barrier)
    T2-->>M: join (implicit barrier)
    M->>M: sequential code
```

- Outside a `parallel` region only the initial (master) thread runs.
- Entering `parallel` creates a *team* of threads that all execute the block.
- Worksharing constructs (`for`, `sections`, `single`) divide the work *among the
  threads of the current team*. Place them inside a `parallel` region or combine
  them with it, as in `parallel for`.
- Leaving the region joins the threads. Worksharing constructs end with an
  implicit barrier unless you add `nowait`.

## Data sharing

Inside a parallel region every variable is one of:

shared
: One copy visible to all threads (the default for variables defined outside the region).

private
: Each thread has its own uninitialised copy (`private(x)`).

firstprivate
: Private, initialised with the value from before the region (`firstprivate(x)`).

lastprivate
: Private, and the value of the last iteration/section is copied back (`lastprivate(x)`).

reduction
: Private partial results combined with an operator at the end (`reduction(+:x)`).

threadprivate
: A global variable that has one persistent copy per thread (`omp("threadprivate(x)")`).

Variables named in `private`, `firstprivate`, `lastprivate`, `shared` or `reduction`
must already exist before the region. See {doc}`/user_guide/clauses`.

## The GIL and free threading

OMP4Py creates real operating-system threads. Whether they run Python bytecode *at
the same time* depends on the interpreter:

- **Free-threaded builds** (`python3.13t` and later) run threads truly in parallel.
  This is what you want for speedups.
- **GIL builds** still run correctly, but only one thread executes Python
  bytecode at any time. You only gain when threads wait on I/O or call into
  libraries that release the GIL.

## The runtime and pure mode

Generated code calls the OMP4Py **runtime** (thread teams, schedulers, barriers, locks). The
runtime exists in two forms: written in Python, and compiled to native code with Cython.
When you `import omp4py`, the compiled runtime is used if it is available; otherwise OMP4Py
falls back to the Python one.

You can force the Python runtime, for example to debug, by setting `OMP4PY_PURE` **before**
importing OMP4Py:

```python
import os

os.environ["OMP4PY_PURE"] = "True"

from omp4py import *
```

or from the shell: `OMP4PY_PURE=1 python program.py`. Do not mix runtimes inside one program:
each one keeps its own state.

## Execution modes

The same source can run in four modes, trading flexibility for speed. They are covered
in {doc}`/user_guide/execution_modes`:

| Mode | How | Runtime | Your function |
| --- | --- | --- | --- |
| Pure | `OMP4PY_PURE=1` | Python | Python |
| Hybrid (default) | `from omp4py import *` | Compiled (Cython) | Python |
| Compiled | `@omp(compile=True)` | Compiled | Compiled |
| Compiled with types | `@omp(compile=True)` + annotations | Compiled | Compiled, statically typed |
