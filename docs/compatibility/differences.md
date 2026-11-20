# Differences from OpenMP in C, C++ and Fortran

OMP4Py follows the **C/C++ nomenclature** of OpenMP: the directive, clause and routine names
(`parallel`, `schedule`, `omp_get_thread_num`, ...) and their meaning are the ones of the
standard. The syntax is adapted as closely as possible to Python, and the differences come from
the language, not from the model. The details of what is supported are in the
{doc}`/user_guide/directives` and {doc}`/user_guide/clauses` pages.

## Syntax

| OpenMP (C) | OMP4Py |
| --- | --- |
| `#pragma omp parallel for` | `with omp("parallel for"):` |
| `#pragma omp barrier` | `omp("barrier")` |
| Directive applies to the next statement | Directive applies to the **block** of the `with` |
| Compiler flag `-fopenmp` | `@omp`, `omp(pkg=...)` or `omp(py=...)` ({doc}`/user_guide/decorator/index`) |
| `#pragma omp threadprivate(x)` after the declaration | `omp("threadprivate(x)")` |

Python has **no pragmas**. The directive travels inside a string, and a statement-like directive
is a call. Everything that C expresses with pragma placement (before a statement, outside a
function) is adapted, or removed when it makes no sense in Python.

## Loops

- The worksharing `for` accepts `for i in range(...)`, not the `i = 0; i < n; i++` form: the
  loop-form restrictions of OpenMP become the restrictions of `range`.
- `break`, `return` and `yield` cannot leave a structured block, as in OpenMP.
- The loop variable is private automatically.

## Variables and scopes

- Python has function scope, not block scope. A variable first assigned inside a `parallel` block is
  local to the generated function, so it is not visible afterwards. Declare it before the region
  (as shared, `lastprivate` or part of a `reduction`).
- Names in data-sharing clauses must already exist.
- `private` variables are **not** initialised from the original value; use `firstprivate`.
- Shared Python objects are shared by reference: concurrent compound updates such as `d[k] += 1`
  are not atomic.

## The decorator is more limited than module level

In C, some directives are written **outside functions**, at file scope:
`threadprivate` and `declare reduction` are the usual examples. Python code at module level is
not inside a function that the decorator can process, so what you can do depends on how you use
`omp`:

| | `@omp` on a function | Package mode (`omp(pkg=...)`) |
| --- | --- | --- |
| Directives inside functions | yes | yes |
| Directives at module level | **no** | **yes** |

With the **decorator**, only the decorated function is processed, so a `threadprivate` has to be
written *inside* a decorated function:

```python
counter = 0

@omp
def per_thread():
    global counter
    omp("threadprivate(counter)")      # inside the function
    ...
```

With **package mode**, the whole module is processed, so the directive can be at module level,
as in C:

```python
# mypkg/state.py  (module of a package marked with omp(pkg=...))
from omp4py import *

counter = 0
omp("threadprivate(counter)")          # module level, like in C
```

See {doc}`/user_guide/decorator/index` for the two modes.

## Reductions

- Reduction variables can be any Python object that supports the operator; numbers are copied by
  assignment, other objects use a generic copy.
- The operators are Python's: `and`/`or` instead of `&&`/`||`.

## Threads and performance

- Threads are Python threads. On a build **with the GIL** only one runs Python code at a time; use
  a free-threaded build ({doc}`/user_guide/performance`).
- A new team is created for every `parallel` region (there is no thread pool yet).
- Numbers are Python objects unless you compile with type annotations
  ({doc}`/user_guide/compiled_mode`).

## Errors

OpenMP "non-conforming program" situations that can be detected statically are reported as Python
`SyntaxError`s at preprocessing time, with the position in your file. Exceptions raised inside a
region propagate to the caller after the threads have been joined.

## Source code is required

The preprocessor reads your source, so directives cannot be used in `exec()`/`eval()` strings or
in the interactive prompt.
