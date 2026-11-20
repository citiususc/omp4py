# Quick Start

Six short steps that cover what you will use most. Each one ends with a link to the full
explanation in the {doc}`User Guide </user_guide/index>`.

All the code here is complete and runnable. Save it in a `.py` file: the preprocessor needs
to read the source, so it does not work from the interactive prompt.

## 1 · Mark your code and open a parallel region

```{literalinclude} ../examples/directives/parallel.py
:language: python
```

```text
hello from thread 3 of 4
hello from thread 2 of 4
hello from thread 1 of 4
hello from thread 0 of 4
```

`@omp` tells OMP4Py to process the function. `with omp("parallel ...")` runs the block once in
each thread. The line order changes from run to run.

→ {doc}`/user_guide/decorator/index`, {doc}`/user_guide/directives`

## 2 · Split a loop and combine results

```{literalinclude} ../examples/pi.py
:language: python
```

`parallel for` divides the iterations among the threads, and `reduction(+:total)` gives every
thread a private partial sum that is added to `total` at the end.

→ {doc}`/user_guide/directives`, {doc}`/user_guide/clauses`

## 3 · Say who owns each variable

```{literalinclude} ../examples/clauses/data.py
:language: python
```

```text
(4, 10, 5, 9)
```

`shared`, `private`, `firstprivate`, `lastprivate` and `reduction` decide whether threads see one
variable or each have their own.

→ {doc}`/user_guide/clauses`

## 4 · Protect shared updates and synchronise

```{literalinclude} ../examples/directives/barrier.py
:language: python
```

`critical` lets one thread at a time into a block; `omp("barrier")` waits for the whole team.

→ {doc}`/user_guide/directives`

## 5 · Control the team from code

```{literalinclude} ../examples/runtime/routines.py
:language: python
```

The `omp_*` functions of the OpenMP API are imported together with `omp`.

→ {doc}`/user_guide/runtime_api`

## 6 · Choose how it runs

| Mode | Enable with | Use it for |
| --- | --- | --- |
| Pure | `OMP4PY_PURE=1` | debugging, no compiler needed |
| Hybrid (default) | `from omp4py import *` | general use, native runtime |
| Compiled | `@omp(compile=True)` | numeric loops |
| Compiled with types | `compile=True` + annotations | best performance |

→ {doc}`/user_guide/execution_modes`, {doc}`/user_guide/compiled_mode`

## Run it

```bash
python my_program.py                    # any Python
uv run -p 3.14t python my_program.py    # a free-threaded interpreter with uv
OMP_NUM_THREADS=8 python my_program.py  # choose the default team size
```
