# Getting Started

OMP4Py targets Python programmers who want OpenMP-like parallel regions,
worksharing loops, tasks, sections, reductions, and synchronization without
rewriting their application in C, C++, or Fortran.

At a high level, OMP4Py code has two parts:

- Use {func}`omp4py.omp` as a decorator on functions or classes that contain
  OpenMP directives.
- Use `with omp("...")` blocks, or standalone `omp("barrier")` style calls,
  to mark structured OpenMP regions.

```python
from omp4py import *

@omp
def hello():
    with omp("parallel num_threads(4)"):
        print("hello from", omp_get_thread_num())

hello()
```

For scalable CPU parallelism, use a free-threaded Python build. Standard GIL
builds can still run OMP4Py programs, but they cannot expose the same level of
threaded execution for Python bytecode.
