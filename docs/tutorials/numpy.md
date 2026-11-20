# Tutorial: Using NumPy Inside Parallel Loops

OMP4Py loops are ordinary Python, so any library can be called from them. The result
scales when the library function either runs quickly in compiled code **and** releases the
GIL, or when you use a free-threaded interpreter and a build of the library that
supports it.

```{literalinclude} ../examples/numpy_rows.py
:language: python
```

```text
True
```

The example computes the Euclidean norm of each row of a matrix. Each iteration calls
`np.linalg.norm` on a different row and writes to a different element of `out`, so there
is no conflict.

## Guidelines

- **Parallelise over independent chunks of work** (rows, blocks, files), not over single
  scalar elements: calling NumPy on one number costs more than the arithmetic.
- **Write results into preallocated arrays** (`out[i] = ...`) rather than appending to a
  shared list.
- **Avoid oversubscription.** If the library is itself multi-threaded (BLAS, for
  instance), `N` OMP4Py threads times `M` library threads can exceed the number of cores.
  Limit the library threads (for example `OMP_NUM_THREADS=1` or the library's own setting)
  when you parallelise outside the call.
- **Check your interpreter.** NumPy needs a build that supports free threading to
  run safely with the GIL disabled.

## Combining with mpi4py

OMP4Py parallelises inside a node; `mpi4py` distributes across nodes. Each MPI rank can
run its own OMP4Py regions:

```python
from mpi4py import MPI
from omp4py import *

comm = MPI.COMM_WORLD

@omp
def local_sum(chunk):
    total = 0
    with omp("parallel for reduction(+:total)"):
        for i in range(len(chunk)):
            total += chunk[i]
    return total

partial = local_sum(my_chunk)
total = comm.allreduce(partial)
```
