"""Task execution constructs for the `omp4py` runtime.

This package contains all task-related logic of the runtime, used to
represent and manage execution units (tasks) within the OpenMP-like
runtime model implemented by `omp4py`.

It defines the `Task` classes, shared execution context, and
synchronization primitives, as well as constructs such as parallel
regions. All tasking and parallel execution logic of the OpenMP model
is implemented within this package.
"""

from omp4py.runtime.tasks.barrier import barrier  # noqa: F401
from omp4py.runtime.tasks.compiler import cy_cast, cy_typeof  # noqa: F401
from omp4py.runtime.tasks.parallelism import parallel  # noqa: F401
from omp4py.runtime.tasks.privatization import copy_var, new_var  # noqa: F401
from omp4py.runtime.tasks.threadprivate import threadprivate, threadprivates  # noqa: F401
from omp4py.runtime.tasks.worksharing import (  # noqa: F401
    ForBounds,
    SingleCopyPrivate,
    for_bounds,
    for_end,
    for_init,
    for_next_dynamic,
    for_next_guided,
    for_next_runtime,
    for_next_static,
    ordered_next,
    ordered_start,
    sections_end,
    sections_init,
    sections_next,
    single_copy_get,
    single_copy_notify,
    single_copy_wait,
    single_end,
    single_init,
)
