# Runtime Routines

Besides the `omp` decorator, the `omp4py` package exports the **OpenMP runtime library
routines**. Import both at once:

```python
from omp4py import *        # omp + all the omp_* routines
```

They can be called from anywhere (inside or outside a parallel region) and behave the same in
all {doc}`execution modes <execution_modes>`. The documentation follows the OpenMP 4.5 API.

```{literalinclude} ../examples/runtime/routines.py
:language: python
```

```text
max threads: 3
inside a region? False
team size: 3 | inside a region? True
schedule(runtime) is now (2, 8)
```

```{contents}
:local:
:depth: 2
```

## Implemented routines

### Thread team

```{eval-rst}
.. autofunction:: omp4py.omp_set_num_threads
.. autofunction:: omp4py.omp_get_num_threads
.. autofunction:: omp4py.omp_get_max_threads
.. autofunction:: omp4py.omp_get_thread_num
.. autofunction:: omp4py.omp_in_parallel
.. autofunction:: omp4py.omp_get_thread_limit
.. autofunction:: omp4py.omp_set_dynamic
.. autofunction:: omp4py.omp_get_dynamic
```

### Nested parallelism

```{eval-rst}
.. autofunction:: omp4py.omp_get_level
.. autofunction:: omp4py.omp_get_active_level
.. autofunction:: omp4py.omp_get_ancestor_thread_num
.. autofunction:: omp4py.omp_get_team_size
.. autofunction:: omp4py.omp_set_max_active_levels
.. autofunction:: omp4py.omp_get_max_active_levels
.. autofunction:: omp4py.omp_set_nested
.. autofunction:: omp4py.omp_get_nested
```

### Scheduling

`omp_set_schedule` and `omp_get_schedule` control the schedule used by `schedule(runtime)`.
The kind is one of the constants below (optionally combined with `omp_sched_monotonic`).

```{eval-rst}
.. autofunction:: omp4py.omp_set_schedule
.. autofunction:: omp4py.omp_get_schedule
.. autodata:: omp4py.omp_sched_static
   :no-value:
.. autodata:: omp4py.omp_sched_dynamic
   :no-value:
.. autodata:: omp4py.omp_sched_guided
   :no-value:
.. autodata:: omp4py.omp_sched_auto
   :no-value:
.. autodata:: omp4py.omp_sched_monotonic
   :no-value:
```

### Cancellation

```{eval-rst}
.. autofunction:: omp4py.omp_get_cancellation
```

## Routines in development

These routines of the OpenMP 4.5 API are not available yet.

```{list-table}
:header-rows: 1
:widths: 25 75

* - Group
  - Routines
* - Execution environment
  - `omp_get_num_procs`, `omp_in_final`
* - Thread affinity
  - `omp_get_proc_bind`, `omp_get_num_places`, `omp_get_place_num_procs`,
    `omp_get_place_proc_ids`, `omp_get_place_num`, `omp_get_partition_num_places`,
    `omp_get_partition_place_nums`
* - Tasking
  - `omp_get_max_task_priority`
* - Teams and devices
  - `omp_get_num_teams`, `omp_get_team_num`, `omp_set_default_device`, `omp_get_default_device`,
    `omp_get_num_devices`, `omp_is_initial_device`, `omp_get_initial_device`
* - Device memory
  - `omp_target_alloc`, `omp_target_free`, `omp_target_is_present`, `omp_target_memcpy`,
    `omp_target_memcpy_rect`, `omp_target_associate_ptr`, `omp_target_disassociate_ptr`
* - Lock routines
  - `omp_init_lock`, `omp_init_nest_lock`, `omp_init_lock_with_hint`,
    `omp_init_nest_lock_with_hint`, `omp_destroy_lock`, `omp_destroy_nest_lock`, `omp_set_lock`,
    `omp_set_nest_lock`, `omp_unset_lock`, `omp_unset_nest_lock`, `omp_test_lock`,
    `omp_test_nest_lock`
* - Timing
  - `omp_get_wtime`, `omp_get_wtick`
```

Meanwhile, use `time.perf_counter()` for wall-clock time and `critical` (or Python's
`threading.Lock`) for mutual exclusion.
