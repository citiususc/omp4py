# OpenMP 4.5 Coverage

OMP4Py targets the **OpenMP 4.5** specification (and accepts the OpenMP 6.0 names of combined
directives, see {doc}`/user_guide/combined_directives`). This page lists, feature by feature,
whether it is available **today**. A feature that is not listed as supported cannot be used.

Legend: ✅ supported · ❌ not supported yet · ◐ partially supported (see the note).

## Directives

### Parallelism and worksharing

| Directive | Status | Note |
| --- | :-: | --- |
| `parallel` | ✅ | |
| `for` | ✅ | Loops over `range(...)` |
| `sections` / `section` | ✅ | |
| `single` | ✅ | |
| `parallel for` | ✅ | |
| `parallel sections` | ✅ | |
| `parallel single`, `parallel masked` | ✅ | OpenMP 6.0 names |
| `simd`, `for simd`, `parallel for simd` | ❌ | |
| `declare simd` | ❌ | |
| `workshare` | — | Fortran only |

### Tasking

| Directive | Status |
| --- | :-: |
| `task` | ❌ |
| `taskloop`, `taskloop simd` | ❌ |
| `taskyield` | ❌ |
| `taskwait` | ❌ |
| `taskgroup` | ❌ |

### Master and synchronisation

| Directive | Status | Note |
| --- | :-: | --- |
| `master` | ✅ | |
| `masked` | ✅ | With `filter` |
| `critical` | ✅ | Named and unnamed |
| `barrier` | ✅ | |
| `ordered` | ✅ | Block form inside a loop with the `ordered` clause |
| `atomic` | ❌ | |
| `flush` | ❌ | |
| `cancel`, `cancellation point` | ❌ | |

### Data environment

| Directive | Status |
| --- | :-: |
| `threadprivate` | ✅ |
| `declare reduction` | ✅ |

### Devices

| Directive | Status |
| --- | :-: |
| `target`, `target data`, `target enter data`, `target exit data`, `target update` | ❌ |
| `declare target` | ❌ |
| `teams` | ❌ |
| `distribute`, `distribute simd`, `distribute parallel for`, `distribute parallel for simd` | ❌ |
| Combined `target ...` / `teams ...` directives | ❌ |

## Clauses

| Clause | Status | Note |
| --- | :-: | --- |
| `if` | ✅ | With the directive-name modifier |
| `num_threads` | ✅ | |
| `default` | ✅ | |
| `shared`, `private`, `firstprivate`, `lastprivate` | ✅ | |
| `reduction` | ✅ | Built-in operators `+ - * & \| ^ and or max min`, and your own operators with `declare reduction` |
| `copyin`, `copyprivate` | ✅ | |
| `schedule` | ✅ | `static`, `dynamic`, `guided`, `auto`, `runtime`; modifiers `monotonic`, `nonmonotonic` |
| `collapse` | ✅ | |
| `ordered` | ✅ | |
| `nowait` | ✅ | |
| `filter` | ✅ | OpenMP 5.1 (`masked`) |
| `proc_bind` | ❌ | Accepted, but threads are not bound |
| `hint` | ❌ | |
| `linear`, `safelen`, `simdlen`, `aligned`, `uniform`, `inbranch`, `notinbranch` | ❌ | |
| `depend`, `final`, `untied`, `mergeable`, `priority`, `grainsize`, `num_tasks`, `nogroup` | ❌ | |
| `map`, `device`, `is_device_ptr`, `use_device_ptr`, `defaultmap`, `to`, `from`, `link` | ❌ | |
| `num_teams`, `thread_limit`, `dist_schedule` | ❌ | |

## Runtime library routines

| Group | Status | Routines |
| --- | :-: | --- |
| Thread team | ✅ | `omp_set_num_threads`, `omp_get_num_threads`, `omp_get_max_threads`, `omp_get_thread_num`, `omp_in_parallel`, `omp_get_thread_limit`, `omp_set_dynamic`, `omp_get_dynamic` |
| Nesting | ✅ | `omp_get_level`, `omp_get_active_level`, `omp_get_ancestor_thread_num`, `omp_get_team_size`, `omp_set_max_active_levels`, `omp_get_max_active_levels`, `omp_set_nested`, `omp_get_nested` |
| Scheduling | ✅ | `omp_set_schedule`, `omp_get_schedule` |
| Cancellation | ✅ | `omp_get_cancellation` |
| Execution environment | ❌ | `omp_get_num_procs`, `omp_in_final` |
| Thread affinity | ❌ | `omp_get_proc_bind`, `omp_get_num_places`, `omp_get_place_*`, `omp_get_partition_*` |
| Tasking | ❌ | `omp_get_max_task_priority` |
| Teams and devices | ❌ | `omp_get_num_teams`, `omp_get_team_num`, `omp_*_default_device`, `omp_get_num_devices`, `omp_is_initial_device`, `omp_get_initial_device` |
| Device memory | ❌ | `omp_target_*` |
| Locks | ❌ | `omp_init_lock`, `omp_set_lock`, `omp_unset_lock`, `omp_test_lock`, and the nested and `_with_hint` variants |
| Timing | ❌ | `omp_get_wtime`, `omp_get_wtick` |

## Environment variables

| Variable | Status | Note |
| --- | :-: | --- |
| `OMP_NUM_THREADS`, `OMP_SCHEDULE`, `OMP_DYNAMIC`, `OMP_NESTED`, `OMP_MAX_ACTIVE_LEVELS`, `OMP_THREAD_LIMIT`, `OMP_STACKSIZE`, `OMP_WAIT_POLICY`, `OMP_CANCELLATION`, `OMP_DEFAULT_DEVICE`, `OMP_MAX_TASK_PRIORITY`, `OMP_DISPLAY_ENV` | ✅ | Read into the ICVs |
| `OMP_PROC_BIND`, `OMP_PLACES` | ◐ | Read, but threads are not bound to places |

## OMP4Py execution modes

| Mode | Status |
| --- | :-: |
| Pure | ✅ |
| Hybrid | ✅ |
| Compiled | ✅ |
| Compiled with types | ✅ |
