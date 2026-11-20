# Environment Variables

Two groups of environment variables influence OMP4Py: the standard OpenMP `OMP_*` variables,
read by the runtime, and a few `OMP4PY_*` variables read by the preprocessor.

## OpenMP variables

OMP4Py implements the **OpenMP 4.5 environment variables** and they behave as the standard
defines: each one initialises an *internal control variable* (ICV) when the runtime starts, an
invalid value is reported on standard error and ignored (the default is used), and the value can
be changed afterwards with the routines of {doc}`runtime_api` (for example
`omp_set_num_threads` overrides `OMP_NUM_THREADS`).

| Variable | Values | ICV it sets |
| --- | --- | --- |
| `OMP_NUM_THREADS` | `4` or `4,2` | Team size per nesting level |
| `OMP_SCHEDULE` | `[modifier:]kind[,chunk]` | Schedule used by `schedule(runtime)` |
| `OMP_DYNAMIC` | `true` / `false` | Dynamic adjustment of the team size |
| `OMP_PROC_BIND` | `true`, `false`, `master`, `close`, `spread` (list) | Thread affinity policy |
| `OMP_PLACES` | `threads`, `cores`, `sockets`, `{0,1},{2,3}`, `{0}:4:2` | Places for affinity |
| `OMP_NESTED` | `true` / `false` | Nested parallelism (deprecated by OpenMP) |
| `OMP_MAX_ACTIVE_LEVELS` | positive integer | Maximum nested active parallel regions |
| `OMP_THREAD_LIMIT` | positive integer | Maximum number of threads of the program |
| `OMP_STACKSIZE` | `N[B\|K\|M\|G]` | Stack size of the created threads |
| `OMP_WAIT_POLICY` | `active` / `passive` | How waiting threads behave |
| `OMP_CANCELLATION` | `true` / `false` | Enable cancellation |
| `OMP_DEFAULT_DEVICE` | integer | Default device number |
| `OMP_MAX_TASK_PRIORITY` | integer | Maximum task priority |
| `OMP_DISPLAY_ENV` | `true` / `verbose` | Print the ICVs at start-up |

```bash
OMP_NUM_THREADS=8 OMP_SCHEDULE="dynamic,4" python program.py
```

```{note}
`OMP_PROC_BIND` and `OMP_PLACES` are read into the ICVs, but binding threads to places is not
implemented yet.
```

### `OMP_DISPLAY_ENV`

Set it to `true` (or `verbose`) to print the values the runtime resolved from the environment,
using the format of the OpenMP specification. It is the quickest way to check that your settings
were understood:

```bash
OMP_DISPLAY_ENV=verbose OMP_NUM_THREADS=3 python -c "from omp4py import *"
```

```text
OPENMP DISPLAY ENVIRONMENT BEGIN
_OPENMP = '201511'
  OMP_SCHEDULE = 'STATIC'
  OMP_NUM_THREADS = '3'
  OMP_DYNAMIC = 'FALSE'
  OMP_PROC_BIND = 'FALSE'
  OMP_NESTED = 'FALSE'
  OMP_WAIT_POLICY = 'PASSIVE'
  OMP_MAX_ACTIVE_LEVELS = '2147483648'
  OMP_THREAD_LIMIT = '2147483648'
  OMP_CANCELLATION = 'FALSE'
  OMP_DEFAULT_DEVICE = '0'
OPENMP DISPLAY ENVIRONMENT END
```

`_OPENMP = '201511'` is the date of OpenMP 4.5, the version OMP4Py implements.

## OMP4Py variables

Set them **before** importing `omp4py`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMP4PY_PURE` | `False` | Use the pure Python runtime even when the compiled one is available ({doc}`/getting_started/concepts`) |
| `OMP4PY_CACHE_DIR` | OS cache dir + `omp4py` | Where transformed code and compiled functions are stored |
| `OMP4PY_IGNORE_CACHE` | `False` | Default for the `ignore_cache` argument |
| `OMP4PY_DUMP` | `False` | Default for the `dump` argument (`1` writes `<file>_<name>_omp_dump.py` next to the source, one per decorated function or class) |
| `OMP4PY_DEBUG` | `False` | Default for the `debug` argument |

Boolean variables accept `1`, `true`, `yes` or `on` (case-insensitive).

The default cache directory is `%TMP%\omp4py` on Windows, `~/Library/Caches/omp4py` on macOS and
`~/.cache/omp4py` on Linux, with `~/.omp4py` as a last resort.
