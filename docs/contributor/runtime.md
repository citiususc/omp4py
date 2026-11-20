# The Runtime

`omp4py.runtime` is what generated code calls while the program executes. This page explains how
it is organised, how **one source** becomes both the pure and the native runtime, and how the
native runtime and your compiled code are built.

```{contents}
:local:
:depth: 2
```

## How it works

### Per-thread state

Every thread owns a `TaskContext` (`tasks/context.py`), reachable through `omp_ctx()`. It holds a
stack of `Task` objects (the innermost is the one running), the thread's ICVs and its
threadprivate storage. The initial thread gets an implicit task on demand; the threads created by
a parallel region get theirs before they start.

### Parallel regions

`tasks/parallelism.py::parallel(f, active, num_threads, proc_bind, copyin)`:

1. copies the current ICVs and computes the team size,
2. allocates a `SharedContext` (lock-free shared state for the team) and a reusable `Barrier`,
3. for threads `team_size-1 … 1` creates a new `TaskContext`, pushes a `ParallelTask`, and starts a
   `threading.Thread`,
4. runs `f` in the calling thread as thread 0,
5. joins the others.

There is no thread pool yet: every region creates its threads.

```{mermaid}
sequenceDiagram
    participant G as generated code
    participant P as _omp.parallel
    participant C as TaskContext (thread 0)
    participant W as worker threads
    G->>P: parallel(f, active, nthreads, bind, copyin)
    P->>C: copy ICVs, set team size
    P->>W: start threads 1..N-1 with ParallelTask
    P->>C: run f() as thread 0
    W->>W: run f()
    W-->>P: join
    P-->>G: return
```

### Worksharing loops

The transformer emits (see {doc}`/user_guide/debugging`):

```python
bounds = _omp.for_bounds(collapse)
bounds.rs[...] = (start, stop, step, ...)         # one triple per collapsed loop
_omp.for_init(bounds, modifier, schedule_char, chunk, ordered)
while _omp.for_next_static(bounds):               # or _dynamic / _guided / _runtime
    for i in range(bounds.init, bounds.end): ...  # the user's loop over one chunk
_omp.for_end(wait)
```

- `for_init` computes the chunk size (`ceil(iterations / team_size)` for `static` without a chunk
  size, `1` otherwise) and registers the shared state of the team.
- `for_next_static` hands each thread its blocks by arithmetic; `for_next_dynamic` and
  `for_next_guided` take chunks from a shared atomic counter.
- `for_last` tells a thread whether it ran the last iteration (for `lastprivate`).
- `ordered_start` / `ordered_next` serialise `ordered` blocks by iteration.
- `sections_*` and `single_*` follow the same init/next/end pattern; `single_copy_*` implement
  `copyprivate`.

### Synchronisation

- `barrier()`: a team barrier. While waiting, threads may run queued tasks.
- `critical_lock/unlock` use one global `Mutex`; named regions use `critical_named`, which
  creates or finds a `Mutex` by name.
- `masked(thread_id)` is the test used by `master` and `masked`.

### Internal control variables (ICVs)

`runtime/icvs/icvs.py` defines the OpenMP ICVs as small classes (`Device`, `Global`,
`ImplicitTask`, `Data`), each with a `copy()` used when entering a region. `icvs/defaults.py`
reads the `OMP_*` environment variables once at import time and fills `icvs.defaults`;
`icvs/places.py` parses `OMP_PLACES`.

## One source, two runtimes

The runtime is written **once**, in Python, and can be used in two forms:

```{mermaid}
flowchart LR
    subgraph low["lowlevel modules"]
        L1["atomic.py / numeric.py / mutex.py"]
        L2["atomic.pyx / numeric.pyx / mutex.pyx"]
    end
    subgraph other["all other modules"]
        M1["module.py<br/>(common implementation)"]
        M2["module.pxd<br/>(C types)"]
    end
    L1 -- "pure mode" --> PURE["Pure runtime"]
    M1 -- "pure mode" --> PURE
    L2 -- "Cython" --> NAT["Native runtime"]
    M1 -- "Cython + .pxd types" --> NAT
    M2 --> NAT
```

- The **low-level modules** (`runtime/lowlevel`: atomics, mutexes, thread-local storage,
  numeric types) have *two* implementations: a Python one (`.py`, built on `threading`) and a Cython
  one (`.pyx`, built on CPython's internal `PyMutex`/`PyEvent` and C atomics). They expose the
  same interface.
- **All the other modules** have **one** common implementation, the `.py` file.
  - In **pure mode** the `.py` file is imported as ordinary Python.
  - In the **native runtime** the same `.py` file is compiled with Cython, which uses the C
    types declared in the matching `.pxd` file.

### Keep them consistent

If you modify the runtime (which is **not recommended** unless you are working on OMP4Py
itself), the change must be valid in both worlds:

- The `.py` and the `.pxd` must agree: a function added to the `.py` needs its declaration in the
  `.pxd`, with compatible types.
- The code must be **cythonizable**: it is compiled by Cython, so use only constructs Cython
  supports and keep type declarations coherent.
- A new module is not compiled until it is added to the `extensions` list of `setup.py`:

```python
extensions: list[str] = [
    # api
    "omp4py/runtime/api/threadteam.py",
    # tasks
    "omp4py/runtime/tasks/mastersync.py",
    "omp4py/runtime/tasks/stopwatch.py",      # <- your new module
    ...
]
```

```{note}
This process applies **only to OMP4Py itself and to forks of it**. If you design an
{doc}`extension <extensions/index>` that OMP loads (devices, subparsers, modifiers), you are free
to organise and distribute it as you like: it does not have to follow this `.py` / `.pxd` /
`setup.py` scheme, and it does not need to be compiled at all.
```

### Marker comments: `BEGIN_CYTHON_*`

A runtime `.py` file is Python source and, at the same time, the input of Cython. Two pairs of
comments let a file say which parts are for which world. They are processed by `setup.py` before
Cython sees the file.

**`# BEGIN_CYTHON_IMPORTS … # END_CYTHON_IMPORTS`**: the imports inside the block are normal
Python imports in pure mode, but are rewritten as **`cimport`s** from the module's `.pxd` when
compiling:

```python
# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array
from omp4py.runtime.tasks.context import omp_ctx
# END_CYTHON_IMPORTS
```

becomes, for the compiler:

```python
from cython.cimports.omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array
from cython.cimports.omp4py.runtime.tasks.context import omp_ctx
```

so Cython links directly to the C declarations in the `.pxd` files instead of calling through
Python.

**`# BEGIN_CYTHON_IGNORE … # END_CYTHON_IGNORE`**: the block is **removed** when compiling. It
holds Python-only code that would conflict with the declarations of the `.pxd`: for example
module-level annotations of names that the `.pxd` already declares.

```python
# BEGIN_CYTHON_IGNORE: conflict with pxd declarations
type omp_sched_t = pyint
omp_sched_static: omp_sched_t
# END_CYTHON_IGNORE
```

Rules: the markers must start at the beginning of a line (they cannot be inside indented code),
and the block between them must be valid Python.

## Compiling user code

`@omp(compile=True)` and `omp(pkg=..., compile=True)` use the same machinery
(`core/preprocessor/cbuild.py::cythonize`) on the *transformed* code of the user:

```{mermaid}
flowchart TB
    A["process_object(fn, opt)"] --> B["hash of the source file →<br/>cache name p&lt;hash&gt;_&lt;module&gt;_&lt;fn&gt;"]
    B --> C{"extension already<br/>in opt.cache?"}
    C -- yes --> L["load the extension"]
    C -- "no / ignore_cache" --> D["process(): transform the AST"]
    D --> E["insert: from cython.cimports.omp4py import runtime as _omp"]
    E --> F["cythonize(): unparse → temporary .py → Cython → build_ext"]
    F --> L
    L --> R["return the compiled object"]
```

- The transformed code imports the runtime as `cython.cimports.omp4py.runtime`, so compiled user
  code links against the **native runtime** (that is why the native runtime must be built first).
- The extension is stored in `Options.cache` (`OMP4PY_CACHE_DIR`), named after a SHA-256 prefix
  of the **whole source file** that defines the function (plus its module and name), so editing anything in that file triggers a rebuild.
- Cython directives set by default: `freethreading_compatible=True`, `annotation_typing=True`,
  `infer_types=True`; override with `compiler_args={"compiler_directives": {...}}`.
  With `debug=True` the annotation report is kept.
- On Windows with MSVC the build adds `/std:c11` and `/experimental:c11atomics`.
- `Cython` and `setuptools` are optional dependencies: if missing, an `ImportError` explains
  `pip install "omp4py[compile]"`.
- Without `compile`, the transformed code is written as `<cache>/<name>.py`, byte-compiled and
  executed in the original module's namespace with `_omp` set to `omp4py.runtime`.

For packages, the same transformation is applied per module by the import loaders
({doc}`loaders`).

## Pure mode

When `Options.pure` is true (`OMP4PY_PURE`), `omp4py.core` imports `core/imports/pure.py`, which:

1. inserts `PureImport` in `sys.meta_path`: for any `omp4py.runtime.*` module whose spec points to
   a compiled extension, it swaps in a source loader for the `.py` twin;
2. registers `omp4py.core.imports.cython` as `sys.modules["cython"]` (a minimal pure-Python
   stand-in for Cython's `cython` module), so the `.py` runtime can `import cython` without Cython
   installed.

This is why pure mode works even in an environment with a compiled build.

## The cache

| Location | Contents | Cleared by |
| --- | --- | --- |
| `Options.cache` (`OMP4PY_CACHE_DIR`) | Transformed function sources and compiled function extensions | `OMP4PY_IGNORE_CACHE=1`, `ignore_cache=True`, deleting the directory |
| `<package>/__omp__/` | Transformed package modules ({doc}`loaders`) | Deleting the folder |

```{tip}
While developing a transformer, set `OMP4PY_IGNORE_CACHE=1`. The cache key is only the
*source file* of the user function, not the version of the transformer.
```
