# Import Loaders

When a package is marked with `omp(pkg=__package__)` ({doc}`/user_guide/decorator/packages`),
its modules are processed **on the fly, while they are imported**. This is done by a custom
import finder and two loaders in `omp4py/core/imports/loader.py`.

## Installing the hook

`omp(pkg="mypkg")` calls `set_omp_package`, which:

1. stores the `Options` given to `omp(...)` for that package,
2. installs `Omp4pyFinder` at the **front** of `sys.meta_path` (once).

Nothing is transformed yet. Modules of other packages are never touched.

## What happens on `import mypkg.kernels`

```{mermaid}
sequenceDiagram
    participant U as import mypkg.kernels
    participant F as Omp4pyFinder
    participant PF as normal PathFinder
    participant L as Omp4pyLoader / Omp4pyCythonLoader
    participant P as preprocessor
    U->>F: find_spec("mypkg.kernels")
    F->>PF: find the module normally
    PF-->>F: spec (SourceFileLoader)
    F->>F: package registered → replace the loader
    F-->>U: spec with an OMP loader
    U->>L: exec the module
    L->>L: valid cached code in __omp__/__pycache__ ?
    alt no cached code
        L->>P: process_source(source)
        P-->>L: transformed code (also written to __omp__/kernels.py)
    end
    L-->>U: module executed
```

The finder asks the normal machinery for the module and, if it is a plain source module of a
registered package, swaps the loader:

| `compile` option | Loader | What it does |
| --- | --- | --- |
| `False` | `Omp4pyLoader` (a `SourceFileLoader`) | Transforms the source with the preprocessor, writes the result to `<package>/__omp__/<module>.py`, and executes that code. |
| `True` | `Omp4pyCythonLoader` (an `ExtensionFileLoader`) | Makes sure the transformed source exists (using `Omp4pyLoader`), then builds a Cython extension from it, or reuses an existing one. |

## The `__omp__` folder

`OMP_FOLDER = "__omp__"`. The loader redirects everything to it:

- The transformed source of each module: `__omp__/<module>.py`.
- The bytecode cache, redirected from `__pycache__` to `__omp__/__pycache__`, so the normal
  cache of the original modules is never mixed with the transformed one.
- With `compile=True`, the compiled extension (`.so` / `.pyd`) and, with `debug=True`, the Cython
  annotation HTML.

```text
mypkg/
├── __init__.py            # omp(pkg=__package__)
├── kernels.py             # your source
└── __omp__/
    ├── kernels.py         # transformed source
    ├── kernels.cpython-314t-x86_64-linux-gnu.so   # only with compile=True
    └── __pycache__/       # bytecode of the transformed source
```

## Reuse and rebuilding

- **Python loader.** It relies on Python's own bytecode caching, redirected to `__omp__`. If valid
  bytecode exists, the preprocessor is not invoked; otherwise `source_to_code` transforms the
  source and records that the module was regenerated (`need_recompile = True`).
- **Cython loader.** It first runs the Python loader's logic to know whether the transformed
  source changed. If it did not change and a compiled extension already exists in `__omp__`, the
  extension is **reused**; otherwise `cythonize` builds it again.
- `ignore_cache=True` makes the loader ignore cached bytecode.

## Distributing a library

Because everything the loader produces lives in `__omp__`, a library can be processed ahead of
time and shipped with that folder: its users load the ready-made modules, and, if you included
compiled extensions, they do not need to compile anything. See
{doc}`/user_guide/decorator/files`.

Things to check before publishing:

- The reuse of the cached bytecode depends on Python's usual validation against the original
  source file, so test the installed package in a clean environment.
- Compiled extensions are specific to a platform and a Python version.
- Include the `__omp__` folder in your package data.

## Extension loading

`omp(extension="my_ext")` does not use these loaders: it only records the module name. The
modules are imported by `load_extensions()` the first time the preprocessor runs
({doc}`extensions/index`).
