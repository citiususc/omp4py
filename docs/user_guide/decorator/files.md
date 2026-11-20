# File Preprocessing and Distributing Libraries

```python
from omp4py import omp

generated = omp(py="mypkg/kernels.py", compile=False)
print(generated)        # path of the processed file, inside the __omp__ folder
```

`omp(py="file.py")` processes **one file ahead of time**. The result is written inside an
`__omp__` folder next to the file and the call returns its path. It accepts the same arguments
as the other forms ({doc}`index`).

It exists mainly for **library authors**, and it is meant to be called from your **`setup.py`**,
at build time, in the same way that `setuptools` projects call `cythonize(...)` to process their
`.pyx` files before building the package.

## Use it in `setup.py`

`cythonize` turns your sources into something that is shipped with the distribution. `omp(py=...)`
plays the same role for files that contain OMP4Py directives: you call it once per module while
the package is being built, and what it generates goes into the distribution.

```python
# setup.py
from pathlib import Path

from omp4py import omp

# Process every module of the package that contains directives (like cythonize(...) would).
for module in Path("mypkg").glob("*.py"):
    if module.name != "__init__.py":
        omp(py=str(module))
```

Compare with the usual Cython setup:

| Cython | OMP4Py |
| --- | --- |
| `ext_modules=cythonize("mypkg/*.pyx")` in `setup.py` | `omp(py="mypkg/module.py")` in `setup.py` |
| Generates C/extension files before the build | Generates the transformed modules in `__omp__` before the build |

The arguments of {doc}`index` are accepted, so the same call decides how the library is processed
(`omp(py=..., device=..., modifiers=...)`).

## Why it matters for libraries

When your users run your library, the first import of a package in {doc}`package mode
<packages>` has to read the sources, parse the directives and transform them (and compile them,
with `compile=True`). You can do all that work once, before publishing, and ship the result:

```{mermaid}
flowchart LR
    subgraph dev["You (library author)"]
        S["mypkg/*.py<br/>with omp(...) directives"] --> P["omp(pkg=...) + import<br/>or omp(py=...)"]
        P --> O["mypkg/__omp__/<br/>transformed source, bytecode,<br/>(compiled extensions)"]
    end
    O --> W["wheel / sdist"]
    subgraph user["Your user"]
        W --> I["import mypkg"]
        I --> L["the package loader finds<br/>the __omp__ results first"]
    end
```

- The package loader works with the `__omp__` folder: it reuses the processed module (and, with
  `compile=True`, its compiled extension) **when they are up to date**, instead of
  transforming or compiling again.
- Shipping compiled extensions means your users can run the compiled version without having to
  build anything.
- Shipping the transformed sources and bytecode saves them the preprocessing step on first import.
- The original sources stay untouched, so a user can always reprocess the library with other
  options (`ignore_cache=True`).

```{note}
Whether a cached module is considered up to date depends on the files in `__omp__` and on the
original source (see {doc}`/contributor/loaders`). Test the shipped package in a clean
environment before publishing it.
```

## Combine it with package mode

The `__omp__` results are consumed by the package loader, so the library still declares its package:

```python
# mypkg/__init__.py
from omp4py import omp

omp(pkg=__package__)          # or omp(pkg=__package__, compile=True)
```

See {doc}`/contributor/loaders` for how the loader decides what to reuse.
