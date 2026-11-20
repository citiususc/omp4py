# Package Mode

Instead of decorating functions one by one, you can mark a **package**. Every module of the
package is processed automatically the first time it is imported, and **no decorators are
needed** inside.

```{literalinclude} ../../examples/pkg_example/mypkg/__init__.py
:language: python
```

```{literalinclude} ../../examples/pkg_example/mypkg/kernels.py
:language: python
```

```{literalinclude} ../../examples/pkg_example/main.py
:language: python
```

```text
500
```

`omp(pkg=__package__)` is called in the package's `__init__.py`. It processes nothing
immediately: it installs an import hook for the package (see {doc}`/contributor/loaders`), and
each module is transformed when it is imported.

```{important}
Call `omp(pkg=...)` **before** the modules of the package are imported. Modules imported
earlier are not processed retroactively.
```

## Why it is the preferred mode for applications and libraries

| Advantage | Explanation |
| --- | --- |
| No decorators | You write plain code with `with omp(...)`; the whole module is covered. |
| More efficient | The module is processed once and loaded from the cache (`.pyc`) afterwards, instead of reading the source of each function. |
| Better compilation | With `compile=True` the **whole module** is compiled, so Cython sees the real module: calls between functions, globals and constants can be optimised. See {doc}`/user_guide/compiled_mode`. |
| Module-level directives | Directives that C puts outside functions work here: `omp("threadprivate(x)")` at the top of the module. |

```python
# mypkg/__init__.py
from omp4py import omp

omp(pkg=__package__, compile=True)       # compile every module of the package
```

All the arguments of {doc}`index` are accepted (`compile`, `device`, `parsers`, `modifiers`,
...); they apply to every module of the package.

## Module-level directives

```python
# mypkg/state.py
from omp4py import *

counter = 0
omp("threadprivate(counter)")      # outside any function: only possible in package mode
```

## Where the results go

Transformed sources and their bytecode are written to a folder named `__omp__` **next to the
original modules**, so the normal `__pycache__` is left untouched. With `compile=True` the
compiled extension is placed there too. This is what makes it possible to ship a library
already processed: see {doc}`files`.

```text
mypkg/
├── __init__.py
├── kernels.py
└── __omp__/
    ├── kernels.py          <- transformed source
    └── __pycache__/        <- its bytecode
```
