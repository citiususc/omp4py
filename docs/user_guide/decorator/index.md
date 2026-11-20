# The `omp` Decorator

`omp` is the heart of OMP4Py. The same object is used to **write directives**, to **decorate
functions and classes**, to **mark whole packages**, to **preprocess files** and to **create
customised decorators**. Everything else in this guide (directives, clauses, execution modes)
is expressed through it.

```python
from omp4py import omp
```

(`from omp4py import *` also gives you `omp` together with the OpenMP runtime routines.)

## The forms of `omp`

```{list-table}
:header-rows: 1
:widths: 28 34 38

* - You write
  - It means
  - Read
* - `with omp("parallel for"):`
  - A **directive** inside a block
  - {doc}`directive`
* - `omp("barrier")`
  - A stand-alone **directive** (a statement)
  - {doc}`directive`
* - `@omp` / `@omp(compile=True)`
  - **Decorate** a function or class
  - {doc}`functions`
* - `omp(pkg=__package__)`
  - Process every module of a **package** when it is imported
  - {doc}`packages`
* - `omp(py="module.py")`
  - **Preprocess a file** ahead of time (to distribute libraries)
  - {doc}`files`
* - `my_omp = omp(alias="my_omp", ...)`
  - Create a **decorator with defaults**
  - {doc}`directive`
* - `omp(extension="my_ext")`
  - Register an **extension** module
  - {doc}`/contributor/extensions/index`
```

```{toctree}
:hidden:
:maxdepth: 1

directive
functions
packages
files
```

## Arguments

Every form that processes code accepts the following keyword arguments. They are stored as
the options of that preprocessing run.

```{list-table}
:header-rows: 1
:widths: 20 18 62

* - Argument
  - Default
  - Meaning
* - `alias`
  - `"omp"`
  - Name used in your source to write directives and the decorator. **Required whenever you
    use a name other than `omp`** (see {doc}`directive`).
* - `compile`
  - `False`
  - Compile the transformed code to native code with Cython ({doc}`/user_guide/compiled_mode`).
* - `compiler`
  - `"cython"`
  - Backend used when `compile=True`.
* - `compiler_args`
  - `{}`
  - Extra arguments for the compiler (for example `compiler_directives`).
* - `ignore_cache`
  - `False`
  - Ignore cached results and regenerate the code (and the compilation).
* - `dump`
  - `False`
  - Write the transformed source to a file: give a path, or `True` for `<file>_<name>_omp_dump.py` next to your source ({doc}`/user_guide/debugging`).
* - `debug`
  - `False`
  - Debug mode. With `compile=True` it also keeps the Cython annotation report.
* - `device`
  - `"omp"`
  - Transformation device; `"omp"` is the standard OpenMP one ({doc}`/contributor/extensions/devices`).
* - `parsers`
  - `[]`
  - Names of extra directive parsers (subparsers) to enable ({doc}`/contributor/extensions/subparsers`).
* - `modifiers`
  - `{}`
  - Enable or disable AST modifiers by name, for example `{"my_pass": True}` ({doc}`/contributor/extensions/modifiers`).
* - `modifiers_args`
  - `{}`
  - Free-form configuration passed to the modifiers.
* - `args_append`
  - `False`
  - When `True`, list/set/dict arguments are **merged** with the defaults of the decorator
    instead of replacing them.
```

The defaults of `ignore_cache`, `dump` and `debug` can be changed with environment variables
(`OMP4PY_IGNORE_CACHE`, `OMP4PY_DUMP`, `OMP4PY_DEBUG`; see {doc}`/user_guide/environment`).

## Choosing a form

```{mermaid}
flowchart TB
    Q{"What do you want<br/>to process?"}
    Q -- "a few functions<br/>or a script" --> F["@omp on each function/class"]
    Q -- "a whole package" --> P["omp(pkg=__package__)<br/>in __init__.py"]
    Q -- "a library you will<br/>distribute" --> D["package mode<br/>+ ship the __omp__ folder"]
    F --> C{"Need native speed?"}
    P --> C
    C -- "just the hot function" --> C1["@omp(compile=True)"]
    C -- "the whole module" --> C2["omp(pkg=..., compile=True)"]
```

| | Decorator on functions | Package mode |
| --- | --- | --- |
| Decorators needed | yes, one per function/class | **no** |
| When it is processed | when the function is defined | when the module is imported |
| Module-level directives (`threadprivate`, ...) | no, only inside functions | **yes** |
| Compilation scope | only the decorated function | the whole module: more optimisation |
| Best for | scripts, prototypes, a few kernels | applications and libraries |
