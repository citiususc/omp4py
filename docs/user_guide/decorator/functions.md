# Decorating Functions and Classes

The most direct way to use OMP4Py is to decorate the functions or classes that contain
directives.

```{literalinclude} ../../examples/decorator/function_and_class.py
:language: python
```

```text
45 45 45
```

The three forms are equivalent: `@omp` on a function, `@omp` on a class (all its methods are
processed), and a plain call `omp(function)` on an object that already exists.

## What happens, step by step

```{mermaid}
sequenceDiagram
    participant Py as Python
    participant D as omp (decorator)
    participant C as Cache
    participant R as Runtime
    Py->>D: def f(...) is evaluated: omp(f)
    D->>D: read the source of f
    D->>C: transformed code for this source already cached?
    alt not cached (or ignore_cache)
        D->>D: parse directives, transform, write .py/.pyc
        D->>C: store
    end
    D->>Py: replace f by the transformed function
    Py->>R: f() runs: calls the OMP4Py runtime
```

1. The decorator runs **at the moment the function is defined** (at import time for a module
   function). There is no separate build step: the code is processed "on the fly".
2. OMP4Py reads the source of the object, finds the directives and transforms them.
3. The result is stored in the cache directory (a `.py` file and its bytecode), keyed by a
   hash of the source file that defines the function. The next run with the same file skips the transformation and loads the
   cached code, so only the first run pays for the processing.
4. The original function is replaced by the transformed one.

The cache directory is `OMP4PY_CACHE_DIR` (see {doc}`/user_guide/environment`). Pass
`ignore_cache=True` to regenerate everything, for example while developing an extension.

## With compilation

```python
@omp(compile=True)
def kernel(n): ...
```

compiles **only this function** into a native extension (also cached). See
{doc}`/user_guide/compiled_mode` for what this does and does not optimise, and
{doc}`packages` for compiling a whole module instead.

## What the decorator requires

- **Source code must be available.** Functions defined with `exec()`, in the interactive
  prompt or from a string cannot be processed (`OSError: could not get source code`).
- **Named definitions only.** Lambdas are rejected.
- **Innermost decorator.** If you stack decorators, `@omp` goes closest to `def`.
- **Directives that live outside functions in C** (for example `threadprivate`, see
  {doc}`/user_guide/directives`) must be written *inside* a decorated function when you use the
  decorator form. In {doc}`package mode <packages>` you can write them at module level.
- **Functions of the same module share some state.** Reductions and threadprivate variables
  declared in one decorated function are known to the other decorated functions of the same
  file.

```{tip}
If you decorate many functions of the same module, or you want to compile it, switch to
{doc}`package mode <packages>`: no decorators are needed.
```
