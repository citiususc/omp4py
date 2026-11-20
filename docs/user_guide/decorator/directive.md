# Directives and the `alias`

The first job of `omp` is to be the **container of directives**: it carries the OpenMP
directive as a string, so that your code stays valid Python.

## `with omp(...)` and `omp(...)`

```python
with omp("parallel for reduction(+:total)"):   # directive applied to a block
    for i in range(n):
        total += i

omp("barrier")                                  # stand-alone directive (a statement)
```

At run time `omp("...")` does nothing: it returns an empty context manager. It only has
meaning for the preprocessor, which replaces the whole statement by parallel code when the
enclosing function is processed ({doc}`functions`, {doc}`packages`).

Rules:

- The argument is **one constant string** (no f-strings, no variables).
- The call is either a statement or the **only item** of a `with`: `with omp("..."), other():`
  and `with omp("...") as x:` are errors.
- Directive and clause names are case-insensitive.
- The directives are only recognised inside code that is being processed.

The directives themselves are described in {doc}`/user_guide/directives`.

## The `alias`

The preprocessor does not run your code; it **reads its syntax**. To find a directive it looks
for calls to a *name*, and that name is the `alias` argument, `"omp"` by default. The same
name is used to find the decorator:

```python
@omp                      # decorator named "omp"
def f():
    with omp("parallel"):  # directive named "omp"
        ...
```

An attribute access also matches, so `omp4py.omp("parallel")` works as well, because its last
component is `omp`.

### Using another name

If you want a decorator with its own name or its own default options, create it with
`omp(**options)` and **set `alias` to that same name**. Otherwise the preprocessor will not
recognise your directives:

```{literalinclude} ../../examples/decorator/with_alias.py
:language: python
```

```text
45
```

```{important}
Without `alias="fast"` the call `fast("parallel for ...")` would be an ordinary call to an
unknown function: it would not be processed and, at run time, `fast(...)` would simply
return an empty context manager. The same applies to `from omp4py import omp as openmp`:
use `omp(alias="openmp")` for it, or keep the name `omp`.
```

### Decorators with defaults

`omp(**options)` returns a new decorator that remembers the options. Every function decorated
with it uses them, and each decoration may add or override options:

```python
from omp4py import omp

fast = omp(alias="fast", compile=True, modifiers={"strip_asserts": True})

@fast                         # compile=True, strip_asserts on
def kernel(...): ...

@fast(ignore_cache=True)      # same defaults + this option
def other_kernel(...): ...
```

With `args_append=True` the collection arguments (`parsers`, `modifiers`, ...) are merged with
the defaults instead of replacing them:

```python
fast(parsers=["timed"], args_append=True)
```

### Several decorators in one program

Different names let you mix configurations (for example a compiled one and a debugging one),
each with its own alias:

```python
fast = omp(alias="fast", compile=True)
debug = omp(alias="debug", dump="out.py", ignore_cache=True)
```

Keep the decorator as the **innermost** one (closest to the `def`) when you stack
decorators; otherwise OMP4Py reports `Invalid decorator order`.
