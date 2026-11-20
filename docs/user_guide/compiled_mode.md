# Compiled Mode

The compiled modes translate your code to C with [Cython](https://cython.org/) and build a
native extension. This page explains what is compiled, how to get real speedups, and how to
check that the generated code is efficient.

```{contents}
:local:
:depth: 2
```

## Requirements

```bash
pip install "omp4py[compile] @ git+https://github.com/citiususc/omp4py.git"
```

You also need a C compiler (GCC or Clang on Linux/macOS, MSVC on Windows) and the native
OMP4Py runtime: compiled user code links directly against it.

## Two ways to compile

### Compiling a function (or class)

```python
from omp4py import omp

@omp(compile=True)
def pi(n):
    w = 1.0 / n
    total = 0.0
    with omp("parallel for reduction(+:total)"):
        for i in range(n):
            x = (i + 0.5) * w
            total += 4.0 / (1.0 + x * x)
    return total * w
```

The decorated object is processed at definition time and **only that function** is compiled into
its own extension module, which is cached. Everything else it uses (other functions of your
module, global variables, imported modules) stays Python: the compiled function reaches them as
Python objects, so each use goes through the interpreter and the optimisation is **limited**.

```{mermaid}
flowchart LR
    subgraph py["Your module (Python)"]
        G["globals, imports,<br/>other functions"]
        F["pi(...)  <- compiled"]
    end
    F -- "Python calls /<br/>object access" --> G
```

### Compiling a module (or a whole package)

```python
# mypkg/__init__.py
from omp4py import omp

omp(pkg=__package__, compile=True)
```

In {doc}`package mode <decorator/packages>` each **module** is compiled as a whole. Cython sees
your functions, constants and globals together, so calls between compiled functions and access
to module-level data can be optimised. This is **much more efficient** than compiling isolated
functions and is the recommended way to compile applications and libraries.

| | Function level | Module level |
| --- | --- | --- |
| Written with | `@omp(compile=True)` | `omp(pkg=__package__, compile=True)` |
| What is compiled | The decorated function | Every module of the package |
| Calls to your other functions | Python calls | Can be optimised by Cython |
| Optimisation | Limited | Much more |
| Decorators needed | Yes | No |

The first run pays the compilation cost; later runs reuse the compiled result. Function
extensions are stored in the cache directory (`OMP4PY_CACHE_DIR`); module extensions are placed in
the package's `__omp__` folder ({doc}`decorator/files`).

## Types matter

By default Cython does not know what your variables are, so it treats them as generic Python
objects. Every operation then keeps the overhead of the dynamic type system: reference counting
and boxed numbers. The code is native, but it behaves almost like interpreted code.

Annotate numeric variables with `int` and `float` and Cython can use native C arithmetic:

```python
@omp(compile=True)
def pi(n: int) -> float:
    w: float = 1.0 / n
    total: float = 0.0
    with omp("parallel for reduction(+:total)"):
        for i in range(n):
            x: float = (i + 0.5) * w
            total += 4.0 / (1.0 + x * x)
    return total * w
```

In the benchmarks of the OMP4Py paper, compiled mode without annotations was on average about
2.5× faster than pure mode with 32 threads, and with explicit native types it was on average
almost three orders of magnitude faster (785×), with the best scalability. The annotations are
what turns "native code" into "fast native code".

Guidelines:

- Annotate the variables that appear in the hot loop: accumulators, intermediates, bounds.
- Keep types consistent: a variable annotated `float` should not receive other types.
- Types only help code that works with numbers. For code that manipulates strings,
  dictionaries or other objects, compilation helps less (the work is done by Python objects).

## Check that the code is efficient: the HTML report

Cython can produce an **annotated HTML report** that shows, line by line, how much
interaction with Python each line has. Activate it with `debug=True`:

```python
@omp(compile=True, debug=True)
def pi(n: int) -> float: ...
```

For a compiled function the report is written next to its extension, in the cache directory,
with the same name and an `.html` extension (for a package, in the `__omp__` folder). Open it in a
browser:

- **White lines** are pure C: what you want in the hot loop.
- **Yellow lines** interact with the Python C-API; the darker, the more. Click a line to see the
  C code generated for it.

If the loop of your kernel is yellow, add type annotations or simplify the expression until it
becomes white. The report also helps you see what the OMP4Py transformation generated
({doc}`debugging`).

## Compiler options

`compiler_args` is handed to `cythonize`. By default OMP4Py sets `annotation_typing=True`,
`infer_types=True` and `freethreading_compatible=True`; you can add others, for example
`cdivision`:

```python
@omp(compile=True, compiler_args={"compiler_directives": {"cdivision": True}})
def kernel(n: int): ...
```

## Limits to keep in mind

- Compiled code cannot be inspected with a Python debugger line by line; use {doc}`pure mode
  <execution_modes>` to debug and compile when the logic is right.
- Use the same Python (and free-threaded or not) to build and to run.
- A compiled extension is built for one platform and one Python version.
