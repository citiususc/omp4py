# Frequently Asked Questions

## Do I need a special Python?

Python 3.13 or newer. To get real parallel speedups use a **free-threaded** build
(`3.13t`, `3.14t`). With a normal (GIL) build programs run correctly but CPU-bound code is
not faster.

## Do I need a C compiler?

Not to use OMP4Py in pure mode, nor in hybrid mode if you install a wheel with the
compiled runtime. You need a compiler, Cython and setuptools only for `compile=True` or to build
the project from source.

## Why do I get "'task' not implemented"?

The error is `SyntaxError: 'task' not implemented`.

Tasking is **under development**. It is part of the work to extend OMP4Py from OpenMP 3.0 to
the full **OpenMP 4.5** specification. In 4.5 tasks can declare dependencies between them
(`depend` clause), so it made no sense to ship a basic version of `task` first: the
implementation is being built directly with dependencies in mind.

The parser already understands the construct, but this version has no transformation for it, so
the preprocessor stops with this error. See the table in {doc}`/compatibility/coverage`.

## Why do I get "'target' not implemented"?

The error is `SyntaxError: 'target' not implemented`.

The device constructs (`target`, `target data`, `teams`, `distribute`, ...) are not available
yet. Like `task`, they are part of the extension to OpenMP 4.5 and are not implemented in this
version.

In short, OMP4Py supports OpenMP 4.5 as a target, and what is still **in development** to
complete it is:

- `target` and the device constructs.
- `task` with dependencies (`depend`) and the rest of the tasking constructs.

Everything else listed as supported in {doc}`/compatibility/coverage` works today.

## Why do I get "could not get source code"?

The error is `OSError: could not get source code`.

The preprocessor re-reads the source of the decorated object. Code defined in `exec()`,
`eval()`, the interactive prompt or a notebook cell executed from a string cannot be
processed. Put the function in a `.py` file.

## Why is a variable "not defined" in a data-sharing clause?

The error is `SyntaxError: name 'x' is not defined` in a clause such as `private(x)`.

Variables in data-sharing clauses must exist before the region. Initialise them first
(`x = 0`) even if the value is never used.

## Why is my parallel loop not faster?

In order of likelihood:

1. The interpreter has the GIL (`python -c "import sys; print(sys._is_gil_enabled())"`).
2. The loop body is tiny compared with the cost of starting threads: use `if(...)`, fewer
   regions, or larger work units.
3. A `critical` section, or a shared structure, serialises the threads.
4. The schedule does not balance the work: try `schedule(dynamic, chunk)`.
5. Pure mode: try hybrid or compiled mode.

See {doc}`/user_guide/performance`.

## Can I use a `break` or `return` inside a parallel region?

No: OpenMP forbids jumping out of a structured block. OMP4Py reports
`invalid branch to/from OpenMP structured block`. Use a shared flag and test it
inside the loop instead.

## Can I use directives inside classes?

Yes. `@omp` can decorate a class; its methods are processed. `yield` is not allowed inside
directive blocks.

## What is the difference between OMP4Py and PyOMP?

PyOMP uses Numba: loops are compiled to native code, but only Numba-supported Python
and libraries can be used. OMP4Py is plain Python (optionally Cython-compiled), so any Python
object or library can be used inside parallel regions, at the cost of lower speed for
numerical code unless compiled with types.

## Can I combine OMP4Py and MPI?

Yes: use `mpi4py` for communication between processes and OMP4Py inside each process
({doc}`/tutorials/numpy`, last section).

## How do I see what the preprocessor did?

`@omp(dump="out.py")` writes the transformed code ({doc}`/user_guide/debugging`).

## My extension changes are ignored

The cache key is the user's source file, not your extension. Use `ignore_cache=True`
({doc}`/contributor/extensions/index`).

## How do I extend OMP4Py?

With the extensions module: custom devices, subparsers and modifiers
({doc}`/contributor/extensions/index`).
