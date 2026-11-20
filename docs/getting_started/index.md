# Getting Started

OMP4Py lets you parallelise Python code with OpenMP directives. This section takes you from
zero to a running parallel program, in three short steps.

````{grid} 1 1 3 3
:gutter: 3

```{grid-item-card} 1 · Install
:link: installation
:link-type: doc

Requirements and installation. Use a free-threaded Python for real speedups.
```

```{grid-item-card} 2 · Quick start
:link: quickstart
:link-type: doc

A guided tour of the six things you will use every day, with runnable code.
```

```{grid-item-card} 3 · Core concepts
:link: concepts
:link-type: doc

How directives, the decorator, threads, data sharing and the execution modes fit together.
```
````

## OMP4Py in 60 seconds

| You want to... | You write... |
| --- | --- |
| Mark code for processing | `@omp` on a function or class (or `omp(pkg=...)` for a package) |
| Run a block with several threads | `with omp("parallel"):` |
| Split a loop among threads | `with omp("parallel for"):` + a `for i in range(...)` loop |
| Combine partial results | `reduction(+:total)` |
| Protect a shared update | `with omp("critical"):` |
| Wait for all threads | `omp("barrier")` |
| Ask about the team | `omp_get_thread_num()`, `omp_get_num_threads()` |
| Choose how it runs | `@omp(compile=True)`, `OMP4PY_PURE=1`, ... |

```{tip}
Everything on this page is expanded in the {doc}`/user_guide/index`. Start with
{doc}`/user_guide/decorator/index`: the `omp` decorator drives all the rest.
```

## Where next?

- Learn the decorator in depth: {doc}`/user_guide/decorator/index`
- All directives with examples: {doc}`/user_guide/directives`
- Make it fast: {doc}`/user_guide/performance` and {doc}`/user_guide/compiled_mode`
- Check what is implemented: {doc}`/compatibility/coverage`
