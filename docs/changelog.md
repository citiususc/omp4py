# Changelog

## 1.0a1 (alpha, in development)

The version in this repository (`omp4py/__init__.py`). It is a redesign that will cover the
**OpenMP 4.5** standard. What it contains today:

- The `omp` entry point: directives, decorator, package mode, file preprocessing and decorators
  with defaults ({doc}`user_guide/decorator/index`).
- A directive parser based on Lark, with the OpenMP grammar and the OpenMP 6.0 combined directive
  names.
- Transformations for `parallel`, `for`, `sections`, `single`, `master`, `masked`, `critical`,
  `barrier`, `ordered` and `threadprivate`, with the data-sharing clauses and built-in reductions.
- The four execution modes: pure, hybrid, compiled and compiled with types.
- The thread-team, nesting and scheduling routines of the runtime API, and the OpenMP 4.5
  environment variables.
- **Extensibility**: an extension API (`omp4py.core.ext`) with subparsers, devices and modifiers,
  loaded lazily ({doc}`contributor/extensions/index`).
- This documentation, the examples and the benchmark programs.

See {doc}`compatibility/coverage` for the exact status of every feature.

## 0.2

The **Cython prototype**, described in *Unlocking Python Multithreading Capabilities using
OpenMP-Based Programming with OMP4Py* (CGO 2026). It introduced the **dual-runtime
architecture**: the pure Python runtime plus a native runtime written with Cython, and the three
(later four) execution modes pure, hybrid, compiled and compiled with explicit data types. It
was evaluated on numerical and non-numerical applications and in hybrid MPI/OpenMP programs.

## 0.1

The **first version**, a proof of concept of OpenMP in **pure Python** covering a subset of the
**OpenMP 3.0** specification, described in *OMP4Py: A pure Python implementation of OpenMP*
(Future Generation Computer Systems, 2026). It showed that the directive-based model of OpenMP can
be brought to Python with no dependencies other than the standard library.
