"""`lowlvel` package: Low-level implementations for the `omp4py` runtime.

This package provides the low-level building blocks for the `omp4py`
runtime. It defines the core behavior of OpenMP runtime and ensures
consistent execution across both pure Python and compiled environments.

Each module may have two implementations: a pure Python version that
serves as the baseline, and an optional compiled version (`.pyx`) that
may include embedded C code for maximum performance. Simple modules
can sometimes leverage the Python implementation with minor adjustments
via the `.pxd` interface files.

From the perspective of other packages and users, all modules expose
the same functionality regardless of the underlying implementation.
"""
