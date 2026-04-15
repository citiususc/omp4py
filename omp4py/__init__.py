"""Public API package for `omp4py`.

This package defines the main entry point of the `omp4py` project and
re-exports the public interface used to write OpenMP-style code in Python.

The `omp` function is the central mechanism that connects Python code with
the `omp4py` preprocessing and runtime system. It does not directly perform
parallel execution. Instead, it acts as a structured annotation layer that
marks regions of code with OpenMP directives and provides an interface for
the preprocessor to transform Python code into a parallel-capable form.

It also exposes the complete runtime API defined in `omp4py.runtime.api`,
which contains the functions and types defined in the OpenMP standard.
"""

__version__ = "1.0a1"

from omp4py.core import omp  # noqa: F401
from omp4py.runtime.api import *  # noqa: F403
