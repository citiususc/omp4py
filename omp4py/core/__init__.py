"""Core code generation and transformation system for `omp4py`.

This package contains the components responsible for code analysis,
transformation, and generation of OpenMP-compatible Python code. It
implements the preprocessing pipeline that converts annotated Python
source code into a form that can be executed by the `omp4py` runtime.

It is responsible for analyzing the abstract syntax tree (AST),
interpreting OpenMP directives, and rewriting the code into an
equivalent form suitable for execution by the runtime.
"""

from omp4py.core.api import omp
from omp4py.core.options import Options

if Options.pure:
    import omp4py.core.imports.pure as _  # noqa: F401

__all__ = ["omp"]
