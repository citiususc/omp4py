"""Parser package for OpenMP-like directives in `omp4py`.

This package contains the parsing infrastructure responsible for
processing OpenMP-style directives embedded in Python source code.

It provides:

- Directive parsing utilities and syntax error helpers.
- The parser tree node definitions used to represent directives,
  constructs, clauses, modifiers, and expressions.

The parser converts raw directive source code into structured tree
representations later consumed by the transformation pipeline.
"""

from omp4py.core.parser.parser import *  # noqa: F403
from omp4py.core.parser.tree import *  # noqa: F403
