"""AST modifier infrastructure for `omp4py`.

This package defines the modifier system used during the preprocessing
pipeline to adapt, optimize, or extend the generated AST.

Modifiers are transformation passes that can run before or after the
main OpenMP transformation stage. They are integrated into the
preprocessing process and can be enabled or disabled through the
configuration arguments passed to the `omp` decorator or preprocessing
entry points.

The package also exposes the base `Modifier` class and the
`modifier` registration decorator used to define custom modifiers.
"""
from omp4py.core.modifiers.engine import Modifier, modifier

__all__ = ["Modifier", "modifier"]
