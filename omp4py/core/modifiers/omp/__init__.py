"""Built-in OpenMP AST modifiers.

This package contains the default modifiers provided by `omp4py`
for adapting, optimizing, or extending the generated AST during
the preprocessing pipeline.

The modifiers included here are automatically available to the
modifier engine and can be enabled or disabled through the
preprocessing configuration system.
"""
import omp4py.core.modifiers.omp.typing  #noqa: F401

__all__ = []
