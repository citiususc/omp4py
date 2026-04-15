"""AST transformation package for the `omp4py` preprocessor.

This package groups all transformation modules responsible for rewriting
the Python AST to implement OpenMP-like constructs and clauses.

Each submodule registers its transformations through the `construct`
dispatcher, covering areas such as:

- Parallel regions
- Worksharing constructs
- Synchronization primitives
- Data environment management
- Reduction operations

Importing this package ensures that all supported constructs are
registered and available to the `OmpTransformer`.
"""

from omp4py.core.preprocessor.transformers.transformer import OmpTransformer

__all__ = ["OmpTransformer"]

# All supported constructs must be imported here
import omp4py.core.preprocessor.transformers.dataenv
import omp4py.core.preprocessor.transformers.mastersync
import omp4py.core.preprocessor.transformers.parallelism
import omp4py.core.preprocessor.transformers.reduction
import omp4py.core.preprocessor.transformers.worksharing  #noqa: F401
