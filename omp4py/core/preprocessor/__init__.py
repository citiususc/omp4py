"""Preprocessor package for OpenMP-like AST transformation system.

This package provides the main entry points for transforming Python code
that contains OpenMP-style directives into executable Python code.

It acts as the high-level interface of the compilation pipeline and
coordinates the full preprocessing workflow, including:

- Parsing Python source code into an AST
- Extracting ASTs from runtime Python objects (functions/classes)
- Applying the OpenMP transformation pipeline via the transformer system
- Generating transformed Python code for execution or output

The package exposes three main interfaces:

- process_file: preprocess a Python file and optionally emit transformed output
- process_source: preprocess a raw source code string
- process_object: preprocess a live Python object (function or class)

Internally, all these entry points converge into the same transformation
pipeline implemented by the `OmpTransformer`, which performs the
low-level AST rewriting of OpenMP constructs, scopes, and data clauses.
"""

from omp4py.core.preprocessor.preprocessor import process_file, process_object, process_source

__all__ = ["process_file", "process_object", "process_source"]
