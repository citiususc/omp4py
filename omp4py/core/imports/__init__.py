"""Import hook support modules for `omp4py`.

This package contains modules that extend or modify Python's import
system to support `omp4py` behavior.

These components are used to intercept imports, apply OpenMP
preprocessing to registered packages during module loading, and control
which runtime implementations are loaded, such as forcing the use of
pure Python sources instead of compiled extensions.

It provides the integration layer between Python's import machinery and
the `omp4py` preprocessing and runtime systems.
"""
