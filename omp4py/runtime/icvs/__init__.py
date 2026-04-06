"""`icvs` package: Internal Control Variables for the `omp4py` runtime.

This package provides access to the internal control variables (ICVs)
used by the `omp4py` runtime to manage OpenMP execution behavior.
ICVs control parallel execution, thread management, scheduling,
device configuration, and task execution.

It defines the core ICV data structures and handles their initialization
from default values and environment variables, including support for
`OMP_DISPLAY_ENV` for debugging.

By importing this package, users and runtime components can access
and modify OpenMP ICVs in a Pythonic way, ensuring consistent behavior
across pure Python and compiled runtimes.
"""
import omp4py.runtime.icvs.defaults as _
