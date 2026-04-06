"""`api` package: Python interface to the OpenMP API.

This package provides a Pythonic interface to OpenMP functionality,
exposing routines for device management, memory handling, synchronization,
tasking, thread teams, thread affinity, and timing.

The submodules included are:

- `deviceinf`: Functions to query and manage OpenMP devices.
- `devicemem`: Memory allocation and management routines for OpenMP devices.
- `lock`: Lock and synchronization primitives.
- `tasking`: Task creation and management routines.
- `teamsregion`: Management of OpenMP parallel regions.
- `threadaffinity`: Functions to query and control thread affinity.
- `threadteam`: Thread team management routines.
- `timing`: Timing and performance measurement utilities.

By importing this package, users can access the full OpenMP API
through Python, with consistent behavior across pure Python and
compiled runtimes.
"""

from omp4py.runtime.api.deviceinf import *  # noqa: F403
from omp4py.runtime.api.devicemem import *  # noqa: F403
from omp4py.runtime.api.lock import *  # noqa: F403
from omp4py.runtime.api.tasking import *  # noqa: F403
from omp4py.runtime.api.teamsregion import *  # noqa: F403
from omp4py.runtime.api.threadaffinity import *  # noqa: F403
from omp4py.runtime.api.threadteam import *  # noqa: F403  # noqa: F403
from omp4py.runtime.api.timing import *  # noqa: F403
