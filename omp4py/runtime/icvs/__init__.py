"""Initialize OpenMP ICV defaults and expose icvs data structures."""

import omp4py.runtime.icvs.defaults as _
from omp4py.runtime.icvs.icvs import Data, Device, Global, ImplicitTask

__all__ = ["Data", "Device", "Global", "ImplicitTask"]
