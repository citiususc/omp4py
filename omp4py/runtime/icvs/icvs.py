"""Internal control variables (ICVs) used by the `omp4py` runtime.

This module defines the data structures that represent the internal
control variables (ICVs) described in the OpenMP specification.
ICVs control the behavior of parallel execution, including thread
management, scheduling, device configuration, and task execution.

The classes defined here group related ICVs following the structure of
the OpenMP runtime model:

- `Device`: Device-related variables.
- `Global`: Global runtime settings.
- `ImplicitTask`: Variables associated with implicit tasks.
- `Data`: Main container holding all ICVs for a given execution context.

Each class provides a `copy()` method used to propagate ICV values when
creating new parallel regions or tasks, following the inheritance rules
defined by OpenMP.

The `defaults` object represents the initial state of the ICVs used by
the runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# BEGIN_CYTHON_IMPORTS: Add 'cython.imports' prefix to omp4py packages
if TYPE_CHECKING:
    from omp4py.runtime.lowlevel.numeric import pyint, pyint_array

# END_CYTHON_IMPORTS

__all__ = ["Data", "Device", "Global", "ImplicitTask", "defaults"]


class Device:
    stacksize: pyint
    wait_policy_active: bool
    max_active_levels: pyint

    def copy(self) -> Device:
        obj: Device = Device.__new__(Device)
        obj.stacksize = self.stacksize
        obj.wait_policy_active = self.wait_policy_active
        obj.max_active_levels = self.max_active_levels
        return obj


class Global:
    cancel: bool
    max_task_priority: pyint

    def copy(self) -> Global:
        obj: Global = Global.__new__(Global)
        obj.cancel = self.cancel
        obj.max_task_priority = self.max_task_priority
        return self


class PlacePartition:
    values: pyint_array
    partitions: pyint_array
    name: str


class ImplicitTask:
    place_partition: PlacePartition | None

    def copy(self) -> ImplicitTask:
        obj: ImplicitTask = ImplicitTask.__new__(ImplicitTask)
        obj.place_partition = self.place_partition
        return obj


class RunSched:
    monotonic: bool
    type: pyint
    chunksize: pyint

    def copy(self) -> RunSched:
        obj: RunSched = RunSched.__new__(RunSched)
        obj.monotonic = self.monotonic
        obj.type = self.type
        obj.chunksize = self.chunksize
        return obj


class Data:
    dyn: bool
    nest: bool
    nthreads: pyint_array
    run_sched: RunSched
    bind: pyint_array
    bind_active: bool
    thread_limit: pyint
    active_levels: pyint
    levels: pyint
    default_device: pyint
    ##
    device_vars: Device
    global_vars: Global
    implicit_task_vars: ImplicitTask

    def copy(self) -> Data:
        obj: Data = Data.__new__(Data)
        obj.dyn = self.dyn
        obj.nest = self.nest
        obj.nthreads = self.nthreads
        obj.run_sched = self.run_sched
        obj.bind = self.bind
        obj.thread_limit = self.thread_limit
        obj.active_levels = self.active_levels
        obj.levels = self.levels
        obj.default_device = self.default_device
        obj.device_vars = self.device_vars
        obj.global_vars = self.global_vars
        obj.implicit_task_vars = self.implicit_task_vars
        return obj


defaults: Data = Data.__new__(Data)
