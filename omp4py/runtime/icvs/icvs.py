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

import typing

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.atomic import AtomicInt

if typing.TYPE_CHECKING:
    from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["Data", "Device", "Global", "ImplicitTask", "PlacePartition", "RunSched", "defaults"]

defaults: Data

# END_CYTHON_IGNORE


class Device:
    """Device-related Internal Control Variables (ICVs).

    This class groups ICVs associated with device execution and limits,
    as defined in the OpenMP specification.

    Attributes:
        stacksize (pyint): Corresponds to the `stacksize-var` ICV. Defines
            the stack size for threads.
        wait_policy_active (bool): Derived from the `wait-policy-var` ICV.
            Indicates whether the wait policy is active (active vs passive).
        max_active_levels (pyint): Corresponds to the
            `max-active-levels-var` ICV. Specifies the maximum number of
            nested active parallel regions.

        threads_busy (AtomicInt): Number of OpenMP threads currently executing
            (implementation state).
    """

    stacksize: pyint
    wait_policy_active: bool
    max_active_levels: pyint
    ## Implementation States
    threads_busy: AtomicInt

    def copy(self) -> Device:
        """Create a shallow copy of the device ICVs.

        Returns:
            Device: A new instance with the same values.
        """
        obj: Device = Device.__new__(Device)
        obj.stacksize = self.stacksize
        obj.wait_policy_active = self.wait_policy_active
        obj.max_active_levels = self.max_active_levels
        ## Implementation States
        obj.threads_busy = AtomicInt.new(1)
        return obj


class Global:
    """Global Internal Control Variables (ICVs).

    This class contains ICVs that affect global runtime behavior.

    Attributes:
        cancel (bool): Corresponds to the `cancel-var` ICV. Enables or
            disables cancellation constructs.
        max_task_priority (pyint): Corresponds to the
            `max-task-priority-var` ICV. Defines the maximum priority value
            for tasks.
    """

    cancel: bool
    max_task_priority: pyint

    def copy(self) -> Global:
        """Create a shallow copy of the global ICVs.

        Returns:
            Global: A new instance with the same values.
        """
        obj: Global = Global.__new__(Global)
        obj.cancel = self.cancel
        obj.max_task_priority = self.max_task_priority
        return self


class PlacePartition:
    """Representation of place partitions for thread affinity.

    This structure models the partitioning of places as defined by the
    OpenMP `places` and `proc_bind` mechanisms.

    Attributes:
        values (pyint_array): List of place identifiers.
        partitions (pyint_array): Partition boundaries.
        name (str): Name of the place partition.
    """

    values: pyint_array
    partitions: pyint_array
    name: str


class ImplicitTask:
    """Implicit task-related Internal Control Variables (ICVs).

    This class stores ICVs associated with implicit tasks, including
    affinity and place partitioning.

    Attributes:
        place_partition (PlacePartition | None): Represents the current
            place-partition-var ICV, defining how threads are mapped to
            hardware resources.
    """

    place_partition: PlacePartition | None

    def copy(self) -> ImplicitTask:
        """Create a shallow copy of the implicit task ICVs.

        Returns:
            ImplicitTask: A new instance with the same values.
        """
        obj: ImplicitTask = ImplicitTask.__new__(ImplicitTask)
        obj.place_partition = self.place_partition
        return obj


class RunSched:
    """Run-schedule Internal Control Variable (ICV).

    This class represents the `run-sched-var` ICV, which controls loop
    scheduling behavior for worksharing constructs.

    Attributes:
        monotonic (bool): Indicates whether the schedule is monotonic
            (as per the OpenMP schedule modifier).
        type (pyint): Scheduling kind (e.g., static, dynamic, guided).
        chunksize (pyint): Chunk size used for loop scheduling.
    """

    monotonic: bool
    type: pyint
    chunksize: pyint

    def copy(self) -> RunSched:
        """Create a shallow copy of the run-sched ICV.

        Returns:
            RunSched: A new instance with the same values.
        """
        obj: RunSched = RunSched.__new__(RunSched)
        obj.monotonic = self.monotonic
        obj.type = self.type
        obj.chunksize = self.chunksize
        return obj


class Data:
    """Container for all OpenMP Internal Control Variables (ICVs).

    This class aggregates the full set of ICVs that define the execution
    context of a thread in the `omp4py` runtime.

    Attributes:
        dyn (bool): Corresponds to the `dyn-var` ICV. Enables dynamic
            adjustment of the number of threads.
        nest (bool): Corresponds to the `nest-var` ICV. Enables nested
            parallelism.
        nthreads (pyint_array): Corresponds to the `nthreads-var` ICV.
            Specifies the number of threads requested for parallel regions.
        run_sched (RunSched): Corresponds to the `run-sched-var` ICV.
        bind (pyint_array): Corresponds to the `bind-var` ICV. Defines
            thread affinity policies.
        bind_active (bool): Indicates whether binding is currently active.
        thread_limit (pyint): Corresponds to the `thread-limit-var` ICV.
            Sets the maximum number of threads.
        active_levels (pyint): Corresponds to the `active-levels-var` ICV.
            Number of active nested parallel regions.
        levels (pyint): Corresponds to the `levels-var` ICV. Total nesting
            depth of parallel regions.
        default_device (pyint): Corresponds to the
            `default-device-var` ICV. Specifies the default target device.

        team_size (pyint): Size of the current team (implementation state).
        thread_num (pyint): Thread identifier within the team
            (implementation state).

        device_vars (Device): Device-related ICV group.
        global_vars (Global): Global ICV group.
        implicit_task_vars (ImplicitTask): Implicit task ICV group.
    """

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
    ## Implementation States
    team_size: pyint
    thread_num: pyint
    ## ICV groups
    device_vars: Device
    global_vars: Global
    implicit_task_vars: ImplicitTask

    def copy(self) -> Data:
        """Create a shallow copy of the full ICV set.

        Returns:
            Data: A new instance with the same values.
        """
        obj: Data = Data.__new__(Data)
        obj.dyn = self.dyn
        obj.nest = self.nest
        obj.nthreads = new_pyint_array(len(self.nthreads))
        obj.nthreads[:] = self.nthreads[:]
        obj.run_sched = self.run_sched
        obj.bind = self.bind
        obj.thread_limit = self.thread_limit
        obj.active_levels = self.active_levels
        obj.levels = self.levels
        obj.default_device = self.default_device
        ## Implementation States
        obj.team_size = self.team_size
        obj.thread_num = self.thread_num
        ## ICV groups
        obj.device_vars = self.device_vars
        obj.global_vars = self.global_vars
        obj.implicit_task_vars = self.implicit_task_vars
        return obj


# Initializations

# Default initialization of the Internal Control Variables (ICVs).
# Values are set according to the OpenMP specification defaults and may
# be overridden using environment variables
defaults = Data.__new__(Data)
