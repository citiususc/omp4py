"""Initialization of OpenMP internal control variables.

This module initializes the internal control variables (ICVs) of the
`omp4py` runtime using environment variables defined by the OpenMP
specification. Each supported `OMP_*` variable is parsed and used
to configure the corresponding fields in `icvs.defaults`.

If an environment variable is not defined or contains an invalid value,
a default value is assigned following the OpenMP specification.

The module also supports the `OMP_DISPLAY_ENV` variable, which enables
printing the resolved configuration to standard error for debugging
purposes.

Initialization is performed at import time by calling `set_defaults()`,
which sets base values and then applies environment-based overrides.
"""

from __future__ import annotations

import os
import re
import sys

import omp4py.runtime.icvs.places as icvs_places

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.icvs import icvs
from omp4py.runtime.lowlevel.atomic import AtomicInt
from omp4py.runtime.lowlevel.numeric import new_pyint_array

# END_CYTHON_IMPORTS

__all__ = []


def parse(name: str, regex: str, sensitive: bool = False) -> tuple[str, ...]:
    if name not in os.environ:
        return ()
    if result := re.match(regex, os.environ[name].strip(), 0 if sensitive else re.IGNORECASE):
        return result.groups("")
    print(f"omp4py: Unknown value for environment variable {name}", file=sys.stderr)  # noqa: T201
    return ()


def display(txt: str) -> None:
    print(txt, file=sys.stderr)  # noqa: T201


def display_var(name: str, value: str) -> None:
    print(f"  {name} = '{value}'".upper(), file=sys.stderr)  # noqa: T201


def set_run_sched(verbose: bool) -> None:
    icvs.defaults.run_sched = icvs.RunSched.__new__(icvs.RunSched)
    if result := parse(
        "OMP_SCHEDULE",
        r"^(?:(monotonic|nonmonotonic):)?(static|dynamic|guided|auto)(?:\s*,\s*([1-9][0-9]*))?$",
    ):
        icvs.defaults.run_sched.monotonic = result[0][0] == "m" if result[1][0] == "s" and result[0] else False
        icvs.defaults.run_sched.type = ord(result[1][0])
        icvs.defaults.run_sched.chunksize = int(result[2]) if result[2] is not None else -1
    else:
        icvs.defaults.run_sched.monotonic = True
        icvs.defaults.run_sched.type = ord("s")
        icvs.defaults.run_sched.chunksize = -1

    if verbose:
        chunk: str = f",{icvs.defaults.run_sched.chunksize}" if icvs.defaults.run_sched.chunksize > 0 else ""
        value: str = ""
        match chr(icvs.defaults.run_sched.type):
            case "s":
                value = f"nonmonotonic:static{chunk}" if not icvs.defaults.run_sched.monotonic else f"static{chunk}"
            case "d":
                value = f"dynamic{chunk}"
            case "g":
                value = f"guided{chunk}"
            case "a":
                value = f"auto{chunk}"

        display_var("OMP_SCHEDULE", value)


def set_nthreads(verbose: bool) -> None:
    if result := parse("OMP_NUM_THREADS", r"^([1-9][0-9]*(?:\s*,\s*[1-9][0-9]*)*)$"):
        values: list[int] = [int(value) for value in result[0].split(",")]
        icvs.defaults.nthreads = new_pyint_array(len(values))
        for i in range(len(values)):
            icvs.defaults.nthreads[i] = values[i]
    else:
        icvs.defaults.nthreads = new_pyint_array(1)
        icvs.defaults.nthreads[0] = os.process_cpu_count() or 1

    if verbose:
        display_var("OMP_NUM_THREADS", ",".join(map(str, icvs.defaults.nthreads)))


def set_dyn(verbose: bool) -> None:
    if result := parse("OMP_DYNAMIC", r"^(true|false)$"):
        icvs.defaults.dyn = result[0] == "true"
    else:
        icvs.defaults.dyn = False

    if verbose:
        display_var("OMP_DYNAMIC", str(icvs.defaults.dyn))


def set_bind(verbose: bool) -> None:
    if result := parse("OMP_PROC_BIND", r"^(true|false|(?:master|close|spread)(?:\s*,\s*(?:master|close|spread))*)$"):
        if result[0] in ("true", "false"):
            icvs.defaults.bind_active = result[0] == "true"
            icvs.defaults.bind = new_pyint_array(0)
        else:
            icvs.defaults.bind_active = True
            values: list[int] = [ord(value[0]) for value in result[0].split(",")]
            icvs.defaults.bind = new_pyint_array(len(values))
            for i in range(len(values)):
                icvs.defaults.bind[i] = values[i]
    else:
        icvs.defaults.bind_active = False
        icvs.defaults.bind = new_pyint_array(0)

    if verbose:
        if len(icvs.defaults.bind) == 0:
            display_var("OMP_PROC_BIND", str(icvs.defaults.bind_active))
        else:
            display_var("OMP_PROC_BIND", ",".join(map(str, icvs.defaults.bind)))


def set_place_partition(verbose: bool) -> None:
    if result := icvs_places.parse("OMP_PLACES"):
        icvs.defaults.implicit_task_vars.place_partition = result

        if verbose:
            display_var("OMP_PLACES", os.environ["OMP_PLACES"])
    else:
        icvs.defaults.implicit_task_vars.place_partition = None


def set_nest(verbose: bool) -> None:
    if result := parse("OMP_NESTED", r"^(true|false)$"):
        icvs.defaults.nest = result[0] == "true"
    else:
        icvs.defaults.nest = False

    if verbose:
        display_var("OMP_NESTED", str(icvs.defaults.nest))


def set_stacksize(verbose: bool) -> None:
    units: dict[str, int] = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    if result := parse("OMP_STACKSIZE", r"^([1-9][0-9]*)\s*([BKMG])?$", sensitive=True):
        icvs.defaults.device_vars.stacksize = int(result[0]) * units.get(result[1], 1)

        if verbose:
            value: int = icvs.defaults.device_vars.stacksize
            for unit, w in reversed(units.items()):
                if value % w == 0:
                    display_var("OMP_STACKSIZE", str(value // w) + unit)
                    break
    else:
        icvs.defaults.device_vars.stacksize = -1


def set_wait_policy(verbose: bool) -> None:
    if result := parse("OMP_WAIT_POLICY", r"^(active|passive)$"):
        icvs.defaults.device_vars.wait_policy_active = result[0] == "active"
    else:
        icvs.defaults.device_vars.wait_policy_active = False

    if verbose:
        display_var("OMP_WAIT_POLICY", "active" if result else "passive")


def set_max_active_levels(verbose: bool) -> None:
    if result := parse("OMP_MAX_ACTIVE_LEVELS", r"^([1-9][0-9]*)$"):
        icvs.defaults.device_vars.max_active_levels = int(result[0])
    else:
        icvs.defaults.device_vars.max_active_levels = 2**31

    if verbose:
        display_var("OMP_MAX_ACTIVE_LEVELS", str(icvs.defaults.device_vars.max_active_levels))


def set_thread_limit(verbose: bool) -> None:
    if result := parse("OMP_THREAD_LIMIT", r"^([1-9][0-9]*)$"):
        icvs.defaults.thread_limit = int(result[0])
    else:
        icvs.defaults.thread_limit = 2**31

    if verbose:
        display_var("OMP_THREAD_LIMIT", str(icvs.defaults.thread_limit))


def set_cancellation(verbose: bool) -> None:
    if result := parse("OMP_CANCELLATION", r"^(true|false)$"):
        icvs.defaults.global_vars.cancel = result[0] == "true"
    else:
        icvs.defaults.global_vars.cancel = False

    if verbose:
        display_var("OMP_CANCELLATION", str(icvs.defaults.nest))


def set_default_device(verbose: bool) -> None:
    if result := parse("OMP_DEFAULT_DEVICE", r"^(\d+)$"):
        icvs.defaults.default_device = int(result[0])
    else:
        icvs.defaults.default_device = 0

    if verbose:
        display_var("OMP_DEFAULT_DEVICE", str(icvs.defaults.default_device))


def set_max_task_priority(verbose: bool) -> None:
    if result := parse("OMP_MAX_TASK_PRIORITY", r"^(\d+)$"):
        icvs.defaults.global_vars.max_task_priority = int(result[0])
    else:
        icvs.defaults.global_vars.max_task_priority = 0

    if verbose:
        display_var("OMP_MAX_TASK_PRIORITY", str(icvs.defaults.default_device))


def set_env() -> None:
    verbose: int = 0
    if result := parse("OMP_DISPLAY_ENV", r"^(true|false|verbose)$"):
        match result[0]:
            case "true":
                verbose = 1
            case "verbose":
                verbose = 2

    if verbose:
        display("OPENMP DISPLAY ENVIRONMENT BEGIN")
        display("_OPENMP = '201511'")

    set_run_sched(verbose > 0)
    set_nthreads(verbose > 0)
    set_dyn(verbose > 0)
    set_bind(verbose > 1)
    set_place_partition(verbose > 1)
    set_nest(verbose > 0)
    set_stacksize(verbose > 1)
    set_wait_policy(verbose > 1)
    set_max_active_levels(verbose > 0)
    set_thread_limit(verbose > 0)
    set_cancellation(verbose > 1)
    set_default_device(verbose > 1)

    if verbose:
        display("OPENMP DISPLAY ENVIRONMENT END")


def set_defaults() -> None:
    icvs.defaults.active_levels = 0
    icvs.defaults.levels = 0
    icvs.defaults.team_size = 1
    icvs.defaults.thread_num = 0
    icvs.defaults.device_vars = icvs.Device.__new__(icvs.Device)
    icvs.defaults.device_vars.threads_busy = AtomicInt.new(1)
    icvs.defaults.global_vars = icvs.Global.__new__(icvs.Global)
    icvs.defaults.implicit_task_vars = icvs.ImplicitTask.__new__(icvs.ImplicitTask)

    set_env()


set_defaults()
