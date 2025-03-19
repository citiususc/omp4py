import typing
from omp4py.runtime.common import controlvars, thread
from omp4py.runtime.common.enums import * #pyexport
from omp4py.runtime.basics.types import * #pyexport
from omp4py.runtime.basics.casting import *

__all__ = [  # Parallel Region Support Routines
    'omp_set_num_threads',
    'omp_get_num_threads',
    'omp_get_thread_num',
    'omp_get_max_threads',
    'omp_get_thread_limit',
    'omp_in_parallel',
    'omp_set_dynamic',
    'omp_get_dynamic',
    'omp_sched_t',
    'omp_sched_static',
    'omp_sched_dynamic',
    'omp_sched_guided',
    'omp_sched_auto',
    'omp_set_schedule',
    'omp_get_schedule_t',
    'omp_get_schedule',
    'omp_get_supported_active_levels',
    'omp_set_max_active_levels',
    'omp_get_max_active_levels',
    'omp_get_level',
    'omp_get_ancestor_thread_num',
    'omp_get_team_size',
    'omp_get_active_level',

    # Teams Region Routines
    # 'omp_get_num_teams',
    # 'omp_set_num_teams',
    # 'omp_get_team_num',
    # 'omp_get_max_teams',
    # 'omp_get_teams_thread_limit',
    # 'omp_set_teams_thread_limit',

    # Tasking Support Routines

    # Device Information Routines

    # Device Memory Routines

    # Interoperability Routines

    # Memory Management Routines

    # Lock Routines
    # 'omp_lock_t',
    # 'omp_nest_lock_t',
    # 'omp_sync_hint_t',
    # 'omp_sync_hint_none',
    # 'omp_sync_hint_uncontended',
    # 'omp_sync_hint_contended',
    # 'omp_sync_hint_nonspeculative',
    # 'omp_sync_hint_speculative',
    # 'omp_init_lock',
    # 'omp_init_lock_with_hint',
    # 'omp_init_nest_lock',
    # 'omp_init_nest_lock_with_hint',
    # 'omp_destroy_lock',
    # 'omp_destroy_nest_lock',
    # 'omp_set_lock',
    # 'omp_set_nest_lock',
    # 'omp_unset_lock',
    # 'omp_unset_nest_lock',
    # 'omp_test_lock',
    # 'omp_test_nest_lock',

    # Thread Affinity Routines

    # Execution Control Routines

    # Typing
    'pyint',
    'pyfloat'
]


#######################################################################################################################
########################################## Parallel Region Support Routines ###########################################
#######################################################################################################################

def omp_set_num_threads(num_threads: pyint):
    thread.cvars().dataenv.nthreads[0] = num_threads


def omp_get_num_threads() -> pyint:
    return thread.cvars().dataenv.team_size


def omp_get_thread_num() -> pyint:
    return thread.cvars().dataenv.thread_num


def omp_get_max_threads() -> pyint:
    return thread.cvars().dataenv.nthreads[0]


def omp_get_thread_limit() -> pyint:
    return thread.cvars().dataenv.thread_limit


def omp_in_parallel() -> bool:
    return thread.cvars().dataenv.active_levels > 0


def omp_set_dynamic(dynamic_threads: bool) -> None:
    thread.cvars().dataenv.dyn = dynamic_threads


def omp_get_dynamic() -> bool:
    return thread.cvars().dataenv.dyn


def omp_set_schedule(kind: omp_sched_t, chunk_size: pyint = -1) -> None:
    sched: controlvars.ScheduleVar = thread.cvars().dataenv.run_sched
    sched.monotonic = cast(pyint, kind & omp_sched_monotonic) != 0
    sched.kind = cast(pyint, (kind & ~omp_sched_monotonic))
    if chunk_size > 0:
        sched.chunk = chunk_size


class omp_get_schedule_t:
    kind: omp_sched_t
    chunk_size: pyint

    @staticmethod
    def _new(kind: omp_sched_t, chunk_size: pyint):
        obj: omp_get_schedule_t = omp_get_schedule_t()
        obj.kind = kind
        obj.chunk_size = chunk_size
        return obj

    def __iter__(self) -> typing.Iterator[tuple[omp_sched_t, pyint]]:
        return iter((self.kind, self.chunk_size))


def omp_get_schedule() -> omp_get_schedule_t:
    sched: controlvars.ScheduleVar = thread.cvars().dataenv.run_sched
    flag: omp_sched_t = omp_sched_t(sched.kind)
    if sched.monotonic:
        flag = omp_sched_monotonic | omp_sched_monotonic
    return omp_get_schedule_t._new(flag, sched.chunk)


def omp_get_supported_active_levels() -> pyint:
    return 2 ** 31


def omp_set_max_active_levels(max_levels: pyint) -> None:
    if max_levels < omp_get_supported_active_levels():
        thread.cvars().dataenv.max_active_levels = max_levels


def omp_get_max_active_levels() -> pyint:
    return thread.cvars().dataenv.max_active_levels


def omp_get_level() -> pyint:
    return thread.cvars().dataenv.levels


def omp_get_ancestor_thread_num(level: pyint) -> pyint:
    return 0  # TODO


def omp_get_team_size(level: pyint) -> pyint:
    return 1  # TODO


def omp_get_active_level() -> pyint:
    return thread.cvars().dataenv.active_levels


#######################################################################################################################
################################################ Teams Region Routines ################################################
#######################################################################################################################
def omp_get_num_teams() -> pyint:
    return thread.cvars().dataenv.league_size


def omp_set_num_teams(num_teams: pyint) -> None:
    thread.cvars().device.nteams = num_teams


def omp_get_team_num() -> pyint:
    return thread.cvars().dataenv.thread_num


def omp_get_max_teams() -> pyint:
    return thread.cvars().device.nteams


def omp_get_teams_thread_limit() -> pyint:
    return thread.cvars().device.teams_thread_limit


def omp_set_teams_thread_limit(thread_limit: pyint) -> None:
    thread.cvars().device.teams_thread_limit = thread_limit


"""
#######################################################################################################################
############################################## Tasking Support Routines ###############################################
#######################################################################################################################

# TODO

#######################################################################################################################
############################################# Device Information Routines #############################################
#######################################################################################################################

# TODO

#######################################################################################################################
############################################### Device Memory Routines ################################################
#######################################################################################################################

# TODO

#######################################################################################################################
############################################## Interoperability Routines ##############################################
#######################################################################################################################

# TODO

#######################################################################################################################
############################################## Memory Management Routines #############################################
#######################################################################################################################

# TODO

#######################################################################################################################
#################################################### Lock Routines ####################################################
#######################################################################################################################

class omp_lock_t:

    def __enter__(self):
        ...

    def __exit__(self, *args, **kwargs):
        ...


class omp_nest_lock_t:

    def __enter__(self):
        ...

    def __exit__(self, *args, **kwargs):
        ...


class omp_sync_hint_t: ...


omp_sync_hint_none: omp_sync_hint_t = ...
omp_sync_hint_uncontended: omp_sync_hint_t = ...
omp_sync_hint_contended: omp_sync_hint_t = ...
omp_sync_hint_nonspeculative: omp_sync_hint_t = ...
omp_sync_hint_speculative: omp_sync_hint_t = ...


def omp_init_lock() -> omp_lock_t:
    ...


def omp_init_lock_with_hint(hint: omp_sync_hint_t) -> omp_lock_t:
    ...


def omp_init_nest_lock() -> omp_nest_lock_t:
    ...


def omp_init_nest_lock_with_hint(hint: omp_sync_hint_t) -> omp_nest_lock_t:
    ...


def omp_destroy_lock(svar: omp_lock_t) -> None:
    ...


def omp_destroy_nest_lock(nvar: omp_nest_lock_t) -> None:
    ...


def omp_set_lock(svar: omp_lock_t) -> None:
    ...


def omp_set_nest_lock(nvar: omp_nest_lock_t) -> None:
    ...


def omp_unset_lock(svar: omp_lock_t) -> None:
    ...


def omp_unset_nest_lock(nvar: omp_nest_lock_t) -> None:
    ...


def omp_test_lock(svar: omp_lock_t) -> bool:
    ...


def omp_test_nest_lock(nvar: omp_nest_lock_t) -> bool:
    ...

#######################################################################################################################
############################################## Thread Affinity Routines ###############################################
#######################################################################################################################

# TODO

#######################################################################################################################
############################################## Execution Control Routines #############################################
#######################################################################################################################
"""
