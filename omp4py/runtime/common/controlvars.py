import os
import re
import typing
import omp4py.runtime.basics.array as array
from  omp4py.runtime.common.enums import omp_sched_names
from omp4py.runtime.basics.types import *

__all__ = ['ScheduleVar', 'GlobalVars', 'DataEnvVars', 'DeviceVars', 'ITaskVars', 'ControlVars']

T = typing.TypeVar('T')


def getenv(name: str, defval: T, parser: typing.Callable[[str], T]) -> T:
    if name in os.environ:
        try:
            return parser(os.getenv(name))
        except:
            return defval
    return defval


def n_cores() -> pyint:
    try:
        return len(os.sched_getaffinity(0))
    except:
        return os.cpu_count()


def p_nthreads(value: str) -> pyint:
    if "n_cores" in value:
        value = value.replace("n_cores", str(n_cores()))
    return cast(pyint, value)


def p_list(p: typing.Callable[[str], T]) -> typing.Callable[[str], list[T]]:
    value: str
    e: str
    return lambda value: [p(e) for e in value.split(',')]


def p_match(f: typing.Callable[[str], bool]) -> typing.Callable[[str], T]:
    def wrap(value: str) -> T:
        if not f(value):
            raise ValueError(value)
        return value

    return wrap


class ScheduleVar:
    monotonic: bool
    kind: pyint
    chunk: pyint

    def set(self, monotonic: bool, kind: pyint, chunk: pyint) -> 'ScheduleVar':
        self.monotonic = monotonic
        self.kind = kind
        self.chunk = chunk
        return self

    @staticmethod
    def kind_int(kind: str) -> pyint:
        return omp_sched_names.get(kind)

    def __copy__(self):
        return ScheduleVar().set(self.monotonic, self.kind, self.chunk)


def p_sched(value: str) -> ScheduleVar:
    pattern: str = r'((monotonic|nonmonotonic):)?(static|dynamic|guided|auto)(\s*,\s*(\d+))?'

    result: re.Match[str] | None = re.search(pattern, value.lower())

    if result:
        kind: str = result.group(2)
        modifier: str = result.group(1) if result.group(1) else ('monotonic' if kind == 'static' else 'nonmonotonic')
        chunk: pyint = cast(pyint, result.group(3)) if result.group(3) else -1
        return ScheduleVar().set(kind == 'monotonic', ScheduleVar.kind_int(modifier), chunk)

    raise ValueError(value)


def p_size(value: str) -> pyint:
    pattern: str = r'\s*(\d+)\s*([BKMG])?\s*'

    ws: list[str] = ['B', 'K', 'M', 'G']

    result: re.Match[str] | None = re.search(pattern, value.upper())

    if result:
        n: pyint = result.group(1)
        w: str = result.group(2).upper() if result.group(2) else 'K'

        return n * 1024 ** ws.index(w)

    raise ValueError(value)


class GlobalVars:
    available_devices: str
    cancel: bool
    debug: bool
    display_affinity: bool
    max_task_priority: pyint
    num_devices: pyint
    target_offload: str

    def default(self):
        self.available_devices = getenv('OMP_AVAILABLE_DEVICES', '', str)  # TODO
        self.cancel = getenv('OMP_CANCELLATION', False, bool)
        self.debug = getenv('OMP_DEBUG', False, bool)
        self.display_affinity = getenv('OMP_DISPLAY_AFFINITY', False, bool)
        self.max_task_priority = getenv('OMP_MAX_TASK_PRIORITY', 0, int)
        self.num_devices = 0  # TODO
        self.target_offload = getenv('OMP_TARGET_OFFLOAD', 'default',
                                     p_match(lambda v: v in ['mandatory', 'disabled', 'default']))

    def __copy__(self):
        other: GlobalVars = GlobalVars()
        other.available_devices = self.available_devices
        other.cancel = self.cancel
        other.debug = self.debug
        other.display_affinity = self.display_affinity
        other.max_task_priority = self.max_task_priority
        other.num_devices = self.num_devices
        other.target_offload = self.target_offload
        return other


class DataEnvVars:
    active_levels: pyint
    bind: str
    default_device: str
    dyn: bool
    explicit_task: bool
    final_task: bool
    free_agent_thread_limit: pyint
    free_agent: bool
    league_size: pyint
    levels: pyint
    max_active_levels: pyint
    nthreads: array.iview
    run_sched: ScheduleVar
    structured_thread_limit: pyint
    team_generator: pyint
    team_num: pyint
    team_size: pyint
    thread_limit: pyint
    thread_num: pyint

    def default(self):
        self.active_levels = 0
        self.bind = getenv('OMP_PROC_BIND', '', str)  # TODO
        self.default_device = getenv('OMP_DEFAULT_DEVICE', '', str)
        self.dyn = getenv('OMP_DYNAMIC', False, bool)
        self.explicit_task = False
        self.final_task = False
        self.free_agent_thread_limit = 0  # TODO OMP_THREAD_LIMIT, OMP_THREADS_RESERVE
        self.free_agent = False
        self.league_size = 1
        self.levels = 0
        self.max_active_levels = getenv('OMP_MAX_ACTIVE_LEVELS', 2 ** 31, int)
        self.nthreads = array.int_from(getenv('OMP_NUM_THREADS', [n_cores()], p_list(p_nthreads)))
        self.run_sched = getenv('OMP_SCHEDULE', ScheduleVar().set(True, 0, -1), p_sched)
        self.structured_thread_limit = 1  # TODO OMP_THREAD_LIMIT, OMP_THREADS_RESERVE
        self.team_generator = 0
        self.team_num = 0
        self.team_size = 1
        self.thread_limit = getenv('OMP_THREAD_LIMIT', 2 ** 31, int)
        self.thread_num = 0

    def __copy__(self):
        other: DataEnvVars = DataEnvVars()
        other.active_levels = self.active_levels
        other.bind = self.bind
        other.default_device = self.default_device
        other.dyn = self.dyn
        other.explicit_task = self.explicit_task
        other.final_task = self.final_task
        other.free_agent_thread_limit = self.free_agent_thread_limit
        other.free_agent = self.free_agent
        other.league_size = self.league_size
        other.levels = self.levels
        other.max_active_levels = self.max_active_levels
        other.nthreads = self.nthreads[:]
        other.run_sched = self.run_sched.__copy__()
        other.structured_thread_limit = self.structured_thread_limit
        other.team_generator = self.team_generator
        other.team_num = self.team_num
        other.team_size = self.team_size
        other.thread_limit = self.thread_limit
        other.thread_num = self.thread_num

        return other


class DeviceVars:
    affinity_format: str
    device_num: pyint
    nteams: pyint
    num_procs: pyint
    stacksize: pyint
    teams_thread_limit: pyint
    wait_policy: str

    def default(self):
        self.affinity_format = getenv('OMP_AFFINITY_FORMAT', '', str)  # TODO
        self.device_num = 0
        self.nteams = getenv('OMP_NUM_TEAMS', 0, int)
        self.num_procs = 0  # TODO
        self.stacksize = getenv('OMP_STACKSIZE', -1, p_size)
        self.teams_thread_limit = getenv('OMP_TEAMS_THREAD_LIMIT', 0, int)
        self.wait_policy = getenv('OMP_WAIT_POLICY', 'active',
                                  p_match(lambda v: v.lower() in ['active', 'passive'])).lower()

    def __copy__(self):
        other = DeviceVars()
        other.affinity_format = self.affinity_format
        other.device_num = self.device_num
        other.nteams = self.nteams
        other.num_procs = self.num_procs
        other.stacksize = self.stacksize
        other.teams_thread_limit = self.teams_thread_limit
        other.wait_policy = self.wait_policy

        return other


class ITaskVars:
    def_allocator: str
    place_assignment: str

    def default(self):
        self.def_allocator = getenv('OMP_ALLOCATOR', '', str)  # TODO
        self.place_assignment = ''  # TODO

    def __copy__(self):
        other = DeviceVars()
        other.def_allocator = self.def_allocator
        other.place_assignment = self.place_assignment

        return other


class ControlVars:
    global_: GlobalVars
    dataenv: DataEnvVars
    device: DeviceVars
    itask: ITaskVars

    def default(self):
        self.global_ = GlobalVars()
        self.global_.default()
        self.dataenv = DataEnvVars()
        self.dataenv.default()
        self.device = DeviceVars()
        self.device.default()
        self.itask = ITaskVars()
        self.itask.default()

    def __copy__(self):
        other = ControlVars()
        other.global_ = self.global_
        other.dataenv = self.dataenv
        other.device = self.device
        other.itask = self.itask

        return other
