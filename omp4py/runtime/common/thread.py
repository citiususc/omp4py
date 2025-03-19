import omp4py.runtime.common.tasks as tasks
import omp4py.runtime.basics.threadlocal as threadlocal
import omp4py.runtime.common.controlvars as controlvars
import omp4py.runtime.common.threadshared as threadshared
from omp4py.runtime.basics.casting import *

__all__ = ['OmpThread', 'init', 'current', 'cvars']


class OmpThread:
    task: tasks.Task
    parallel: tasks.ParallelTask | None
    teams: tasks.TeamsTask | None

    def set_task(self, task: tasks.Task) -> 'OmpThread':
        task.parallel = self.parallel
        task.teams = self.teams
        task.parent = self.task
        self.task = task

        return self

    def pop_task(self) -> None:
        self.parallel = self.task.parallel
        self.teams = self.task.teams
        self.task = self.task.parent

    def set_parallel(self, task: tasks.ParallelTask) -> None:
        self.parallel = task

    def set_teams(self, task: tasks.TeamsTask) -> None:
        self.teams = task


def init(task: tasks.Task) -> OmpThread:
    local: OmpThread = OmpThread.__new__(OmpThread)
    local.task = task
    local.parallel = None
    local.teams = None
    threadlocal.set_storage(local)
    return local


def current() -> OmpThread:
    if not threadlocal.has_storage():
        cvars: controlvars.ControlVars = controlvars.ControlVars()
        cvars.default()
        task: tasks.ParallelTask = tasks.ParallelTask.new(cvars, threadshared.SharedFactory(cvars))

        init(task).set_parallel(task)
    return cast(OmpThread, threadlocal.get_storage())


def cvars() -> controlvars.ControlVars:
    return current().task.cvars
