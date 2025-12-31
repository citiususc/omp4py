# BEGIN_CYTHON_IMPORTS: Add 'cython.imports' prefix to omp4py packages
import omp4py.runtime.common.tasks as tasks
import omp4py.runtime.basics.threadlocal as threadlocal
import omp4py.runtime.common.controlvars as controlvars
import omp4py.runtime.common.threadshared as threadshared
# END_CYTHON_IMPORTS


from cython import cast

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
        self.task = self.task.parent
        self.parallel = self.task.parallel
        self.teams = self.task.teams

    def set_parallel(self, task: tasks.ParallelTask) -> 'OmpThread':
        self.parallel = task
        return self

    def set_teams(self, task: tasks.TeamsTask) -> 'OmpThread':
        self.teams = task
        return self


def init(task: tasks.Task) -> OmpThread:
    local: OmpThread = OmpThread.__new__(OmpThread)
    local.task = task
    local.parallel = task.parallel
    local.teams = task.teams
    threadlocal.set_storage(local)
    return local


def current() -> OmpThread:
    if not threadlocal.has_storage():
        cvars: controlvars.ControlVars = controlvars.ControlVars.new()
        cvars.default()
        task: tasks.ParallelTask = tasks.ParallelTask.new(cvars, threadshared.SharedFactory.new(cvars))
        task.parallel = task
        task.teams = None
        task.parent = None

        init(task)
    return cast(OmpThread, threadlocal.get_storage())


def cvars() -> controlvars.ControlVars:
    return current().task.cvars
