import os
import sys
import time
import types
import pickle
import threading
from threading import Thread
from typing import Any, Callable
from multiprocessing import get_context

from tblib import Traceback, pickling_support

__all__ = ['proctest', 'proctest10']

mp = get_context("spawn")


def _end__coverage() -> None:
    pass


def _coverage() -> None:
    global _end__coverage
    if 'COV_CORE_SOURCE' in os.environ:
        import pytest_cov.embed as cov
        if cov._active_cov is None:
            cov.init()
        _end__coverage = cov.cleanup


def _thread_trace(frame: types.FrameType, next: types.TracebackType | None = None) -> types.TracebackType | None:
    if frame.f_code.co_name == '_run_subproc_child':
        return next

    return _thread_trace(frame.f_back, types.TracebackType(tb_next=next,
                                                           tb_frame=frame,
                                                           tb_lasti=frame.f_lasti,
                                                           tb_lineno=frame.f_lineno))


def _timeout(n: float, q: mp.Queue) -> None:
    time.sleep(n)
    try:
        thread_frame: types.FrameType = sys._current_frames()[threading.main_thread().ident]
        raise TimeoutError().with_traceback(_thread_trace(thread_frame, None))

    except BaseException as ex:
        pickling_support.install()
        q.put((False, ex))
        q.close()
        q.join_thread()
        _end__coverage()
        os._exit(1)


def _run_subproc_child(bf: bytes, timeout: float | None, q: mp.Queue, args, kwargs) -> None:
    _coverage()
    f: Callable = pickle.loads(bf)
    if timeout is not None:
        Thread(target=_timeout, args=(timeout, q), daemon=True).start()
    try:
        result: Any = f(*args, **kwargs)
        q.put((True, result))
    except BaseException as ex:
        if ex.__class__.__module__ == 'builtins' and not hasattr(sys.modules['builtins'], ex.__class__.__name__):
            if ex.__class__.__base__ is not None and \
                    ex.__class__.__base__.__module__ != 'builtins' and \
                    hasattr(sys.modules[ex.__class__.__base__.__module__], ex.__class__.__name__):
                ex.__class__.__module__ = ex.__class__.__base__.__module__
            else:
                ex = RuntimeError(f'builtin <{ex.__class__.__name__}> exception: {ex}').with_traceback(ex.__traceback__)

        pickling_support.install()
        q.put((False, ex))


def _run_subproc_parent(bf: bytes, timeout: float | None, args, kwargs) -> (bool, Any):
    q: mp.Queue = mp.Queue()

    p: mp.Process = mp.Process(target=_run_subproc_child, args=(bf, timeout, q, args, kwargs))
    p.start()
    p.join()
    if q.empty():
        return False, None
    return q.get()


def proctest(f0: Callable | None = None, /, *, timeout: float | None = None) -> Any:
    def wrapper(f: Callable):
        bf: bytes = pickle.dumps(f)

        def test(*args, **kwargs):
            check, result = _run_subproc_parent(bf, timeout, args, kwargs)
            if check:
                return result
            if not result:
                raise ChildProcessError("process terminated unexpectedly")

            bt: types.TracebackType = Traceback(result.__traceback__).tb_next.as_traceback()
            try:
                raise result
            except BaseException as ex:
                ex.__traceback__ = bt
                raise

        return test

    if f0 is None:
        return wrapper
    return wrapper(f0)


proctest10 = proctest(timeout=10.0)
