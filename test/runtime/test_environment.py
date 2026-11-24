"""The environment variables of the standard: they set the initial value of the ICVs when the program starts.

Every case starts a new interpreter, because the variables are read when omp4py is imported.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import omp4py
from omp4py import omp_sched_dynamic, omp_sched_guided, omp_sched_monotonic, omp_sched_static

ROOT = Path(omp4py.__file__).resolve().parent.parent
PRELUDE = "import os, sys\nfrom omp4py import *\n"


def run(code: str, **variables: str) -> subprocess.CompletedProcess:
    """Run `code` in a new interpreter where the OMP_* variables are exactly the given ones."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith("OMP_")}
    environment.update(variables)
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), environment.get("PYTHONPATH", "")])
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", PRELUDE + code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
        cwd=ROOT,
    )


def value(expression: str, **variables: str) -> str:
    result = run(f"print({expression})", **variables)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


# ---- numbers -----------------------------------------------------------------------------------------------------


def test_omp_num_threads_is_the_initial_number_of_threads():
    assert value("omp_get_max_threads()", OMP_NUM_THREADS="3") == "3"


def test_a_region_without_num_threads_uses_omp_num_threads():
    code = """
@omp
def f():
    sizes = []
    with omp("parallel"):
        sizes.append(omp_get_num_threads())
    return sizes
print(f())
"""
    result = run(code, OMP_NUM_THREADS="4")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[4, 4, 4, 4]"


def test_a_list_gives_the_number_of_threads_of_each_nesting_level():
    code = """
@omp
def f():
    sizes = []
    with omp("parallel"):
        with omp("master"):
            with omp("parallel"):
                with omp("master"):
                    sizes.append((omp_get_team_size(1), omp_get_team_size(2)))
    return sizes
print(f())
"""
    result = run(code, OMP_NUM_THREADS="3,2", OMP_MAX_ACTIVE_LEVELS="2")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[(3, 2)]"


def test_omp_max_active_levels():
    assert value("omp_get_max_active_levels()", OMP_MAX_ACTIVE_LEVELS="3") == "3"


def test_omp_nested_is_the_old_name_of_a_nesting_limit_greater_than_one():
    assert value("omp_get_max_active_levels() > 1", OMP_NESTED="true") == "True"
    assert value("omp_get_max_active_levels()", OMP_NESTED="false") == "1"


def test_omp_thread_limit():
    assert value("omp_get_thread_limit()", OMP_THREAD_LIMIT="2") == "2"


@pytest.mark.parametrize(("variable", "expected"), [("true", "True"), ("false", "False")])
def test_omp_dynamic(variable, expected):
    assert value("omp_get_dynamic()", OMP_DYNAMIC=variable) == expected


@pytest.mark.parametrize(("variable", "expected"), [("true", "True"), ("false", "False")])
def test_omp_cancellation(variable, expected):
    assert value("omp_get_cancellation()", OMP_CANCELLATION=variable) == expected


# ---- schedule ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "kind", "chunk"),
    [
        ("dynamic,4", omp_sched_dynamic, 4),
        ("guided,2", omp_sched_guided, 2),
        ("static,3", omp_sched_static, 3),
        ("dynamic", omp_sched_dynamic, -1),
        ("monotonic:dynamic,2", omp_sched_dynamic | omp_sched_monotonic, 2),
        ("nonmonotonic:guided", omp_sched_guided, -1),
    ],
)
def test_omp_schedule(variable, kind, chunk):
    assert value("omp_get_schedule()", OMP_SCHEDULE=variable) == str((kind, chunk))


def test_a_runtime_schedule_loop_uses_omp_schedule():
    code = """
@omp
def f():
    owners = [0] * 12
    with omp("parallel for schedule(runtime) num_threads(3)"):
        for i in range(12):
            owners[i] = omp_get_thread_num()
    return owners
print(f())
"""
    result = run(code, OMP_SCHEDULE="static,2")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == str([(k // 2) % 3 for k in range(12)])


# ---- values that are not valid and display -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value_"),
    [("OMP_NUM_THREADS", "many"), ("OMP_NUM_THREADS", "-2"), ("OMP_SCHEDULE", "sometimes"), ("OMP_DYNAMIC", "maybe")],
)
def test_a_value_that_is_not_valid_does_not_stop_the_program(variable, value_):
    result = run("print(omp_get_max_threads() >= 1, omp_get_dynamic() in (True, False))", **{variable: value_})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "True True"


def test_omp_display_env_prints_the_values_when_the_program_starts():
    result = run("pass", OMP_DISPLAY_ENV="true", OMP_NUM_THREADS="3", OMP_SCHEDULE="dynamic,4")
    assert result.returncode == 0, result.stderr
    text = result.stdout + result.stderr
    assert "OPENMP DISPLAY ENVIRONMENT BEGIN" in text
    assert "OPENMP DISPLAY ENVIRONMENT END" in text
    assert "OMP_NUM_THREADS = '3'" in text
    assert "OMP_SCHEDULE = 'DYNAMIC,4'" in text.upper().replace("'DYNAMIC,4'", "'DYNAMIC,4'")


def test_omp_display_env_verbose_prints_more_variables():
    plain = run("pass", OMP_DISPLAY_ENV="true")
    verbose = run("pass", OMP_DISPLAY_ENV="verbose")
    assert len(verbose.stdout + verbose.stderr) > len(plain.stdout + plain.stderr)
    assert "OMP_WAIT_POLICY" in verbose.stdout + verbose.stderr
