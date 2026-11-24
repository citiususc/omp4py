"""`sections`: every `section` block is executed once, by one of the threads of the team, and the blocks can run at
the same time. The construct ends with a barrier unless it has `nowait`."""

import threading

import pytest

from test.helpers import omp_function

DIRECTIVES = ["parallel sections", "sections"]


def sections(directive: str, count: int) -> str:
    """Source of a function with `count` sections that append their number to `out`."""
    blocks = "".join(f'with omp("section"):\n    out.append({i})\n' for i in range(count))
    if directive.startswith("parallel"):
        indented = "".join("        " + line + "\n" for line in blocks.splitlines())
        return f'def f(out):\n    with omp("parallel sections num_threads(3)"):\n{indented}'
    indented = "".join("            " + line + "\n" for line in blocks.splitlines())
    return f'def f(out):\n    with omp("parallel num_threads(3)"):\n        with omp("sections"):\n{indented}'


@pytest.mark.parametrize("count", [1, 2, 3, 7])
@pytest.mark.parametrize("directive", DIRECTIVES)
def test_every_section_is_executed_once(directive, count):
    out = []
    omp_function(sections(directive, count))(out)
    assert sorted(out) == list(range(count))


def test_the_sections_can_run_at_the_same_time():
    """Each of the two sections waits for the other one: they only finish if two threads execute them."""
    barrier = threading.Barrier(2, timeout=10)
    f = omp_function(
        """
        def f(barrier):
            with omp("parallel sections num_threads(2)"):
                with omp("section"):
                    barrier.wait()
                with omp("section"):
                    barrier.wait()
        """,
    )
    f(barrier)


def test_the_construct_ends_with_a_barrier():
    f = omp_function(
        """
        from time import sleep

        def f(out):
            done = []
            with omp("parallel num_threads(3)"):
                with omp("sections"):
                    with omp("section"):
                        for _ in range(3):
                            sleep(0.05)
                        done.append(True)
                out.append(bool(done))
        """,
    )
    out = []
    f(out)
    assert out == [True] * 3


def test_nowait_lets_the_other_threads_go_on():
    """The thread that executes the section waits for another one to have gone past the construct."""
    passed = threading.Barrier(2, timeout=10)
    f = omp_function(
        """
        def f(passed, out):
            with omp("parallel num_threads(2)"):
                with omp("sections nowait"):
                    with omp("section"):
                        out.append(omp_get_thread_num())
                        passed.wait()
                if len(out) == 1 and omp_get_thread_num() != out[0]:
                    passed.wait()
        """,
    )
    out = []
    f(passed, out)
    assert len(out) == 1


def test_lastprivate_takes_the_value_of_the_last_section():
    f = omp_function(
        """
        from time import sleep

        def f():
            x = 1
            with omp("parallel sections lastprivate(x) num_threads(2)"):
                with omp("section"):
                    sleep(0.1)
                    x = 3
                with omp("section"):
                    x = 2
            return x
        """,
    )
    assert f() == 2


def test_reduction_of_the_sections():
    f = omp_function(
        """
        def f():
            x = 1
            with omp("parallel sections reduction(+:x) num_threads(2)"):
                with omp("section"):
                    x += 3
                with omp("section"):
                    x += 2
            return x
        """,
    )
    assert f() == 6


def test_private_and_firstprivate_variables():
    f = omp_function(
        """
        def f(out):
            x = 0
            y = 5
            with omp("parallel sections private(x) firstprivate(y) num_threads(2)"):
                with omp("section"):
                    x = 1
                    y += 1
                    out.append(y)
                with omp("section"):
                    x = 2
                    y += 2
                    out.append(y)
            return x, y
        """,
    )
    out = []
    assert f(out) == (0, 5)
    assert sorted(out) in ([6, 7], [6, 8], [7, 7])


# ---- invalid code ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        'with omp("section"):\n            pass\n        x = 1',
        'with omp("section"):\n            pass\n        with open("file"):\n            pass',
        'with omp("section"):\n            with omp("section"):\n                pass',
    ],
    ids=["other statement", "other block", "nested section"],
)
def test_only_section_blocks_can_be_inside_sections(body):
    with pytest.raises(SyntaxError):
        omp_function(
            f"""
            def f():
                with omp("parallel sections"):
                    {body}
            """,
        )


def test_a_section_must_be_inside_sections():
    with pytest.raises(SyntaxError):
        omp_function(
            """
            def f():
                with omp("parallel"):
                    with omp("section"):
                        pass
            """,
        )
