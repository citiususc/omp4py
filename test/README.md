# Tests

The tests check that what omp4py implements works and follows the OpenMP standard. They do not depend on the
runtime that is used: every test must pass with the pure Python runtime (`pytest --pure`) and with the compiled one
(`pytest`).

| Folder | What it checks |
| --- | --- |
| `worksharing/` | The worksharing constructs: the loop (iterations, `schedule`, `collapse`, `ordered` and the clauses of `for`), `sections` and `single` |
| `parallel/` | The `parallel` region, data-sharing clauses and `default`, `threadprivate` and `copyin`, `reduction` and `declare reduction` |
| `sync/` | `critical`, `barrier`, and `master` and `masked` |
| `runtime/` | The library routines (ICVs, nesting levels, schedule) and the environment variables |
| `combined/` | Combined directives (`parallel for`, `parallel sections`, `masked parallel`...) and the ones that do not exist |
| `parser/` | The strings of the directives (every way of writing them, and where the errors point), the Python code of the clauses, the grammar and the error messages |

Every file starts with a docstring that says which part of the standard it covers. Features that are not
implemented (tasks, `atomic`, `target`...) have no tests yet: add them with the implementation.

## How a test is written
`helpers.omp_function` creates a function from text, so a directive can be written once and run with many
parameters:

```python
@pytest.mark.parametrize("schedule", ["schedule(static, 2)", "schedule(dynamic)"])
def test_every_iteration_runs_once(schedule):
    f = omp_function("""
        def f(out):
            with omp("parallel for $schedule num_threads(3)"):
                for i in range(20):
                    out.append(i)
    """, schedule=schedule)
    out = []
    f(out)
    assert sorted(out) == list(range(20))
```

* A test checks a property that the standard guarantees (each iteration once, the order of `ordered`, the value of
  `lastprivate`...), not what one particular run did. Do not wait with `sleep`: use `threading` barriers and events
  with a timeout.
* Invalid code is tested by calling `omp_function` inside `pytest.raises(SyntaxError)`.
* Each file starts with a docstring that says which part of the standard it covers.

## Running
`pytest` runs the tests in an isolated worker (see `conftest.py`): a test that hangs fails and the others go on.
`pytest --no-isolate` runs them in the process (to use a debugger).
