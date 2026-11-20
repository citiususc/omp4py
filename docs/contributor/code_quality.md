# Code Quality

Code that goes into `omp4py/` must pass **Ruff** (linting) and **Ty** (type checking). Install the
tools with the `checks` dependency group and run them through the `poe` tasks:

```bash
uv sync --group checks
uv run poe ruff      # ruff check omp4py
uv run poe ty        # ty check
```

## Ruff

Configured in `ruff.toml`: all the rule families are enabled (`select = ["ALL"]`, preview rules
on), with a line length of 120. The generated parsers (`*/parser/*_lark.py`), the Cython shim
(`cython.py`) and `docs/conf.py` are excluded.

Docstrings follow the **Google style** and are checked by the docstring rules, so document
arguments, return values and raised exceptions. They are also what the generated reference
({doc}`internals`) is built from.

## Ty

Configured in `ty.toml`. It checks the `omp4py` package, excluding the generated parsers and the
Cython shim. Keep annotations complete and use `typing.TYPE_CHECKING` imports for types that
are only needed by the checker, as the existing modules do.

## Before opening a pull request

1. `poe ruff` and `poe ty` must report nothing.
2. If you changed a `.lark` grammar, regenerate the parsers with `poe lark` and commit the
   generated files.
3. Update the documentation of what you changed ({doc}`/user_guide/index`, and
   {doc}`/compatibility/coverage` if support for a feature changed).
