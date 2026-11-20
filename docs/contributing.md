# Contributing

Contributions (bug reports, documentation fixes, new constructs, benchmarks, extensions) are
welcome.

- **Report problems** at <https://github.com/citiususc/omp4py/issues>: include the directive,
  the Python build (`python -VV`, GIL or free-threaded), the mode (pure/hybrid/compiled)
  and the full error.
- **Send changes** as pull requests against <https://github.com/citiususc/omp4py>.

## Before you open a pull request

1. Read the {doc}`contributor guide <contributor/index>`, especially
   {doc}`contributor/architecture` and {doc}`contributor/adding_directive`.
2. Add or update tests ({doc}`contributor/code_quality`) and run them in both pure and compiled
   runtimes when the change touches `omp4py/runtime`.
3. Run `poe ruff` and `poe ty`.
4. If you changed a grammar file, regenerate the parsers with `poe lark` and commit the
   generated files.
5. Update the documentation: new or changed behaviour belongs in the user guide, and
   {doc}`compatibility/coverage` must stay accurate.

## Documentation conventions

- The documentation is written in Markdown (MyST) under `docs/`, built with Sphinx and the
  Shibuya theme, and diagrams use Mermaid (```` ```{mermaid} ````).
- Code examples that appear in several places live in `docs/examples/` and are included
  with `literalinclude`, so they are run and kept in sync. Run an example before quoting its
  output.
- API pages use `autodoc`: write docstrings in Google style, and they appear in the
  reference automatically.
- Be explicit about what is implemented. Do not describe a construct as supported until a test
  or example runs it.

## Building the documentation

```bash
uv sync --group dev --group docs
uv run poe docs                      # sphinx-build -b html docs html -v
```

To fail on any warning (as in a clean build):

```bash
uv run sphinx-build -W --keep-going -b html docs html
```
