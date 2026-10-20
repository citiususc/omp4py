from __future__ import annotations
from curses import raw
from omp4py.core.parser.source_view import SourceView

import ast as pyast
import dataclasses
import enum
import json
import traceback
import typing
import argparse
from pathlib import Path

from .openmp_parser import Tree

from . import tree
from .parser import _parse, preprocesor

if typing.TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any


def read_interactive() -> Generator[str]:
    """Reads a string from the user interactively simulating a prompt."""
    try:
        while True:
            try:
                user_input = input(">>> ")
                if user_input.startswith("#"):
                    continue
                elif user_input.endswith("\\"):
                    user_input = user_input[:-1] + "\n"
                    while True:
                        more_input = input("... ")
                        if more_input.startswith("#"):
                            continue
                        elif more_input.endswith("\\"):
                            user_input += more_input[:-1] + "\n"
                        else:
                            user_input += more_input
                            break
                yield user_input
            except KeyboardInterrupt:
                print()
    except EOFError:
        print()
        return


def ast_to_json(node: Any, sv: SourceView, expand_ast: bool = False, **kwargs) -> str:
    """Serialize the given AST as JSON for debugging."""
    # Utility function
    def span_from_pyast(obj: pyast.AST) -> str | None:
        if not hasattr(obj, "lineno"):
            return "???"
        lineno         = typing.cast("int", obj.lineno)
        end_lineno     = typing.cast("int", getattr(obj, "end_lineno",    lineno))
        col_offset     = typing.cast("int", getattr(obj, "col_offset",    0))
        end_col_offset = typing.cast("int", getattr(obj, "end_col_offset", col_offset))
        span = tree.Span(
            lineno=lineno,
            offset=col_offset,
            end_lineno=end_lineno,
            end_offset=end_col_offset,
        )
        return f'{span.lineno}:{span.offset}-{span.end_lineno}:{span.end_offset} ==> <{sv.source_text(span)}>'

    # Recursive function
    def node_to_json(obj: Any) -> Any:
        # Spans are shown with a compact representation and which parts of the code point to
        if isinstance(obj, tree.Span):
            return f'{obj.lineno}:{obj.offset}-{obj.end_lineno}:{obj.end_offset} ==> <{sv.source_text(obj)}>'

        # Python code
        if isinstance(obj, pyast.AST):
            # If not allowed, just print the original code
            if not expand_ast:
                return pyast.unparse(obj)

            # Else, serialize
            result: dict[str, Any] = {"type": type(obj).__name__}
            loc = span_from_pyast(obj)
            if loc:
                result["span"] = loc
            for field, value in pyast.iter_fields(obj):
                result[field] = node_to_json(value)
            return result

        # Enums for the clause kinds
        if isinstance(obj, enum.Enum):
            return obj.name

        # Serialize dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            result = {"type": type(obj).__name__}
            for f in dataclasses.fields(obj):
                if not f.init:
                    continue
                result[f.name] = node_to_json(getattr(obj, f.name))
            return result

        # Recursive calls for each element in the list/dict
        if isinstance(obj, list):
            return [node_to_json(v) for v in obj]
        if isinstance(obj, dict):
            return {k: node_to_json(v) for k, v in obj.items()}

        if isinstance(obj, Tree):
            return {
                obj.data: {
                    "position": node_to_json(sv.meta2span(obj.meta)),
                    "content": [node_to_json(c) for c in obj.children],
                },
            }

        # Fallback: see if json.dumps() can handle it
        return obj

    return json.dumps(node_to_json(node), **kwargs)


def run_interactive_parser() -> None:
    # To simulate a complete code
    prefix = """\
from omp4py import *

@omp
def pi(n: int) -> float:
    w = 1.0 / n
    pi_value = 0.0

    with omp("""
    suffix = """):
        for i in range(n):
            local = (i + 0.5) * w
            pi_value += 4.0 / (1.0 + local * local)

    return pi_value * w

print(pi(1_000_000))
"""

    for user_input in read_interactive():
        try:
            raw_source = (
                f'"{user_input}"'
                if len(user_input.splitlines()) == 1
                else f'"""{user_input}"""'
            )

            # Format complete code
            complete_code = prefix + raw_source + suffix
            lines = complete_code.splitlines()

            # Find the directive's position for the span
            start = complete_code.index(raw_source)
            before = complete_code[:start]
            before_lines = before.splitlines()
            start_line = len(before_lines)     # 1-based
            start_col  = len(before_lines[-1]) # 0-based

            after_start = start + len(raw_source)
            end_lines = complete_code[:after_start].splitlines()
            end_line = len(end_lines)
            end_col  = len(end_lines[-1])

            sv = SourceView(
                tree.Span(
                    lineno=start_line, offset=start_col,
                    end_lineno=end_line, end_offset=end_col,
                ),
                "<stdin>",
                lines,
                complete_code
            )

            code = preprocesor.parse(raw_source)

            print("==== COMPLETE_CODE ====")
            print(complete_code, end="")
            print("==== SPAN =============")
            print(*sv.annotate(sv.span), end="", sep="")
            print("==== AST ==============")
            ast = _parse(code, sv)
            print(ast_to_json(ast, sv, indent=2, expand_ast=True))

        except Exception as e:
            traceback.print_exception(e)


def run_interactive_preprocessor() -> None:
    for user_input in read_interactive():
        try:
            ast = preprocesor.parse(user_input)
            print(user_input)
            print(ast)

        except Exception as e:
            traceback.print_exception(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("omp", "pre"))
    args = parser.parse_args()

    if args.what == "omp":
        run_interactive_parser()
    elif args.what == "pre":
        run_interactive_preprocessor()


