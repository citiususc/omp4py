"""Every syntax error of the directive grammar must have a specific message.

``parser_error`` returns a customised ``SyntaxError`` for the cases it knows and raises a generic
``parsing error`` for the rest. This module walks **all the states of the LALR automaton** and, in each
one, feeds three kinds of invalid input to check that the generic error never happens:

* ``end``: the end of the input (``$END``) where more input is required,
* ``terminal``: a terminal that the state does not accept,
* ``python``: ``PYTHON_BODY``, the terminal that matches any Python code.

If a new rule or clause of ``openmp.lark`` adds a state with no specific message, the failure lists the
directive prefix that leads to it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pytest

from omp4py.core.parser.error import parser_error
from omp4py.core.parser.openmp_lark import ImmutableInteractiveParser, LexerState, Token, UnexpectedInput
from omp4py.core.parser.parser import openmp_parser
from omp4py.core.parser.tree import Span

# Text used to build the SyntaxError (only its number of lines matters).
SOURCE = "\n" * 10
SPAN = Span(0, 0, 0, 0)

KINDS = ("end", "terminal", "python")


@dataclass
class Probe:
    """Result of feeding an invalid token in a state of the parser.

    Attributes:
        kind (str): Kind of invalid input: ``end``, ``terminal`` or ``python``.
        path (str): Tokens that lead to the state, from the start.
        token (str): The invalid token that was fed.
        message (str | None): Message of the specific error, or ``None`` if it was generic.
    """

    kind: str
    path: str
    token: str
    message: str | None

    @property
    def generic(self) -> bool:
        return self.message is None

    def __str__(self) -> str:
        return f"{self.path or '<start>'} + {self.token}"


def check_token(
    parser_state: ImmutableInteractiveParser, path: str, last_token: Token | None, kind: str, label: str
) -> Probe | None:
    """Feed an invalid token and classify the error that ``parser_error`` produces."""
    # The directive-name modifier marker (_HM) is an artificial token that the lexer predicts by looking
    # ahead, so it cannot be followed by an arbitrary token before its colon.
    if "_HM" in path and ("COLON" not in path or path.index("_HM") > path.index("COLON")):
        return None

    token = Token(label, label, 0, 0, 0, 0, 0, 0)
    try:
        parser_state.feed_token(token)
    except UnexpectedInput as ex:
        ex.interactive_parser = parser_state
        ex.state.lexer.state = LexerState("", None, last_token)  # what the real lexer would have set
        try:
            error = parser_error(openmp_parser, ex, SPAN, SOURCE, "")
        except SyntaxError:  # it raises (instead of returning) the generic error
            return Probe(kind, path, label, None)
        return Probe(kind, path, label, error.msg)
    return None  # the token was accepted: not an invalid input


@dataclass
class Exploration:
    """Everything found while visiting the parser.

    Attributes:
        states (int): Number of different parser states that were visited.
        probes (list[Probe]): Invalid inputs that were fed, with the error each one produced.
    """

    states: int
    probes: list[Probe]


def explore() -> Exploration:
    """Visit every state of the parser once (breadth first) and probe it with invalid tokens."""
    start = openmp_parser.parse_interactive()
    all_terminals = set(openmp_parser.lexer_conf.terminals_by_name)

    probes: list[Probe] = []
    visited: set[int] = set()
    queue = deque([("", start.as_immutable(), start.parser.parse_table.start_states["start"], None)])

    while queue:
        path, parser_state, state, last_token = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        if path.endswith("$END"):
            continue

        rejected = set(all_terminals)
        for label in sorted(parser_state.accepts()):
            rejected.discard(label)
            token = Token(label, label)
            next_state = parser_state.feed_token(token)
            queue.append((f"{path} {token}".strip(), next_state, next_state.parser_state.position, token))

        for kind, label in zip(KINDS, ("$END", min(rejected), "PYTHON_BODY"), strict=True):
            if (probe := check_token(parser_state, path, last_token, kind, label)) is not None:
                probes.append(probe)

    return Exploration(len(visited), probes)


@pytest.fixture(scope="module")
def exploration() -> Exploration:
    return explore()


def test_explores_the_whole_parser(exploration: Exploration) -> None:
    """Guard against an exploration that silently stops visiting states."""
    specific = [p for p in exploration.probes if not p.generic]
    assert exploration.states > 100
    assert len(specific) > exploration.states


@pytest.mark.parametrize("kind", KINDS)
def test_no_generic_error(exploration: Exploration, kind: str) -> None:
    generic = [p for p in exploration.probes if p.kind == kind and p.generic]
    assert not generic, f"{len(generic)} generic 'parsing error' with {kind!r} input:\n" + "\n".join(
        f"  {p}" for p in generic
    )
