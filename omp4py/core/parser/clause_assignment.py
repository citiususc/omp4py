"""Assign clauses to the individual constructs of a compound directive.

This module provides the logic for distributing clauses among the
constructs that form a compound directive. It determines which
constructs support each clause and assigns it according to its type
and the corresponding assignment rules.

Clauses can be assigned to the innermost or outermost matching
construct, distributed to all matching constructs, or handled by
custom assignment functions. The ``innermost`` and ``outermost``
sets define the default placement rules, while the ``custom``
dictionary allows specific clause types to provide their own
assignment logic.

Custom rules are registered by associating a clause class with
a function that receives the names of the eligible constructs and
returns the names selected to receive the clause.

The main entry point is ``assign_clauses``, which applies these
rules and returns the clauses grouped by construct name.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import partial

from omp4py.core.parser import tree

__all__ = ["assign_clauses"]


# Clauses assigned to the innermost matching construct.
innermost: set[type[tree.Clause]] = {tree.Private, tree.Linear}

# Clauses assigned to the outermost matching construct.
outermost: set[type[tree.Clause]] = {tree.NoGroup, tree.NoWait}

# Custom assignment rules for specific clause types.
custom: dict[type[tree.Clause], Callable[[list[str]], list[str]]] = {}


def assign_clauses(
    constructs: dict[str, tree.Construct],
    clauses: list[tree.Clause],
) -> dict[str, list[tree.Clause]]:
    """Assign clauses to the constructs that support them.

    Each clause is assigned according to its type: innermost clauses
    go to the last matching construct, outermost clauses to the first,
    and custom clauses follow their registered assignment rule.
    All other clauses are assigned to every matching construct.

    Args:
        constructs (dict[str, tree.Construct]): Constructs available
            for clause assignment, keyed by name.
        clauses (list[tree.Clause]): Clauses to distribute.

    Returns:
        dict[str, list[tree.Clause]]: Clauses grouped by construct name.
    """
    result: dict[str, list[tree.Clause]] = defaultdict(list)

    for clause in clauses:
        cid = clause.id()
        allowed = [name for name, cons in constructs.items() if hasattr(cons, cid)]

        if not allowed:
            continue

        if type(clause) in innermost:
            result[allowed[-1]].append(clause)
        elif type(clause) in outermost:
            result[allowed[0]].append(clause)
        elif type(clause) in custom:
            for c in custom[type(clause)](allowed):
                result[c].append(clause)
        else:
            for c in allowed:
                result[c].append(clause)

    return result


@partial(custom.__setitem__, tree.FirstPrivate)
def firstprivate(constructs: list[str]) -> list[str]:
    """Select constructs that receive a firstprivate clause.

    Prioritizes ``distribute`` over ``teams``, includes matching work-
    sharing constructs, and assigns the clause to ``target`` when
    present. Falls back to ``parallel`` when no other construct
    has been selected.

    Args:
        constructs (list[str]): Names of constructs supporting the
            clause.

    Returns:
        list[str]: Names of constructs selected for assignment.
    """
    asign = []
    if "distribute" in constructs:
        asign.append("distribute")
    elif "teams" in constructs:
        asign.append("teams")
    asign.extend(c for c in ("single", "sections", "for", "loop") if c in constructs)
    if "tasklog" in constructs:
        asign.append("taskloop")
    if not asign and "parallel" in constructs:
        asign.append("parallel")
    if "target" in constructs:
        asign.append("target")

    return asign


@partial(custom.__setitem__, tree.FirstPrivate)
def reduction(constructs: list[str]) -> list[str]:
    """Select constructs that receive a reduction clause.

    Assigns the clause to all matching constructs, except that
    ``parallel`` is excluded when a work-sharing construct is present,
    and ``teams`` is excluded when ``loop`` is present.

    Args:
        constructs (list[str]): Names of constructs supporting the
            clause.

    Returns:
        list[str]: Names of constructs selected for assignment.
    """
    asign = constructs.copy()
    if any(c for c in ("single", "sections", "for", "loop") if c in constructs) and "parallel" in constructs:
        asign.remove("parallel")
    if "teams" in constructs and "loop" in constructs:
        asign.remove("teams")
    return asign
