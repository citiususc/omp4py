"""Assign clauses to the individual constructs of a compound directive.

This module provides the logic for distributing clauses among the
individual constructs that form a compound OpenMP directive. It
determines which constructs support each clause and assigns clauses
according to their type and the corresponding assignment rules.

Clauses can be assigned to the innermost or outermost matching
construct, distributed to all matching constructs, or handled by
custom assignment methods. The `innermost` and `outermost` sets
define the default placement rules, while custom methods provide
specific assignment logic for individual clause types.

The `ClauseAssignment` class manages the available constructs and
their assignable clauses. It determines the eligible constructs for
each clause and applies the appropriate assignment strategy.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable

from omp4py.core.parser import tree

__all__ = ["ClauseAssignment"]

# Clauses assigned to the innermost matching construct.
innermost: set[type[tree.Clause]] = {tree.Private, tree.Linear}

# Clauses assigned to the outermost matching construct.
outermost: set[type[tree.Clause]] = {tree.NoGroup, tree.NoWait}


class ClauseAssignment:
    """Manage the assignment of clauses to compound directive constructs.

    This class determines which constructs support each clause and
    distributes clauses according to their type and assignment rules.
    It supports default placement strategies as well as custom
    assignment methods for specific clause types.

    Attributes:
        constructs (dict[str, tree.Construct]): Available constructs,
            keyed by their names.
        assignable_clauses (dict[str, list[tree.Clause]]): Clauses
            already assigned to each construct, keyed by construct name.
    """
    constructs: dict[str, tree.Construct]
    assignable_clauses: dict[str, list[tree.Clause]]

    def __init__(self, constructs: dict[str, tree.Construct], assignable_clauses: dict[str, list[tree.Clause]]):
        """Initialize the clause assignment manager.

        Args:
            constructs (dict[str, tree.Construct]): Available constructs
                that can receive clauses, keyed by their names.
            assignable_clauses (dict[str, list[tree.Clause]]): Clauses
                already assigned to each construct, keyed by construct
                name. These clauses are used when determining the
                assignment of subsequent clauses.
        """
        self.constructs = constructs
        self.assignable_clauses = assignable_clauses

    def get_context(self, clause: tree.Clause) -> list[str]:
        """Get the construct names associated with a clause's context.

        If the clause specifies a directive name, the context consists
        of the construct names included in that directive. Otherwise,
        all available constructs are considered part of the context.

        Args:
            clause (tree.Clause): Clause whose directive context is
                being determined.

        Returns:
            list[str]: Lowercase construct names defining the clause's
            context, in directive order or dictionary insertion order.
        """
        if clause.directive_name:
            return [name.string.lower() for name in clause.directive_name.directives]
        return list(self.constructs.keys())

    def get_allowed(self, clause: tree.Clause) -> list[str]:
        """Get the constructs that support a clause.

        A construct is eligible if it provides an attribute matching
        the clause identifier and, when the clause specifies a
        directive context, its name belongs to that context.

        Args:
            clause (tree.Clause): Clause whose eligible constructs
                are being determined.

        Returns:
            list[str]: Names of constructs that support the clause,
            preserving the insertion order of `constructs`.
        """
        cid = clause.id()
        mod = [name.string.lower() for name in clause.directive_name.directives] if clause.directive_name else []
        return [name for name, cons in self.constructs.items() if hasattr(cons, cid) and (not mod or name in mod)]

    def assign(self, clauses: list[tree.Clause]) -> dict[str, list[tree.Clause]]:
        """Assign clauses to the constructs that support them.

        Each clause is assigned according to its type and the
        corresponding assignment strategy:

        - Clauses with a custom assignment method are distributed
          according to that method's rules.
        - Clauses in `innermost` are assigned to the last matching
          construct.
        - Clauses in `outermost` are assigned to the first matching
          construct.
        - All other clauses are assigned to every matching construct.

        Clauses without eligible constructs are ignored.

        Args:
            clauses (list[tree.Clause]): Clauses to distribute among
                the available constructs.

        Returns:
            dict[str, list[tree.Clause]]: Clauses grouped by construct
            name. Each value contains the clauses assigned to that
            construct.
        """
        custom: Callable[[tree.Clause, list[str]], dict[str, list[tree.Clause]]] | None
        result: dict[str, list[tree.Clause]] = defaultdict(list)

        for clause in clauses:
            allowed = self.get_allowed(clause)
            custom = getattr(self, f"assign_{clause.cid()}", None)

            if not allowed:
                continue
            if custom is not None:
                for name, rs in custom(clause, allowed).items():
                    result[name].extend(rs)
            elif type(clause) in innermost:
                result[allowed[-1]].append(clause)
            elif type(clause) in outermost:
                result[allowed[0]].append(clause)
            else:
                for name in allowed:
                    result[name].append(clause)

        return result

    def assign_first_private(self, clause: tree.FirstPrivate, allowed: list[str]) -> dict[str, list[tree.Clause]]:
        """Select constructs that receive a `firstprivate` clause.

        The assignment strategy prioritizes `distribute` over `teams`.
        It also includes matching work-sharing constructs and assigns
        the clause to `taskloop` when the directive context contains
        `taskloop`. If no construct has been selected, `parallel` is
        used as a fallback. The `target` construct is included
        whenever it is present in the context.

        Args:
            clause (tree.FirstPrivate): Clause to assign.
            allowed (list[str]): Names of constructs supporting the
                clause.

        Returns:
            dict[str, list[tree.Clause]]: Mapping of selected construct
            names to lists containing the clause.
        """
        context = self.get_context(clause)
        asign = []
        if "distribute" in context:
            asign.append("distribute")
        elif "teams" in context:
            asign.append("teams")
        asign.extend(c for c in ("single", "sections", "for", "loop") if c in context)
        if "taskloop" in context:
            asign.append("taskloop")
        if not asign and "parallel" in context:
            asign.append("parallel")
        if "target" in context:
            asign.append("target")

        return {name: [clause] for name in asign}

    def assign_reduction(self, clause: tree.Reduction, allowed: list[str]) -> dict[str, list[tree.Clause]]:
        """Select constructs that receive a `reduction` clause.

        The clause is initially assigned to all eligible constructs.
        Two exceptions modify this behavior:

        - `parallel` is excluded when a work-sharing construct
          (`single`, `sections`, `for`, or `loop`) is present in the
          directive context.
        - `teams` is excluded when `loop` is present in the
          directive context.

        Args:
            clause (tree.Reduction): Clause to assign.
            allowed (list[str]): Names of constructs supporting the
                clause.

        Returns:
            dict[str, list[tree.Clause]]: Mapping of selected construct
            names to lists containing the clause.
        """
        context = self.get_context(clause)
        asign = allowed.copy()
        if any(c for c in ("single", "sections", "for", "loop") if c in context) and "parallel" in context:
            asign.remove("parallel")
        if "teams" in context and "loop" in context:
            asign.remove("teams")

        return {name: [clause] for name in asign}

    def assign_last_private(self, clause: tree.LastPrivate, allowed: list[str]) -> dict[str, list[tree.Clause]]:
        """Assign a `lastprivate` clause and resolve conflicting variables.

        The clause is initially assigned to every eligible construct.
        For `parallel` and `teams`, variables already covered by
        `firstprivate`, `private`, `reduction`, or `shared` clauses
        are excluded from the clause. If any targets remain, they are
        converted into a `shared` clause and assigned to the corresponding
        construct.

        Args:
            clause (tree.LastPrivate): Clause to assign.
            allowed (list[str]): Names of constructs supporting the
                clause.

        Returns:
            dict[str, list[tree.Clause]]: Mapping of construct names to
            their assigned clauses. Eligible constructs initially
            receive the original `lastprivate` clause. `parallel` and
            `teams` may instead receive a generated `shared` clause
            containing the remaining targets.
        """
        result: dict[str, list[tree.Clause]] = {name: [clause] for name in allowed}
        for name in ("parallel", "teams"):
            if name not in self.constructs:
                continue
            used: list[str] = []
            dup = copy.deepcopy(clause)
            for data in self.assignable_clauses[name]:
                if isinstance(data, (tree.FirstPrivate, tree.Private, tree.Reduction, tree.Shared)):
                    used.extend(data.str_targets)
            dup.targets = [var for var in dup.targets if var.string not in used]
            if dup.targets:
                shared = tree.Shared(dup.span, dup.name, dup.targets)
                result[name]= [shared]

        return result
