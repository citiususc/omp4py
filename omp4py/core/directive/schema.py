import dataclasses
from typing import Any, Callable

from omp4py.core.directive.tokenizer import NAME, TokenInfo
from omp4py.core.directive.names import *

__all__ = ['Arguments', 'Modifier', 'Clause', 'Directive', 'Group', 'MODIFIERS', 'CLAUSES', 'DIRECTIVES']


@dataclasses.dataclass(frozen=True, slots=True)
class Group:
    """
    Represents a group names of `Clauses` or `Modifiers`.

    Attributes:
        elems (tuple[str, ...]): Elements (names) that belong to the group.
        name (str | None): The name of the group. Can be None if no name is provided.
        required (bool): Indicates whether the group is required.
        exclusive (bool): Indicates whether the group is exclusive.
    """
    elems: tuple[str, ...]
    name: str | None = None
    required: bool = False
    exclusive: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class Arguments:
    """
    Represents the arguments for a `Directive`, `Clause` or `Modifier`.

    Attributes:
        post_modified (bool): Flag indicating if the modifiers are after the arguments.
        multiple (bool): Multiple sets of arguments are accepted with ';' as a separator.
        modifiers (tuple[str, ...]): Strings representing the modifiers for the argument.
        modifiers_groups (tuple[Group, ...]): `Group` objects that define groups of modifiers.
        num_args (int): The number of arguments expected. Defaults to -1, indicating any number of arguments.
        choices (tuple[str, ...] | None): A tuple of valid choices for the argument.
        transform (str): The transformation applied to the argument.
        require_args (bool): Flag indicating if arguments are required.
        parser (str): The parser type for the argument.
        custom (dict[str, Any]): A dictionary for custom attributes related to the argument.
    """
    post_modified: bool = False
    multiple: bool = False
    modifiers: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    modifiers_groups: tuple[Group, ...] = dataclasses.field(default_factory=tuple)
    num_args: int = -1
    choices: tuple[str, ...] | None = None
    transform: str = T_ITEM_ID
    require_args: bool = True
    parser: str = A_PARSER_BASIC
    custom: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """
        This method consolidates all the elements from each `Group` in the `modifiers_groups` list,
        removes duplicates, and appends the unique elements to the existing `modifiers` tuple.
        """
        others: tuple[str, ...] = tuple(dict.fromkeys(sum([group.elems for group in self.modifiers_groups], tuple())))
        object.__setattr__(self, 'modifiers', self.modifiers + others)


@dataclasses.dataclass(frozen=True, slots=True)
class Modifier:
    """
    Represents a modifier that can be used in arguments.

    Attributes:
        match (Callable[[str, int], bool] | None): A function that checks if a value matches the modifier based on its
                                                    token string and type.
        token_match (Callable[[list[TokenInfo]], bool] | None): A function that checks if a value matches the modifier
                                                    based on its tokens.
        values (tuple[str, ...] | None): A list of valid values for the modifier. Like "math = lambda n,t: n in values".
        msg_value: (str | None): Value to show on required error instead of stringify values or modifier name.
        required (bool): The modifier is required.
        repeatable (bool): The modifier can be repeated.
        ultimate (bool): The modifier must be the last one a sequence.
        transform (str): The transformation applied to the modifier.
        args (Arguments | None): An optional `Arguments` object representing the arguments associated with the modifier.
    """
    match: Callable[[str, int], bool] | None = None
    token_match: Callable[[list[TokenInfo]], bool] | None = None
    values: tuple[str, ...] | None = None
    msg_value: str | None = None
    required: bool = False
    repeatable: bool = False
    ultimate: bool = False
    transform: str = T_ITEM_ID
    args: Arguments | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class Clause:
    """
    Represents a Clause within a Directive.

    Attributes:
        required (bool): The clause is required. Defaults to False.
        repeatable (bool): The clause can appear more than once in a directive.
        ultimate (bool): The clause must be the last one in a directive.
        args (Arguments | None): An optional `Arguments` object representing the arguments associated with the clause.

    """
    required: bool = False
    repeatable: bool = False
    ultimate: bool = False
    args: Arguments | None = None

    def __post_init__(self):
        """
        This ensures that the clause will always have the `M_DIRECTIVE_NAME` modifier at the beginning of its
        modifiers list.
        """
        if self.args is not None:
            object.__setattr__(self.args, 'modifiers', (M_DIRECTIVE_NAME,) + self.args.modifiers)


@dataclasses.dataclass(frozen=True, slots=True)
class Directive:
    """
    Represents a Directive.

    Attributes:
        prefix (bool): Directive is composed and requires another directive (e.g., "declare reduction").
        clauses (tuple[str, ...]): The clauses that the directive can accept.
        clauses_groups (tuple[Group, ...]): `Group` objects that define groups of clauses.
        args (Arguments | None): An optional `Arguments` object representing the arguments associated with the directive.
    """
    prefix: bool = False
    clauses: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    clauses_groups: tuple[Group, ...] = dataclasses.field(default_factory=tuple)
    args: Arguments | None = None

    def __post_init__(self):
        """
        This ensures that the `clauses` attribute contains  all possible clauses, including those defined by
        the `clauses_groups`.
        """
        others: tuple[str, ...] = tuple(dict.fromkeys(sum([group.elems for group in self.clauses_groups], tuple())))
        object.__setattr__(self, 'clauses', self.clauses + others)


def combine(a: str, b: str, exclude: set[str] = ()) -> (str, Directive):
    """
    Combines two directives to create a combined directives.

    Args:
        a (str): The name of the first directive to combine.
        b (str): The name of the second directive to combine.
        exclude (set[str], optional): Clause names to exclude from the combined directive.

    Returns:
        tuple: A tuple containing:
            - A string representing the combined name of the two directives.
            - A `Directive` object with the combined directives.
    """
    return (f"{a} {b}",
            Directive(clauses=tuple((c for c in DIRECTIVES[a].clauses + DIRECTIVES[b].clauses if c not in exclude))))


MODIFIERS: dict[str, Modifier] = {
    # Generic
    M_DIRECTIVE_NAME: Modifier(match=lambda n, t: n in DIRECTIVES, msg_value="directive"),
    # Data environment directives
    M_REDUCTION_ID: Modifier(required=True, transform=T_ITEM_TOKEN,
                             match=lambda n, t: n in "+ * & | ^ && ||".split() or t == NAME,
                             msg_value="'+' or '*' or '&' or '|' or '^' or '&&' or '||' or identifier"),
    M_MAPPER_IDENTIFIER: Modifier(match=lambda n, t: t == NAME, msg_value="identifier"),
    M_STORAGE: Modifier(values=(M_STORAGE,), ),
    M_FROM: Modifier(values=(M_FROM,)),
    M_TO: Modifier(values=(M_TO,)),
    M_ALWAYS: Modifier(values=(M_ALWAYS,)),
    M_CLOSE: Modifier(values=(M_CLOSE,)),
    M_PRESENT: Modifier(values=(M_PRESENT,)),
    M_MAPPER: Modifier(values=(M_MAPPER,), args=Arguments(num_args=1)),
    M_ITERATOR: Modifier(values=(M_ITERATOR,)),
    # Memory management directives

    # Variant directives

    # Informational and utility directives

    # Loop-transforming constructs

    # Parallelism constructs
    M_VARIABLE_CATEGORY: Modifier(values=(K_AGGREGATE, K_ALL, K_ALLOCATABLE, K_POINTER, K_SCALAR)),
    M_SAVED: Modifier(values=(M_SAVED,)),
    M_LOWER_BOUND: Modifier(match=lambda n, t: True, transform=T_ITEM_EXP, msg_value="expression"),
    M_ORDER_MODIFIER: Modifier(values=(K_REPRODUCIBLE, K_UNCONSTRAINED)),
    # Work-distribution constructs

    # Tasking constructs

    # Device directives and constructs

    # Interoperability construct

    # Synchronization constructs

    # Cancellation constructs

}

CLAUSES: dict[str, Clause] = {
    # Data environment directives
    C_INITIALIZER: Clause(required=True, args=Arguments(transform=T_ITEM_STM)),
    C_COMBINER: Clause(required=True, args=Arguments(transform=T_ITEM_STM)),
    C_COLLECTOR: Clause(required=True, args=Arguments(transform=T_ITEM_STM)),
    C_INDUCTOR: Clause(required=True, args=Arguments(transform=T_ITEM_STM)),
    C_EXCLUSIVE: Clause(args=Arguments()),
    C_INCLUSIVE: Clause(args=Arguments()),
    C_INIT_COMPLETE: Clause(args=Arguments(require_args=False, transform=T_ITEM_ID,
                                           num_args=1, choices=(K_CREATE_INIT_PHASE,))),
    C_MAP: Clause(required=True, repeatable=True,
                  args=Arguments(transform=T_ITEM_VAR,
                                 modifiers_groups=(Group(required=True, elems=(M_STORAGE, M_FROM, M_TO, M_TOFROM)),
                                                   Group(elems=(M_ALWAYS, M_CLOSE, M_PRESENT, M_MAPPER, M_ITERATOR))))),
    C_DEVICE_TYPE: Clause(args=Arguments(num_args=1, choices=(K_HOST, K_NOHOST, K_ANY), )),
    # Memory management directives

    # Variant directives

    # Informational and utility directives

    # Loop-transforming constructs

    # Parallelism constructs
    C_ALLOCATE: Clause(args=Arguments()),  # TODO check modifiers
    C_COPYIN: Clause(args=Arguments()),
    C_DEFAULT: Clause(args=Arguments(num_args=1, choices=(K_FIRSTPRIVATE, K_NONE, K_PRIVATE, K_SHARED),
                                     modifiers=(M_VARIABLE_CATEGORY,), post_modified=True), ),
    C_FIRSTPRIVATE: Clause(repeatable=True, args=Arguments(modifiers=(M_SAVED,)), ),
    C_IF: Clause(args=Arguments(num_args=1, transform=T_ITEM_EXP)),
    C_MESSAGE: Clause(args=Arguments(num_args=1, transform=T_ITEM_EXP)),
    C_NUM_THREADS: Clause(args=Arguments(transform=T_ITEM_EXP)),
    C_PRIVATE: Clause(repeatable=True, args=Arguments(), ),
    C_PROC_BIND: Clause(args=Arguments(num_args=1, choices=(K_CLOSE, K_PRIMARY, K_SPREAD))),
    C_REDUCTION: Clause(repeatable=True, args=Arguments(transform=T_ITEM_VAR, modifiers=(M_REDUCTION_ID,)), ),
    C_SAFESYNC: Clause(args=Arguments(require_args=False, num_args=1, transform=T_ITEM_EXP)),
    C_SEVERITY: Clause(args=Arguments(num_args=1, choices=(K_FATAL, K_WARNING))),
    C_SHARED: Clause(repeatable=True, args=Arguments(), ),
    C_NUM_TEAMS: Clause(args=Arguments(num_args=1, transform=T_ITEM_EXP, modifiers=(M_LOWER_BOUND,))),
    C_THREAD_LIMIT: Clause(args=Arguments(num_args=1, transform=T_ITEM_EXP)),
    # Work-distribution constructs
    C_COPYPRIVATE: Clause(args=Arguments(), ),
    C_NOWAIT: Clause(args=Arguments(require_args=False, num_args=1, transform=T_ITEM_EXP)),
    C_LASTPRIVATE: Clause(args=Arguments(modifiers=(M_LASTPRIVATE_MODIFIER,)), ),
    C_COLLAPSE: Clause(args=Arguments(num_args=1, transform=T_ITEM_CONST)),
    C_INDUCTION: Clause(args=Arguments(), ),  # TODO check modifiers
    C_LINEAR: Clause(args=Arguments(), ),  # TODO check modifiers
    C_ORDER: Clause(args=Arguments(num_args=1, choices=(K_CONCURRENT,), modifiers=(M_ORDER_MODIFIER,))),
    C_ORDERED: Clause(args=Arguments(require_args=False, num_args=1, transform=T_ITEM_EXP)),
    C_SCHEDULE: Clause(args=Arguments(transform=T_ITEM_KIND, modifiers=(M_SIMD,),
                                      choices=(K_STATIC, K_DYNAMIC, K_GUIDED, K_RUNTIME, K_AUTO),
                                      modifiers_groups=(Group(elems=(M_ORDER_MODIFIER, M_ORDERING_MODIFIER)),), )),
    # Tasking constructs
    C_UNTIED: Clause(),

    # Device directives and constructs

    # Interoperability construct

    # Synchronization constructs

    # Cancellation constructs
}
"""
A dictionary of parseable  clauses, each associated with a Clause object.
"""

DIRECTIVES: dict[str, Directive] = {
    # Data environment directives
    D_THREADPRIVATE: Directive(args=Arguments()),
    D_DECLARE: Directive(prefix=True),
    D_DECLARE_REDUCTION: Directive(clauses=(C_INITIALIZER, C_COMBINER),
                                   args=Arguments(modifiers=(M_REDUCTION_ID,))),
    D_DECLARE_INDUCTION: Directive(clauses=(C_COLLECTOR, C_INDUCTOR),
                                   args=Arguments(modifiers=(M_REDUCTION_ID,))),
    D_SCAN: Directive(clauses_groups=(Group(elems=(C_EXCLUSIVE, C_INCLUSIVE, C_INIT_COMPLETE)),), args=Arguments()),
    D_DECLARE_MAPPER: Directive(clauses=(C_MAP,), args=Arguments(num_args=1, modifiers=(M_MAPPER_IDENTIFIER,))),
    D_GROUPPRIVATE: Directive(clauses=(C_DEVICE_TYPE,)),
    # Memory management directives

    # Variant directives

    # Informational and utility directives

    # Loop-transforming constructs

    # Parallelism constructs
    D_PARALLEL: Directive(clauses=(C_ALLOCATE, C_COPYIN, C_DEFAULT, C_FIRSTPRIVATE, C_IF, C_MESSAGE, C_NUM_THREADS,
                                   C_PRIVATE, C_PROC_BIND, C_REDUCTION, C_SAFESYNC, C_SEVERITY, C_SHARED)),
    D_TEAMS: Directive(clauses=(C_ALLOCATE, C_DEFAULT, C_FIRSTPRIVATE, C_IF, C_NUM_TEAMS, C_PRIVATE, C_REDUCTION,
                                C_SHARED, C_THREAD_LIMIT)),

    # Work-distribution constructs
    D_SINGLE: Directive(clauses=(C_ALLOCATE, C_COPYPRIVATE, C_FIRSTPRIVATE, C_NOWAIT, C_PRIVATE)),
    D_SCOPE: Directive(clauses=(C_ALLOCATE, C_FIRSTPRIVATE, C_NOWAIT, C_PRIVATE, C_REDUCTION)),
    D_SECTIONS: Directive(clauses=(C_ALLOCATE, C_FIRSTPRIVATE, C_LASTPRIVATE, C_NOWAIT, C_PRIVATE, C_REDUCTION)),
    D_SECTION: Directive(),
    D_FOR: Directive(clauses=(C_ALLOCATE, C_COLLAPSE, C_FIRSTPRIVATE, C_INDUCTION, C_LASTPRIVATE, C_LINEAR, C_NOWAIT,
                              C_ORDER, C_ORDERED, C_PRIVATE, C_REDUCTION, C_SCHEDULE)),
    D_DISTRIBUTE: Directive(),
    D_LOOP: Directive(),

    # Tasking constructs
    D_TASK: Directive(clauses=(C_IF, C_UNTIED, C_DEFAULT, C_PRIVATE, C_FIRSTPRIVATE, C_SHARED)),
    D_TASK_WAIT: Directive(),

    # Device directives and constructs

    # Interoperability construct

    # Synchronization constructs
    D_CRITICAL: Directive(),  # TODO hint clause
    D_BARRIER: Directive(),

    # Cancellation constructs

}
"""
A dictionary of parseable  directives, each associated with a Directive object.
"""

# Update the DIRECTIVES dictionary with the combined directives.
DIRECTIVES.update([
    combine(D_PARALLEL, D_FOR, exclude={C_NOWAIT})
])
