import dataclasses

from omp4py.core.directive import tokenizer
from omp4py.core.directive.argsparser import OmpArgs, OmpItem, parse_args
from omp4py.core.directive.names import M_DIRECTIVE_NAME
from omp4py.core.directive.schema import *

TokenInfo = tokenizer.TokenInfo

__all__ = ['OmpClause', 'OmpDirective', 'OmpArgs', 'OmpItem', 'parse_line']


@dataclasses.dataclass(frozen=True)
class OmpClause:
    """
    Represents a parsed clause within a directive.

    Attributes:
        token (TokenInfo): The token representing the clause.
        args (OmpArgs | None): Arguments of the clause.
    """
    directive: str
    token: TokenInfo
    args: OmpArgs | None

    def __str__(self) -> str:
        """Returns a string representation of the token for this parsed clause."""
        return self.token.id


@dataclasses.dataclass(frozen=True)
class OmpDirective:
    """
    Represents a parsed directive, including its name, the tokens that compose it, and the associated clauses.

    Attributes:
        name (str): Full name of the directive.
        tokens (list[TokenInfo]): Tokens of the directive.
        args (OmpArgs | None): Arguments of the directive.
        clauses (tuple[OmpClause,...]): List of clauses of the directive.
    """
    name: str
    tokens: tuple[TokenInfo, ...]
    args: OmpArgs | None
    clauses: tuple[OmpClause, ...]

    def __str__(self) -> str:
        """Returns the name of the parsed directive."""
        return self.name


def parse_line(filename: str, line: str, lineno: int, preproc=False) -> OmpDirective:
    """
    Parses an OpenMP line of code.

    Args:
        filename (str): The name of the file where the line is located.
        line (str): The line of code to parse, containing the OpenMP directives.
        lineno (int): The line number in the file where the line is located.
        preproc (bool, optional): If True, preprocesses the line before parsing.

    Returns:
        OmpDirective: A `OmpDirective` object representing the parsed OpenMP directives and clauses.

    Raises:
        SyntaxError: If there is a syntax error in the line.
    """
    tokens: list[TokenInfo]
    token_error: str
    tokens, token_error = tokenizer.generate_tokens(filename, line, lineno, preproc)

    result_name: str = ""
    result_tokens: list[TokenInfo] = []
    result_args: OmpArgs | None = None
    result_clauses: list[OmpClause] = []

    used_directives: list[str] = []
    used_clauses: list[str] = []

    i: int = 0
    prefix: str = ""
    token_name: str
    # Directives
    while i < len(tokens) - 1:
        if i < len(tokens) and tokens[i].type != tokenizer.NAME:
            raise tokenizer.expected_error(tokens[i], "identifier")

        # Assume the current token is a clause
        if tokens[i].id not in DIRECTIVES and not prefix and len(result_name) > 0:
            break

        # Check that the combined directive is valid
        new_result_name: str = result_name + ("_" if prefix else " " if result_name else "") + tokens[i].id
        if new_result_name not in DIRECTIVES:
            if prefix:
                break  # raise an error later
            if len(result_name) == 0:
                raise tokenizer.expected_error(tokens[i], "a valid directive")
            raise tokens[i].make_error(f"'{tokens[i]}' is not valid for 'omp {result_name}'")
        result_name = new_result_name
        result_tokens.append(tokens[i])

        # Retrieve the directive and check if it is a prefix
        token_name = prefix + tokens[i].id
        specs: Directive = DIRECTIVES[token_name]
        if specs.prefix:
            prefix = token_name + "_"
            i += 1
            continue
        prefix = ""

        # Process the directive
        n_args: int
        n_args, result_args = parse_args(specs.args, tokens[i:])
        used_directives.append(token_name)
        i += n_args + 1

    # If the directive ends in a prefix, raise an error with the possible compositions
    if prefix:
        opts: str = ' or '.join(f"'{d.split('_')[-1]}'" for d in DIRECTIVES if d.startswith(prefix))
        raise tokenizer.expected_error(tokens[i], opts)

    # Clauses
    while i < len(tokens) - 1:
        if i < len(tokens) and tokens[i].type != tokenizer.NAME:
            raise tokenizer.expected_error(tokens[i], "identifier")

        token_name = tokens[i].id
        if token_name not in CLAUSES:
            raise tokens[i].make_error(f"'{token_name}' is not valid clause")

        n_args: int
        args: OmpArgs | None
        specs: Clause = CLAUSES[tokens[i].id]
        n_args, args = parse_args(specs.args, tokens[i:])
        if token_name in used_clauses and not specs.repeatable:
            raise tokens[i].make_error(f"too many '{token_name}' clauses")

        dir_name: str | None = None
        # Search the directive of the clause, first using the modifier and then iteratively
        if args is not None:
            mod: OmpItem
            for mod in args.modifiers:
                if mod.name == M_DIRECTIVE_NAME:
                    dir_name = mod.value
                    break
            if dir_name is not None and dir_name not in used_directives:
                valid_dirs: list[str] = [dir for dir in used_directives if token_name in DIRECTIVES[dir].clauses]
                if len(valid_dirs) > 0:
                    raise tokenizer.expected_error(tokenizer.merge(mod.tokens), " or ".join(valid_dirs))
                else:
                    raise tokenizer.merge(mod.tokens).make_error(
                        f"{dir_name} is not a valid directive for {token_name}")
        if dir_name is None:
            for dir_name in used_directives:
                if token_name in DIRECTIVES[dir_name].clauses:
                    break
            else:
                raise tokens[i].make_error(f"'{token_name}' is not valid clause for 'omp {result_name}'")

        used_clauses.append(token_name)
        result_clauses.append(OmpClause(dir_name, tokens[i], args))
        i += n_args + 1

    clause_name: str
    for dir_name in used_directives:
        dir_specs: Directive = DIRECTIVES[dir_name]
        for clause_name in dir_specs.clauses:  # check required clauses
            if CLAUSES[clause_name].required and clause_name not in used_clauses:
                raise tokenizer.expected_error(tokens[-1], f"'{clause_name}'")

        group_clause: Group
        for group_clause in dir_specs.clauses_groups:
            uses: list[int] = [1 for elem in group_clause.elems if elem in used_clauses]

            if group_clause.required and len(uses) == 0:
                msg: str = ' or '.join(elem for elem in group_clause.elems)
                raise tokenizer.expected_error(tokens[-1], msg)

            if group_clause.exclusive and len(uses) > 1:
                positions: list[int] = [index for index, value in enumerate(uses) if value == 1]
                a: OmpClause = result_clauses[positions[0]]
                b: OmpClause = result_clauses[positions[1]]

                raise b.token.make_error(f"'{a.token.string}' and '{b.token.string}' cannot be used together")

    clause: OmpClause
    j: int
    for j, clause in enumerate(result_clauses[:-1]):  # check ultimate clauses
        if CLAUSES[str(clause)].ultimate and not CLAUSES[str(clause)].ultimate:
            raise result_clauses[j].token.make_error(f"{result_clauses[j].token.string} must be the last clause")

    if token_error is not None:
        # The parser should have already failed earlier if there was an error.
        # This check ensures that in case the parser didn't fail as expected.
        raise tokens[-1].make_error(token_error)

    return OmpDirective(name=result_name, tokens=tuple(result_tokens), args=result_args, clauses=tuple(result_clauses))
