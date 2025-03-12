import ast
import dataclasses
from typing import Any, Callable, Protocol, Optional

from omp4py.core.directive import tokenizer
from omp4py.core.directive.schema import *
from omp4py.core.directive.names import *

TokenInfo = tokenizer.TokenInfo

__all__ = ['parse_args', 'ArgsParser', 'OmpArgs', 'OmpItem', 'PARSERS']

PARSERS: dict[str, 'ArgsParser'] = {}
"""
Dictionary that maps parser names to their corresponding `ArgsParser` objects.
"""

TRANSFORMERS: dict[str, 'ItemTransformer'] = {}
"""
Dictionary that maps transformer names to their corresponding `ItemTransformer` objects.
"""


@dataclasses.dataclass(frozen=True)
class OmpArgs:
    """
    Represents the arguments for an OpenMP `Directive`, `Clause` or `Modifier`.

    Attributes:
        lpar (TokenInfo): The `TokenInfo` object representing the left parenthesis `(`.
        modifiers (tuple['OmpItem', ...]): `OmpItem` objects representing modifiers associated with arguments.
        array (tuple['OmpItem', ...]): `OmpItem` objects representing the array of arguments.
        rpar (TokenInfo): The `TokenInfo` object representing the right parenthesis `)`.

    """
    lpar: TokenInfo
    modifiers: tuple['OmpItem', ...]
    array: tuple['OmpItem', ...]
    rpar: TokenInfo
    next_args: Optional['OmpArgs'] = None


@dataclasses.dataclass(frozen=True)
class OmpItem:
    """
    Represents an OpenMP item, which contains information about its name, tokens, value, and associated arguments.

    Attributes:
        name (str): The name of the OpenMP item.
        tokens (tuple['OmpItem', ...]): `TokenInfo` objects representing the tokens associated with the item.
        value (typing.Any): The value associated with the item. Can be of any type, depending on the context.
        args (OmpArgs | None ): Optional `OmpArgs` object that contains arguments for the item.

    """
    name: str
    tokens: tuple[TokenInfo, ...]
    value: Any
    args: OmpArgs | None = None


class ArgsParser(Protocol):
    """
     A protocol for a callable object that parses arguments
    """

    def __call__(self, specs: Arguments, lpar: TokenInfo, args: list[TokenInfo], rpar: TokenInfo) -> OmpArgs:
        """
        Processes the arguments enclosed within parentheses and returns the parsed `OmpArgs` object.

        Args:
            specs (Arguments): An `Arguments` object that provides context and specifications for parsing.
            lpar (TokenInfo): A `TokenInfo` object representing the opening parenthesis `(`.
            args (list[TokenInfo]): A list of `TokenInfo` objects representing the arguments enclosed within the parentheses.
            rpar (TokenInfo): A `TokenInfo` object representing the closing parenthesis `)`.

        Returns:
            OmpArgs: A structure containing the parsed arguments and modifiers.
        """
        pass


class ItemTransformer(Protocol):
    """
    A protocol for a callable object that transforms tokens to `OmpItem`.
    """

    def __call__(self, i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
        """
        Transforms a list of tokens into a `OmpItem`.

        Args:
            i (int): An index representing the position of the item in the modifiers or arguments.
            name (str): The name of the modifier or `M_ARGS`.
            specs (Arguments): An `Arguments` object that provides context and specifications for parsing the tokens.
            tokens (list[TokenInfo]): `TokenInfo` objects representing the tokens that need to be transformed.
            sep (TokenInfo): A `TokenInfo` object representing the separator token (e.g., a comma) used between items.

        Returns:
            OmpItem: A  `OmpItem` that represents the transformed OpenMP item.
        """
        pass


def arg_parser(name: str) -> Callable[[ArgsParser], ArgsParser]:
    """
    A decorator function for registering a function as a parser for OpenMP arguments.

    Args:
        name (str): The name under which the parser will be registered in the `PARSERS` dictionary.

    Returns:
        Callable[[ArgsParser], ArgsParser]: A decorator function that registers the given `ArgsParser` function.
    """

    def w(f: ArgsParser) -> ArgsParser:
        PARSERS[name] = f
        return f

    return w


def transformer(name: str) -> Callable[[ItemTransformer], ItemTransformer]:
    """
    A decorator function for registering a function as a transformer for OpenMP items.

    Args:
        name (str): The name under which the transformer will be registered in the `TRANSFORMERS` dictionary.

    Returns:
        Callable[[ItemTransformer], ItemTransformer]: A decorator function that registers the given `ItemTransformer`
                                                       function.
    """

    def w(f: ItemTransformer) -> ItemTransformer:
        TRANSFORMERS[name] = f
        return f

    return w


def consume_args(tokens: list[TokenInfo]) -> int:
    """
    Consumes the arguments enclosed in parentheses from a list of tokens.

    Args:
        tokens (list[TokenInfo]): A list of TokenInfo objects representing the tokens to parse.

    Returns:
        int: The index of the last token in the argument list (just after the closing parenthesis), or -1 if the
              opening parenthesis '(' is not found.

    Raises:
        SyntaxError: If the parentheses are not balanced.
    """
    lvl: list[TokenInfo] = list()
    i: int = 0
    token: TokenInfo
    for i, token in enumerate(tokens):
        if token.type == tokenizer.LPAR:
            lvl.append(token)
        elif token.type == tokenizer.RPAR:
            lvl.pop()
        elif len(lvl) == 0:
            return -1
        if len(lvl) == 0:
            break
    if len(lvl) > 0:
        raise lvl[-1].make_error(f"'{lvl[-1]}' was never closed")

    return i if i > 0 else -1


def parse_args(specs: Arguments, tokens: list[TokenInfo]) -> (int, OmpArgs | None):
    """
    Parses the arguments of a directive or clause using the provided parser.

    Args:
        tokens (list[TokenInfo]): A list of TokenInfo objects representing the tokens to be parsed.

    Returns:
        tuple: A tuple where the first element is the number of arguments parsed,
               and the second is the result of parsing the arguments, or None if no parser function was provided.

    Raises:
        SyntaxError: If the arguments are not correctly enclosed in parentheses.
    """
    if specs is None:
        return 0, None

    i_args: int = consume_args(tokens[1:])
    if i_args < 0:
        if not specs.require_args:
            return 0, None
        raise tokenizer.expected_error(tokens[1], "'('")
    token_args: list[TokenInfo] = tokens[1: 2 + i_args]
    return i_args + 1, PARSERS[specs.parser](specs, token_args[0], token_args[1:-1], token_args[-1])


def find_separator(tokens: list[TokenInfo], sep: int) -> int:
    """
    Finds the position of a separator token in a list of tokens, considering various delimiters like square brackets,
    parentheses, and braces, and ensuring that they are properly balanced before identifying the separator.

    Args:
        tokens (list[TokenInfo]): A list of `TokenInfo` objects representing the tokens to be parsed.
        sep (int): The type of the separator token to be found (e.g., a COMMA, COLON, etc.).

    Returns:
        int: The index of the separator token in the list of tokens.

    Raises:
        SyntaxError: If there is an unbalanced.
    """
    sqbs: list[TokenInfo] = []
    pars: list[TokenInfo] = []
    braces: list[TokenInfo] = []

    i: int = 0
    token: TokenInfo
    try:
        for i, token in enumerate(tokens):
            if token.type == tokenizer.LSQB:
                sqbs.append(token)
            elif token.type == tokenizer.RSQB:
                sqbs.pop()
            elif token.type == tokenizer.LPAR:
                pars.append(token)
            elif token.type == tokenizer.RPAR:
                pars.pop()
            elif token.type == tokenizer.LBRACE:
                braces.append(token)
            elif token.type == tokenizer.RBRACE:
                braces.pop()
            elif len(sqbs) + len(pars) + len(braces) == 0 and token.type == sep:
                return i
        else:
            i = len(tokens)

        if len(sqbs) + len(pars) + len(braces) > 0:
            t: TokenInfo = (sqbs + pars + braces)[-1]
            raise t.make_error(f"'{t}' was never closed")
    except IndexError:
        raise tokens[i].make_error(f"'{tokens[i]}' was never open")

    return i


def parse_python(tokens: list[TokenInfo]) -> ast.Module:
    """
    Parses a list of tokens into an abstract syntax tree (AST) for Python code.

    This function takes a list of `TokenInfo` objects, untokenizes them into a code string, and parses it into an
    AST module. It also adjusts the line and column numbers for each node in the AST based on the token positions
    to reflect the correct source location.

    Args:
        tokens (list[TokenInfo]): A list of `TokenInfo` objects representing the Python source code tokens.

    Returns:
        ast.Module: The AST module representing the parsed Python code.

    Raises:
        SyntaxError: If there is a syntax error during parsing, an exception is raised with the adjusted line and
                     column information.
    """
    offsets: list[int] = [0]
    lineno:int = tokens[0].lineno
    new_tokens: list[TokenInfo] = []
    token: TokenInfo
    for token in tokens:
        while token.start[0] > len(offsets):
            offsets.append(0)
        if token.start[0] == len(offsets):
            offsets.append(token.start[1])
        new_tokens.append(dataclasses.replace(token,
                                              start=(token.start[0], token.start[1] - offsets[token.start[0]]),
                                              end=(token.end[0], token.end[1] - offsets[token.end[0]])
                                              ))

    code: str = tokenizer.untokenize(new_tokens)
    module: ast.Module
    try:
        module= ast.parse(code)
    except SyntaxError as ex:
        raise SyntaxError(ex.msg, (tokens[0].filename, ex.lineno + lineno, ex.offset + offsets[ex.lineno],
                                   tokens[0].line, ex.end_lineno + lineno, ex.end_offset + offsets[ex.end_lineno])
                          ) from None
    node: ast.AST
    for node in ast.walk(module):
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno') and \
                hasattr(node, 'col_offset') and hasattr(node, 'end_col_offset'):
            node.col_offset += offsets[node.lineno]
            node.end_col_offset += offsets[node.end_lineno]
            node.lineno += lineno - 1
            node.end_lineno += lineno - 1

    return module


def str_mod(mod_name: str) -> str:
    """
    Creates the string representation of a modifier.

    Args:
        mod_name (str): The modifier name.

    Returns:
        str: The string representation of a modifier.
    """
    mod_specs: Modifier = MODIFIERS[mod_name]
    if mod_specs.msg_value is not None:
        return mod_specs.msg_value
    elif mod_specs.values is not None:
        return ' or '.join(f"'{v}'" for v in mod_specs.values)
    else:
        return mod_name


def _parse_modifiers(specs: Arguments, lpar: TokenInfo, tokens: list[TokenInfo], rpar: TokenInfo) -> list[OmpItem]:
    """
    Parses a list of tokens to extract modifiers.

    Args:
        specs (Arguments): The argument specifications.
        lpar (TokenInfo): A `TokenInfo` object representing the opening parenthesis '('.
        tokens (list[TokenInfo]): A list of `TokenInfo` objects representing the tokens.
        rpar (TokenInfo): A `TokenInfo` object representing the closing parenthesis ')'.

    Returns:
        list[OmpItem]: A list of parsed modifier items, each representing a modifier and its associated arguments.

    Raises:
        SyntaxError: If an invalid modifier is encountered or required modifiers are missing.
    """

    parsed_modifiers: list[OmpItem] = list()
    avail_mods: list[str] = list(specs.modifiers)
    used_mods: set[str] = set()

    i: int = 0
    while i < len(tokens):
        c_sep: int = max(find_separator(tokens[i:], sep=tokenizer.COMMA) + i, i + 1)
        mod_name: str
        for mod_name in avail_mods:
            mod_specs: Modifier = MODIFIERS[mod_name]

            old_sep: int = c_sep
            # modifiers with '(' are complex modifiers
            if i + 1 < len(tokens) and tokens[i + 1] == tokenizer.LPAR and mod_specs.transform == T_ITEM_ID:
                c_sep = i + 1

            if mod_specs.values is not None and len(tokens[i:c_sep]) == 1 and tokens[i].id in mod_specs.values:
                pass
            elif mod_specs.token_match is not None and mod_specs.token_match(tokens[i:c_sep]):
                pass
            elif c_sep == i + 1 and mod_specs is not None and mod_specs.match(tokens[i].id, tokens[i].type):
                pass
            elif c_sep > i + 1 and mod_specs is not None and \
                    mod_specs.match(tokenizer.untokenize(tokens[i:c_sep]).strip(), -1):
                pass
            else:
                c_sep = old_sep
                continue
            break
        else:
            raise tokens[i].make_error(f"'{tokens[i].string}' is not a valid modifier")

        mod: OmpItem = TRANSFORMERS[mod_specs.transform](len(parsed_modifiers), mod_name, specs, tokens[i:c_sep],
                                                         tokens[c_sep] if c_sep < len(tokens) else rpar)
        n_args: int
        args: OmpArgs
        n_args, args = parse_args(mod_specs.args, tokens[i + 1:])

        used_mods.add(mod_name)
        if not mod_specs.repeatable:
            avail_mods.remove(mod_name)

        parsed_modifiers.append(dataclasses.replace(mod, args=args) if n_args > 0 else mod)
        i += c_sep + n_args

    for mod_name in avail_mods:  # check required modifiers
        if MODIFIERS[mod_name].required and mod_name not in used_mods:
            raise tokenizer.expected_error(tokens[-1] if tokens else rpar, str_mod(mod_name))

    group_mod: Group
    for group_mod in specs.modifiers_groups:
        uses: list[int] = [1 for elem in group_mod.elems if elem in used_mods]

        if group_mod.required and len(uses) == 0:
            msg: str = ' or '.join(str_mod(mod_name) for mod_name in group_mod.elems)
            raise tokenizer.expected_error(tokens[-1] if tokens else rpar, msg)

        if group_mod.exclusive and len(uses) > 1:
            positions: list[int] = [index for index, value in enumerate(uses) if value == 1]
            a: OmpItem = parsed_modifiers[positions[0]]
            b: OmpItem = parsed_modifiers[positions[1]]

            raise tokenizer.merge(b.tokens).make_error(f"'{tokenizer.untokenize(a.tokens).strip()}' and "
                                                       f"'{tokenizer.untokenize(b.tokens).strip()}' cannot be "
                                                       f"used together")

    mod: OmpItem
    j: int
    for j, mod in enumerate(parsed_modifiers[:-1]):  # check ultimate modifiers
        if MODIFIERS[mod.name].ultimate and not MODIFIERS[mod.name].ultimate:
            raise tokenizer.expected_error(tokenizer.merge(parsed_modifiers[j + 1].tokens), f"')'")

    return parsed_modifiers


@arg_parser(name=A_PARSER_BASIC)
def parser_basic(specs: Arguments, lpar: TokenInfo, tokens: list[TokenInfo], rpar: TokenInfo) -> OmpArgs:
    """
    Parses the arguments for a basic `Directive`, `Clause` or `Modifier`.

    Args:
        specs (Arguments): The argument specifications.
        lpar (TokenInfo): A `TokenInfo` object representing the opening parenthesis '('.
        tokens (list[TokenInfo]): A list of `TokenInfo` objects representing the tokens.
        rpar (TokenInfo): A `TokenInfo` object representing the closing parenthesis ')'.

    Returns:
        OmpArgs: An object containing the parsed modifiers and arguments.

    Raises:
        tokenizer.expected_error: If there is an error in the modifiers or arguments.
    """
    if len(tokens) == 0:
        TRANSFORMERS[specs.transform](0, M_ARGS, specs, tokens, rpar)  # transformer will raise the error

    next_args: OmpArgs | None = None
    if specs.multiple:
        # Workaround for clauses with multiple sets of arguments.
        # Currently, only one clause use it, so I prefer a simpler approach.
        semi_sep = find_separator(tokens, tokenizer.SEMI)
        if semi_sep < len(tokens):
            next_args = parser_basic(specs, tokens[semi_sep], tokens[semi_sep + 1:], rpar)
        rpar = tokens[semi_sep]  # to be coherent
        tokens = tokens[:semi_sep]

    colon_sep: int = find_separator(tokens, sep=tokenizer.COLON)
    if colon_sep == len(tokens):
        colon_sep = -1

    modifiers: list[OmpItem] = []
    i: int = 0
    if not specs.post_modified and colon_sep != -1:
        modifiers = _parse_modifiers(specs, lpar, tokens[:colon_sep], rpar)
        i = colon_sep + 1

    args: list[OmpItem] = []
    args_transform: ItemTransformer = TRANSFORMERS[specs.transform]
    i = colon_sep + 1
    while i < len(tokens):
        c_sep: int = max(find_separator(tokens[i:], sep=tokenizer.COMMA) + i, i + 1)
        args.append(args_transform(len(args), M_ARGS, specs, tokens[i:c_sep], rpar))
        i = c_sep + 1

    if len(args) == 0 or specs.num_args > len(args):
        TRANSFORMERS[specs.transform](len(args), M_ARGS, specs, tokens, rpar)  # transformer will raise the error
    elif len(args) > specs.num_args > 0:
        raise tokenizer.expected_error(args[specs.num_args].tokens[0], f"')'")

    if specs.post_modified and colon_sep != -1:
        modifiers = _parse_modifiers(specs, lpar, tokens[colon_sep + 1:], rpar)

    return OmpArgs(lpar, tuple(modifiers), tuple(args), rpar, next_args)


def new_item_concat(*args: str) -> ItemTransformer:
    """
    Creates a new `ItemTransformer` that concatenates multiple transformations.

    Args:
        *args (str): The names of the transformations to be concatenated. These names correspond to the entries
                     in the `TRANSFORMERS` dictionary, and each will be applied in the order given.

    Returns:
        ItemTransformer: A new transformer function that applies the specified transformations sequentially.
                         The resulting `OmpItem` will be produced by applying each transformer in turn.

    """

    def w(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
        if i >= len(args):
            raise tokenizer.expected_error(tokens[0], f"'{sep}'")
        return TRANSFORMERS[args[i]](i, name, specs, tokens, sep)

    return w


@transformer(name=T_ITEM_ID)
def item_id(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
    """
    Parses a single argument that must be a valid identifier.

    This function expects exactly one argument, which should be a valid identifier. If the identifier is invalid,
    an error is raised.
    """
    if len(tokens) == 0:
        if name in MODIFIERS:
            raise tokenizer.expected_error(sep, str_mod(name))
        else:
            raise tokenizer.expected_error(sep, 'identifier')

    if len(tokens) > 1 or tokens[0].type != tokenizer.NAME:
        raise tokenizer.expected_error(tokenizer.merge(tokens), 'identifier')

    if name == M_ARGS and specs.choices:
        if str(tokens[0]) not in specs.choices:
            v: str
            raise tokenizer.expected_error(tokens[0], " or ".join([f"'{v}'" for v in specs.choices]))

    return OmpItem(name=name, tokens=tuple(tokens), value=str(tokens[0]))


@transformer(name=T_ITEM_VAR)
def item_var(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
    """
    Parses a token list and returns an OmpItem representing an identifier or array section.
    """
    if len(tokens) == 0:
        if name in MODIFIERS:
            raise tokenizer.expected_error(sep, str_mod(name))
        else:
            raise tokenizer.expected_error(sep, 'identifier')

    module: ast.Module = parse_python(tokens)

    if len(module.body) > 1 or not isinstance(module.body[0], ast.Expr) or \
            not isinstance(module.body[0].value, (ast.Name, ast.Subscript)):
        raise tokenizer.expected_error(tokenizer.merge(tokens), 'identifier or array section')

    exp: ast.expr = module.body[0].value

    if isinstance(exp, ast.Subscript) and not isinstance(exp.value, ast.Name):
        raise tokenizer.expected_error(tokenizer.merge(tokens), 'local identifier')

    return OmpItem(name=name, tokens=tuple(tokens), value=exp)

@transformer(name=T_ITEM_CONST)
def item_const(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
    """
    Parses a single argument that must be a constant valid Python expression.

    This function expects exactly one argument, which should be a valid Python expression. If the expression is invalid,
    an error is raised.
    """
    if len(tokens) == 0:
        if name in MODIFIERS:
            raise tokenizer.expected_error(sep, str_mod(name))
        else:
            raise tokenizer.expected_error(sep, 'constant')

    try:
        return OmpItem(name=name, tokens=tuple(tokens), value=eval(tokenizer.untokenize(tokens), {}, {}))
    except Exception as ex:
        raise tokenizer.expected_error(tokenizer.merge(tokens), 'expression must be constant') from ex

@transformer(name=T_ITEM_EXP)
def item_exp(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
    """
    Parses a single argument that must be a valid Python expression.

    This function expects exactly one argument, which should be a valid Python expression. If the expression is invalid,
    an error is raised.
    """
    if len(tokens) == 0:
        if name in MODIFIERS:
            raise tokenizer.expected_error(sep, str_mod(name))
        else:
            raise tokenizer.expected_error(sep, 'expression')

    module: ast.Module = parse_python(tokens)

    if not isinstance(module.body[0], ast.Expr):
        raise tokenizer.expected_error(tokenizer.merge(tokens), 'expression')

    if len(module.body) > 1:
        for i in range(len(tokens)):
            start: tuple[int, int] = tokens[i].start
            if hasattr(module.body[1], 'lineno') and \
                    module.body[1].lineno <= start[0] + tokens[i].lineno and module.body[1].col_offset <= start[1]:
                break
        else:
            i = 0

        raise tokenizer.expected_error(tokenizer.merge(tokens[i:]), "')'")

    return OmpItem(name=name, tokens=tuple(tokens), value=module.body[0].value)


@transformer(name=T_ITEM_STM)
def item_smt(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
    """
    Transforms a list of tokens into an OmpItem for a Python statement.

    This transformer parses a list of tokens, processes them as Python code, and returns an `OmpItem` object
    containing the parsed Python module body as its value.
    """
    if len(tokens) == 0:
        if name in MODIFIERS:
            raise tokenizer.expected_error(sep, str_mod(name))
        else:
            raise tokenizer.expected_error(sep, 'statement')

    module: ast.Module = parse_python(tokens)
    return OmpItem(name=name, tokens=tuple(tokens), value=module.body)


@transformer(name=T_ITEM_TOKEN)
def item_token(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
    """
    Transforms a list of tokens into an OmpItem by untokenizing them.

    This transformer takes a list of tokens and returns an `OmpItem` object where the `value` is the untokenized
    string representation of the tokens.
    """
    if len(tokens) == 0:
        if name in MODIFIERS:
            raise tokenizer.expected_error(sep, str_mod(name))
        else:
            raise tokenizer.expected_error(sep, "argument")

    if len(tokens) > 1:
        raise tokenizer.expected_error(tokens[1], f"'{sep}'")

    return OmpItem(name=name, tokens=tuple(tokens), value=str(tokens[0]))


@transformer(name=T_ITEM_KIND)
def item_kind(i: int, name: str, specs: Arguments, tokens: list[TokenInfo], sep: TokenInfo) -> OmpItem:
    if i == 0:
        return TRANSFORMERS[T_ITEM_ID](i, name, specs, tokens, sep)
    elif i == 1:
        return TRANSFORMERS[T_ITEM_EXP](i, name, specs, tokens, sep)
    elif len(tokens) > 0:
        raise tokenizer.expected_error(tokens[0], f"'{sep}'")
    else:
        raise tokenizer.expected_error(sep, f"')'")
