"""Extension API for integrating custom `omp4py` components.

This module provides the public registration interface used by external
extensions that integrate with the `omp4py` preprocessing system.

Modules registered through `omp(extension=...)` are loaded when the
preprocessor becomes available and can use this API to extend the
transformation pipeline with custom parsers, devices, and modifiers.

Registered parsers become available through the `parsers` option of
the `omp` decorator. Once enabled, directives using the corresponding
parser name (for example, `myparser(...)`) are recognized and processed
by the transformation pipeline.

The module also re-exports utility symbols commonly required when
implementing custom transformations. In particular, `omp_construct`
allows extensions to delegate processing to the default OpenMP
construct transformer, while `syntax_error_ctx` provides helpers for
raising syntax errors with accurate source-location information.
"""
from __future__ import annotations

import typing

from omp4py.core.modifiers.engine import Modifier, modifier
from omp4py.core.preprocessor.transformers.transformer import DEVICES, PARSERS, syntax_error_ctx
from omp4py.core.preprocessor.transformers.transformer import construct as omp_construct

if typing.TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    from omp4py.core.parser import Construct, Directive, Span
    from omp4py.core.preprocessor.transformers.context import Context

__all__ = ["Modifier", "add_device", "add_modifier", "add_parser", "omp_construct", "syntax_error_ctx"]

type Parser = Callable[[str, Span, str], Directive]
type Device = Callable[[Construct, list[ast.stmt], Context], list[ast.stmt]]

def add_parser(name: str) -> Callable[[Parser], Parser]:
    """Register a custom directive parser.

    This decorator registers a parser implementation under the specified
    name. Registered parsers can later be enabled through the `parsers`
    option of the `omp` decorator and are used to recognize directive
    syntaxes beyond those supported by the default parser.

    Once enabled, directives using the registered parser name are
    dispatched to the associated parsing function during preprocessing.

    Args:
        name (str):
            Unique parser identifier used in source code and in the
            `parsers` preprocessing option.

    Returns:
        Callable[[Parser], Parser]:
            A decorator that registers the parser function and returns
            it unchanged.
    """
    def add(f: Parser) -> Parser:
        PARSERS[name] = f
        return f
    return add

def add_device(name: str) -> Callable[[Device], Device]:
    """Register a custom transformation device.

    Devices are responsible for generating AST nodes for specific
    execution targets. During transformation, constructs are delegated
    to the device whose name matches the value of the `device` option
    specified in the `omp` decorator.

    Args:
        name (str):
            Unique device identifier used to select the device during
            transformation.

    Returns:
        Callable[[Device], Device]:
            A decorator that registers the device implementation and
            returns it unchanged.
    """
    def add(f: Device) -> Device:
        DEVICES[name] = f
        return f
    return add

def add_modifier(cls: type[Modifier]) -> type[Modifier]:
    """Register a custom AST modifier.

    This helper registers a modifier class with the global modifier
    registry, making it available to the modifier engine.

    Registered modifiers can be enabled or disabled through the
    `modifiers` option of the `omp` decorator and participate in the
    preprocessing pipeline before or after the main OpenMP
    transformation stage.

    Args:
        cls (type[Modifier]):
            Modifier class to register.

    Returns:
        type[Modifier]:
            The same modifier class after registration.
    """
    return modifier(cls)
