"""AST modifier system for the omp4py preprocessing pipeline.

This module defines the infrastructure used to register and execute
custom AST transformations (modifiers) that run before or after the
core OpenMP transformation phase.

Modifiers provide a flexible extension mechanism that allows adapting,
optimizing, or augmenting the generated AST. They can be selectively
enabled or disabled through the `omp(modifiers={})` and can declare
dependencies on other modifiers to control execution order.

Modifiers operate directly on the Python AST (`ast.Module`) and can
share state through a metadata dictionary, allowing coordination between
multiple passes.
"""

from __future__ import annotations

import ast
import typing

if typing.TYPE_CHECKING:
    from omp4py.core.options import Options

__all__ = ["Modifier", "ModifierEngine", "modifier"]

MODIFIERS: dict[str, type[Modifier]] = {}


def modifier(cls: type[Modifier]) -> type[Modifier]:
    """Register a modifier class in the global modifier registry.

    This decorator adds the given modifier class to the global
    `MODIFIERS` registry using its declared `name`. Registered
    modifiers become available for execution through the
    `ModifierEngine`.

    Args:
        cls (type[Modifier]):
            Modifier class to register.

    Returns:
        type[Modifier]:
            The same class, unmodified.
    """
    MODIFIERS[cls.name] = cls
    return cls


class ModifierEngine:
    """Execution engine for AST modifiers.

    This class is responsible for selecting, ordering, and executing
    registered modifiers based on the provided configuration and
    declared dependencies.

    Modifiers are executed in stages according to a dependency graph,
    ensuring that each modifier runs only after all its requirements
    have been satisfied.

    Attributes:
        options (Options):
            Configuration object controlling which modifiers are enabled.

        graph (dict[str, set[str]]):
            Dependency graph where each key is a modifier name and the
            associated set contains the names of modifiers it depends on.

        metadata (dict[str, Any]):
            Shared dictionary used to exchange information between
            modifiers during execution.
    """

    options: Options
    graph: dict[str, set[str]]
    metadata: dict[str, typing.Any]

    def __init__(self, options: Options) -> None:
        """Initialize the modifier engine.

        Builds the dependency graph by selecting only the modifiers
        that should run according to the provided options.

        Args:
            options (Options):
                Preprocessing configuration.
        """
        self.options = options
        self.graph = {m.name: set(m.requires) for m in MODIFIERS.values() if m.should_run(options)}
        self.metadata = {}

    def stage(self, node: ast.Module, stage_name: str | None = None) -> None:
        """Execute modifiers for a given stage.

        Modifiers are executed in dependency order. At each iteration,
        all modifiers with no remaining dependencies are selected and run.

        If a `stage_name` is provided, it is removed from the dependency
        lists, allowing staged execution relative to specific phases
        (e.g., before or after the OpenMP transformation).

        Args:
            node (ast.Module):
                AST module to transform.

            stage_name (str | None):
                Optional stage identifier used to release dependencies.
        """
        if stage_name is not None:
            for deps in self.graph.values():
                deps.discard(stage_name)

        while self.graph:
            ready: list[str] = [name for name, deps in self.graph.items() if not deps]

            if not ready:
                return

            for name in ready:
                MODIFIERS[name](self.options, self.metadata).run(node)
                del self.graph[name]

            for deps in self.graph.values():
                deps.symmetric_difference_update(ready)


class Modifier(ast.NodeTransformer):
    """Base class for AST modifiers.

    Subclasses of `Modifier` implement custom transformations applied
    to the Python AST. Each modifier can declare dependencies on other
    modifiers and define whether it is enabled by default.

    Modifiers are automatically registered via the `modifier` decorator
    and executed through the `ModifierEngine`.

    Attributes:
        name (ClassVar[str]):
            Unique identifier for the modifier.

        requires (ClassVar[list[str]]):
            List of modifier names that must be executed before this one.

        default (ClassVar[bool]):
            Whether the modifier is enabled by default if not explicitly
            configured.

        metadata (dict[str, Any]):
            Shared state dictionary for communication between modifiers.

        options (Options):
            Preprocessing configuration.
    """

    name: typing.ClassVar[str]
    requires: typing.ClassVar[list[str]]
    default: typing.ClassVar[bool]
    metadata: dict[str, typing.Any]
    options: Options

    def __init_subclass__(
        cls, /, name: str, requires: list[str] | None = None, default: bool = False, **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize subclass metadata.

        This method automatically assigns the modifier name,
        dependency list, and default activation state when a
        subclass is defined.

        Args:
            name (str):
                Unique name for the modifier.

            requires (list[str] | None):
                List of dependencies. Defaults to `["omp_transformer"]`.

            default (bool):
                Whether the modifier is enabled by default.

            **kwargs:
                Additional keyword arguments forwarded to the parent
                `__init_subclass__` implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.name = name
        cls.requires = requires or ["omp_transformer"]
        cls.default = default

    def __init__(self, options: Options, metadata: dict[str, typing.Any]) -> None:
        """Initialize the modifier instance.

        Args:
            options (Options):
                Preprocessing configuration.

            metadata (dict[str, Any]):
                Shared metadata dictionary.
        """
        self.options = options
        self.metadata = metadata

    @classmethod
    def should_run(cls, options: Options) -> bool:
        """Determine whether the modifier should be executed.

        The decision is based on the `Options.modifiers` configuration,
        falling back to the class default if not explicitly specified.

        Args:
            options (Options):
                Preprocessing configuration.

        Returns:
            bool:
                `True` if the modifier should be executed, `False` otherwise.
        """
        return options.modifiers.get(cls.name, cls.default)

    def run(self, node: ast.Module) -> None:
        """Execute the modifier on the given AST.

        This method applies the transformation by visiting the AST
        using the standard `ast.NodeTransformer` mechanism.

        Args:
            node (ast.Module):
                AST module to transform.
        """
        self.visit(node)
