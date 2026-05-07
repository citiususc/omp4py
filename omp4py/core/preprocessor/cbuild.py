"""Cython-based compilation utilities for transformed `omp4py` modules.

This module provides the integration layer between the `omp4py` AST
transformation system and native extension compilation using Cython
and setuptools.

The `cythonize` function converts a transformed Python AST into a
temporary Python source file, compiles it as a native extension module,
and returns the path to the generated binary.

Compilation is fully configurable through `Options.compiler_args` and
automatically enables directives required for `omp4py` runtime support,
including free-threading compatibility and type inference.

Because compilation support is optional, required dependencies are
loaded lazily and validated at runtime through the optional extras
system.
"""

from __future__ import annotations

import ast
import pathlib
import shutil
import sys
import tempfile
import typing

from omp4py.core.imports.extra import require_extra

if typing.TYPE_CHECKING:
    from omp4py.core.options import Options


def cythonize(name: str, output: str, module: ast.Module, opt: Options) -> str:
    """Compile a transformed AST module into a native extension.

    The provided AST is converted into Python source code, compiled
    with Cython, and built using setuptools as a platform-specific
    extension module.

    Compilation behavior is controlled through `Options.compiler_args`,
    while additional directives required by omp4py are automatically
    injected into the Cython configuration.

    Temporary build files are created inside an isolated temporary
    directory and removed automatically after compilation.

    Args:
        name (str):
            Fully qualified module name used for the generated extension.

        output (str):
            Output directory where the compiled extension module will
            be written.

        module (ast.Module):
            Transformed Python AST module to compile.

        opt (Options):
            Compilation and preprocessing configuration.

    Returns:
        str:
            Absolute path to the generated extension module binary.

    Raises:
        ImportError:
            If required optional dependencies (`Cython` or `setuptools`)
            are not installed.
    """
    try:
        from Cython.Build import cythonize  # noqa: PLC0415
    except ImportError as _:
        require_extra("compile", "compile", "Cython")
    try:
        import setuptools  # noqa: PLC0415
    except ImportError as _:
        require_extra("compile", "compile", "setuptools")

    args = opt.compiler_args.copy()
    args.setdefault("quiet", not opt.debug)
    args.setdefault("annotate", opt.debug)
    args.setdefault("compiler_directives", {})
    args["compiler_directives"] |= {"freethreading_compatible": True}
    args["compiler_directives"].setdefault("annotation_typing", True)
    args["compiler_directives"].setdefault("infer_types", True)

    with tempfile.TemporaryDirectory() as tmp:
        source = (pathlib.Path(tmp) / f"{name}.py")
        source.write_text(ast.unparse(module))
        ext: list[setuptools.Extension] = cythonize(setuptools.Extension(name=name, sources=[source]), **args)

        if opt.debug:
            shutil.copy(source.with_suffix(".html"), output)

        dist = setuptools.Distribution(
            {
                "name": name,
                "ext_modules": ext,
            }        )

        cmd = dist.get_command_obj("build_ext")
        cmd.build_lib = output

        cmd.build_temp = tmp
        cmd.ensure_finalized()
        if sys.platform == "win32": # windows require extra flags
            org = cmd.build_extensions
            def patched()->None:
                if cmd.compiler.compiler_type == "msvc":
                    for ext in cmd.extensions:
                        ext.extra_compile_args += [
                            "/std:c11",
                            "/experimental:c11atomics",
                        ]
                org()

            cmd.build_extensions = patched

        cmd.run()

    return cmd.get_ext_fullpath(name)
