import ast
import contextlib
import re
import sys
import sysconfig
from pathlib import Path

from Cython.Build import cythonize
from Cython.Build.Dependencies import safe_makedirs_once
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

free_threading: bool = sysconfig.get_config_vars().get("Py_GIL_DISABLED") == 1
compiler_directives: dict = {"freethreading_compatible": free_threading, "annotation_typing": True}

__all__ = []

old_extensions: list[str] = [
    "omp4py/runtime/api_.py",
    "omp4py/runtime/basics/array.pyx",
    "omp4py/runtime/basics/atomic.pyx",
    "omp4py/runtime/basics/casting.py",
    "omp4py/runtime/basics/lock.pyx",
    "omp4py/runtime/basics/math.pyx",
    "omp4py/runtime/basics/threadlocal.pyx",
    "omp4py/runtime/basics/types.pyx",
    "omp4py/runtime/common/barrier.py",
    "omp4py/runtime/common/controlvars.py",
    "omp4py/runtime/common/enums.py",
    "omp4py/runtime/common/tasks.py",
    "omp4py/runtime/common/thread.py",
    "omp4py/runtime/common/threadshared.py",
    "omp4py/runtime/parallelism.py",
    "omp4py/runtime/synchronization.py",
    "omp4py/runtime/tasking.py",
    "omp4py/runtime/variables.py",
    "omp4py/runtime/workdistribution.py",
]

new_extensions: list[str] = [
    # api
    "omp4py/runtime/api/threadteam.py",
    # icvs
    "omp4py/runtime/icvs/defaults.py",
    "omp4py/runtime/icvs/icvs.py",
    "omp4py/runtime/icvs/places.py",
    # lowlevel
    "omp4py/runtime/lowlevel/atomic.pyx",
    "omp4py/runtime/lowlevel/mutex.pyx",
    "omp4py/runtime/lowlevel/numeric.pyx",
    "omp4py/runtime/lowlevel/threadlocal.py",
    # tasks
    "omp4py/runtime/tasks/barrier.py",
    "omp4py/runtime/tasks/context.py",
    "omp4py/runtime/tasks/parallelism.py",
    "omp4py/runtime/tasks/task.py",
    "omp4py/runtime/tasks/threadprivate.py",
]

extensions = old_extensions
#extensions = new_extensions

preproc_pattern: re.Pattern[str] = re.compile(
    r"\s*#\s*BEGIN_CYTHON_(?P<name>\w+)[^\n]*\n(?P<body>[\s\S]*?)\n\s*#\s*END_CYTHON_(?P=name)",
    flags=re.IGNORECASE,
)


def preprocessor(m: re.Match[str], file: str) -> str:
    result = ""
    match m.group("name").upper():
        case "IMPORTS":
            imports: list[tuple[int, int]] = []
            lines = m.group("body").splitlines(keepends=True)
            try:
                module: ast.Module = ast.parse(m.group("body"), file)
            except IndentationError:
                msg = f"{file} 'BEGIN_CYTHON_IMPORTS' cannot be placed inside indented lines"
                raise ValueError(msg) from None
            except SyntaxError:
                msg = f"{file} syntax error"
                raise ValueError(msg) from None

            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    imports.extend((alias.lineno - 1, alias.col_offset) for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append((node.lineno - 1, node.col_offset + 4))  # col_offset + from

            for module in imports[::-1]:
                lines[module[0]] = lines[module[0]][: module[1]] + " cython.cimports." + lines[module[0]][module[1]:]

            result = "\n" + ("".join(lines))
        case "IGNORE":
            result = ""
        case _:
            msg = f"{file} unrecognized cython preprocessor '{m.group('name')}'"
            raise ValueError(msg)
    old_lines = m.group(0).count("\n")
    new_lines = result.count("\n")
    return result + ("\n" * (old_lines - new_lines))


class Build(build_ext):
    def parse_imports(self, ext: Extension) -> None:
        with open(ext.sources[0]) as file:
            source: str = file.read()

        new_path = Path("build") / Path().joinpath(*ext.name.split(".")).with_suffix(Path(ext.sources[0]).suffix)
        safe_makedirs_once(new_path.parent)

        with open(new_path, "w") as file:
            file.write(preproc_pattern.sub(lambda m: preprocessor(m, ext.sources[0]), source))

    def build_extension(self, ext: Extension) -> None:
        self.parse_imports(ext)
        with contextlib.chdir("build"):
            ext.sources = [str(Path("build") / file) for file in cythonize(
                ext, language_level="3", annotate=self.editable_mode, compiler_directives=compiler_directives,
                include_path=[".."],
            )[0].sources]
        if self.compiler.compiler_type == "msvc":
            ext.extra_compile_args += ["/std:c11", "/experimental:c11atomics"]

        super().build_extension(ext)


def ext_name(path: str) -> str:
    return ".".join(Path(path).with_suffix("").parts)


if free_threading and sys.version_info >= (3, 13):
    setup(
        cmdclass={"build_ext": Build},
        ext_modules=[Extension(name=ext_name(path), sources=[path]) for path in extensions],
    )
else:
    setup()

if __name__ == "__main__":
    pass
