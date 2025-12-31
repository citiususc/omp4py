import os
import re
import sys
import sysconfig
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Build.Dependencies import safe_makedirs_once
from setuptools.command.build_ext import build_ext

free_threading: bool = sysconfig.get_config_vars().get("Py_GIL_DISABLED") == 1
compiler_directives: dict = {'freethreading_compatible': free_threading,
                             'annotation_typing': True}

extensions: list[str] = [
    "omp4py/runtime/api.py",
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
    "omp4py/runtime/workdistribution.py"
]

pattern: re.Pattern[str] = re.compile(r"(# BEGIN_CYTHON_IMPORTS.*?# END_CYTHON_IMPORTS)", flags=re.DOTALL)


class Build(build_ext):

    def parse_imports(self, ext):
        with open(ext.sources[0], "r") as file:
            source: str = file.read()

        new_path: str = os.path.join("build", ext.name.replace(".", "/") + os.path.splitext(ext.sources[0])[1])
        safe_makedirs_once(os.path.dirname(new_path))

        def f(m: re.Match[str]) -> str:
            return m.group(0).replace("omp4py.", 'cython.cimports.omp4py.')

        with open(new_path, "w") as file:
            file.write(pattern.sub(f, source, count=1))
        ext.sources[0] = new_path
        return True

    def build_extension(self, ext) -> None:
        self.parse_imports(ext)
        ext.sources = cythonize(ext, language_level="3", annotate=self.editable_mode,
                                compiler_directives=compiler_directives)[0].sources
        if self.compiler.compiler_type == "msvc":
            ext.extra_compile_args += ['/std:c11', '/experimental:c11atomics']

        super().build_extension(ext)


def ext_name(path: str) -> str:
    return os.path.splitext(path)[0].replace('/', '.')


if free_threading and sys.version_info[1] >= 13:
    setup(
        cmdclass={'build_ext': Build},
        ext_modules=[Extension(name=ext_name(path), sources=[path]) for path in extensions],
    )
else:
    setup()
