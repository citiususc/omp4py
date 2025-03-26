import os
import re
import shutil
import typing
import sysconfig
from Cython.Build import cythonize
from setuptools import Distribution
from setuptools.command.build_ext import build_ext
from shutil import copytree, copyfile
from glob import glob

sep: str = os.sep


def copy(src: str, target: str, update: typing.Callable[[str], str]) -> str:
    with open(src, "r", encoding="utf-8") as f:
        txt: str = f.read()

    with open(target, "w", encoding="utf-8") as f:
        f.write(update(txt))
    return target


def import_update(ifaces: list[str]) -> typing.Callable[[str], str]:
    p: re.Pattern[str] = re.compile(r'(from|import)\s+(\S+)[^\n;]+')

    modules: dict[str, str] = {}

    iface: str
    for iface in ifaces:
        name: str = iface.replace(sep, '.')[:-4]
        if name.endswith('.__init__'):
            name = name[:-len('.__init__')]

        modules[name.replace('.cruntime', '.runtime')] = f'cython.cimports.{name}'

    def wrap(m: re.Match[str]) -> str:
        if m.group(2) == 'omp4py.runtime.basics.casting':
            if m.group(1) == 'import':
                return m.group(0).replace(m.group(2), 'cython')
            return 'from cython import cast'

        if m.group(2) in modules:
            head: str = ''
            if '#pyexport' in m.group(0):
                code: str = m.group(0).replace('.runtime', '.cruntime')
                head = f'exec("{code}");'
            return head + m.group().replace(m.group(2), modules[m.group(2)])
        return m.group(0)

    txt: str
    return lambda txt: p.sub(wrap, txt)


if __name__ == "__main__":
    freethreading: bool = sysconfig.get_config_vars().get("Py_GIL_DISABLED") == 1
    pyfiles: set[str] = set(glob(f'omp4py{sep}runtime{sep}**{sep}*.p*', recursive=True))
    cfiles: set[str] = set(glob(f'omp4py{sep}cruntime{sep}**{sep}*.p*', recursive=True))
    ifaces: list[str] = sorted([e for e in cfiles if e.endswith('.pxd')])

    update: typing.Callable[[str], str] = import_update(ifaces)

    os.makedirs(f'build', exist_ok=True)
    shutil.rmtree(f'build{sep}omp4py', ignore_errors=True)
    shutil.rmtree(f'build{sep}libs', ignore_errors=True)
    copytree('omp4py', f'build{sep}omp4py', dirs_exist_ok=True)

    extensions: list[str] = []
    pure_files: set[str] = {file for file in cfiles if file.endswith('.py') and file[:-2] + 'pxd' not in ifaces}
    file: str
    for file in ifaces:
        if file[:-3] + 'pyx' in cfiles:
            extensions.append(os.path.join('build', file[:-3] + 'pyx'))
        else:
            pyfile: str = file[:-3].replace(f'{sep}cruntime', f'{sep}runtime') + 'py'
            if os.path.basename(file) == '__init__.pxd':
                pure_files.add(copy(pyfile, os.path.join('build', file[:-3] + 'py'),
                                       lambda txt: txt.replace('.runtime', '.cruntime')))
                continue

            if pyfile not in pyfiles:
                continue
            extensions.append(copy(pyfile, os.path.join('build', file[:-3] + 'py'), update))

    with open(f'build{sep}omp4py{sep}cruntime{sep}__init__.py', 'a') as out:
        out.write('\nomp4py_compiled = True\n')

    compiler_directives: dict = {'freethreading_compatible': freethreading, 'annotation_typing': True}
    ext_modules = cythonize(extensions, language_level="3", annotate=True, compiler_directives=compiler_directives)


    class Build(build_ext):
        def build_extensions(self):
            if self.compiler.compiler_type == "msvc":
                for e in self.extensions:
                    e.extra_compile_args += ['/std:c11', '/experimental:c11atomics']
            build_ext.build_extensions(self)


    cmd = Build(Distribution({
        "name": "package",
        "ext_modules": ext_modules
    }))
    cmd.ensure_finalized()
    cmd.run()

    output: str
    for output in pure_files:
        target: str = os.path.join(f'build{sep}libs', os.path.relpath(output, 'build'))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        copyfile(output, target)

    for output in cmd.get_outputs():
        copyfile(output, os.path.join('build', os.path.relpath(output, cmd.build_lib)))
        copyfile(output, os.path.join(f'build{sep}libs', os.path.relpath(output, cmd.build_lib)))
