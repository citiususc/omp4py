import os
import re
import shutil
import typing
from Cython.Build import cythonize
from setuptools import Distribution
from setuptools.command.build_ext import build_ext
from shutil import copytree, copyfile
from glob import glob


def copy(src: str, target: str, update: typing.Callable[[str], str]) -> str:
    with open(src, "r", encoding="utf-8") as f:
        txt: str = f.read()

    with open(target, "w", encoding="utf-8") as f:
        f.write(update(txt))
    return target


def import_update(ifaces: list[str]) -> typing.Callable[[str], str]:
    p: re.Pattern[str] = re.compile(r'(from|import)\s+(\S+)\s+(import\s)?')

    names: dict[str, str] = {}
    iface: str
    for iface in ifaces:
        name: str = iface.replace('/', '.')[:-4]
        if name.endswith('.__init__'):
            name = name[:-len('.__init__')]

        names[name.replace('.cruntime', '.runtime')] = f'cython.cimports.{name}'

    def update(m: re.Match[str]) -> str:
        if m.group(2) not in names:
            return m.group(0)

        head: str = ''
        if m.group(2) == 'omp4py.runtime.basics.types':
            head = 'from cython import cast;'

        return head + m.group().replace(m.group(2), names[m.group(2)])

    return lambda s: p.sub(update, s)


def enable_cruntime(txt: str) -> str:
    return txt.replace('runtime', 'cruntime') + '\nomp4py_compiled=True\n'


if __name__ == "__main__":
    pyfiles: set[str] = set(glob('omp4py/runtime/**/*.p*', recursive=True))
    cfiles: set[str] = set(glob('omp4py/cruntime/**/*.p*', recursive=True))
    ifaces: list[str] = sorted([e for e in cfiles if e.endswith('.pxd')])

    update: typing.Callable[[str], str] = import_update(ifaces)

    os.makedirs('build', exist_ok=True)
    if os.path.exists('build/omp4py'):
        shutil.rmtree('build/omp4py')
    copytree('omp4py', 'build/omp4py', dirs_exist_ok=True)

    extensions: list[str] = []
    file: str
    for file in ifaces:
        if file.endswith('/__init__.pxd'):
            continue

        if file[:-3] + 'pyx' in cfiles:
            extensions.append(os.path.join('build', file[:-3] + 'pyx'))
        else:
            pyfile: str = file[:-3].replace('/cruntime', '/runtime') + 'py'
            if pyfile not in pyfiles:
                continue
            extensions.append(copy(pyfile, os.path.join('build', file[:-3] + 'py'), update))

    copy('build/omp4py/runtime/__init__.py', 'build/omp4py/cruntime/__init__.py', enable_cruntime)

    compiler_directives: dict = {'freethreading_compatible': True}
    ext_modules = cythonize(extensions, language_level="3", annotate=True, compiler_directives=compiler_directives)

    cmd = build_ext(Distribution({
        "name": "package",
        "ext_modules": ext_modules
    }))
    cmd.ensure_finalized()
    cmd.run()

    for output in cmd.get_outputs():
        copyfile(output, os.path.join('build', os.path.relpath(output, cmd.build_lib)))
