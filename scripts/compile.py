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
    p: re.Pattern[str] = re.compile(r'(from|import)\s+(\S+)[^\n;]+')

    modules: dict[str, str] = {}

    iface: str
    for iface in ifaces:
        name: str = iface.replace('/', '.')[:-4]
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
        if file[:-3] + 'pyx' in cfiles:
            extensions.append(os.path.join('build', file[:-3] + 'pyx'))
        else:
            pyfile: str = file[:-3].replace('/cruntime', '/runtime') + 'py'
            if file.endswith('/__init__.pxd'):
                copy(pyfile, os.path.join('build', file[:-3] + 'py'), lambda txt: txt.replace('.runtime', '.cruntime'))
                continue

            if pyfile not in pyfiles:
                continue
            extensions.append(copy(pyfile, os.path.join('build', file[:-3] + 'py'), update))

    with open('build/omp4py/cruntime/__init__.py', 'a') as out:
        out.write('\nomp4py_compiled = True\n')

    compiler_directives: dict = {'freethreading_compatible': True, 'annotation_typing': True}
    ext_modules = cythonize(extensions, language_level="3", annotate=True, compiler_directives=compiler_directives)

    cmd = build_ext(Distribution({
        "name": "package",
        "ext_modules": ext_modules
    }))
    cmd.ensure_finalized()
    cmd.run()

    for output in cmd.get_outputs():
        copyfile(output, os.path.join('build', os.path.relpath(output, cmd.build_lib)))
