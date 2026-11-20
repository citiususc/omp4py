import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Document the pure Python runtime so autodoc never needs the compiled extensions.
os.environ.setdefault("OMP4PY_PURE", "1")

for line in (ROOT / "omp4py" / "__init__.py").read_text().split("\n"):
    if line.startswith("__version__"):
        exec(line)
        break

project = "OMP4Py"
author = "César Pomar, Juan C. Pichel"
copyright = "2026"
release = globals()["__version__"]
version = re.match(r"^[0-9.]+", release).group()

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "shibuya"
html_static_path = ["_static"]
html_title = "OMP4Py documentation"
html_show_sourcelink = True
html_favicon = "_static/favicon.ico"
html_logo = "_static/logo.png"

mermaid_version = "11.4.1"
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
# "ref.python": ambiguous short names in autodoc type hints (e.g. Context, Task) exist in several modules.
suppress_warnings = ["myst.header", "ref.python"]
exclude_patterns += ["examples/**"]

