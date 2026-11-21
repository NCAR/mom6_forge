# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../../mom6_forge/"))


# -- Project information -----------------------------------------------------

project = "mom6_forge"
copyright = "2026, NCAR"
author = "NCAR"

# The full version, including alpha/beta/rc tags
release = "0.3.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# import sphinx_rtd_theme
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "myst_parser",
]

# Never re-execute notebooks during docs build — use existing cell outputs. The demo
# notebook (7_demo_editors) opts in through its metadata, so that its widgets are
# drawn from fresh widget state.
nbsphinx_execute = "never"

# Hand-written pages are MyST Markdown; sphinx-apidoc still emits .rst.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'bizstyle'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images"]

# docstrings ----------------------
napoleon_numpy_docstring = True
autoclass_content = "both"


# -- Live widgets (JupyterLite) ----------------------------------------------
# The widgets page runs TopoEditor and VGridCreator in the browser: each {replite}
# block opens a JupyterLite console with a Pyodide kernel that installs the
# mom6_forge wheel from <repo>/dist (docs.yml builds it). Without jupyterlite-sphinx
# or the wheel, the docs still build and each block becomes a short note.
_wheels = sorted((Path(__file__).parents[2] / "dist").glob("mom6_forge-*.whl"))
try:
    import jupyterlite_sphinx  # noqa: F401
except ImportError:
    _wheels = []

if _wheels:
    extensions.append("jupyterlite_sphinx")
    jupyterlite_bind_ipynb_suffix = False  # nbsphinx renders the notebooks
    jupyterlite_build_command_options = {"piplite-wheels": str(_wheels[-1])}
else:
    from docutils import nodes
    from docutils.parsers.rst import Directive, directives

    class _NoReplite(Directive):
        has_content = True
        option_spec = dict.fromkeys(
            ["kernel", "height", "prompt"], directives.unchanged
        )

        def run(self):
            text = "The live demo is left out of this build (see Contributing)."
            return [nodes.note("", nodes.paragraph(text=text))]

    def setup(app):
        app.add_directive("replite", _NoReplite)
