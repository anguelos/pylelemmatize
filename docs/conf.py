# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


version = '0.1.1'
release = '0.1.1'
import os
import sys
from datetime import datetime

# If your package is in ./src/your_package
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

project = 'PyLeLemmatize'
author = 'Anguelos Nicolaou'
copyright = f"{datetime.now():%Y}, {author}"

extensions = [
    "myst_parser",                   # Markdown support (.md via MyST)
    "sphinx.ext.autodoc",            # pull docstrings from code
    "sphinx.ext.autosummary",        # summary tables for API
    "sphinx.ext.napoleon",           # Google/NumPy style docstrings
    "sphinx_autodoc_typehints",      # show type hints in docs
    "sphinx.ext.viewcode",           # links to highlighted source
    "sphinx.ext.intersphinx",        # cross-link to stdlib/others
    "sphinx_copybutton",             # copy button on code blocks
    "sphinxcontrib.mermaid",         # diagrams (optional)
]

# Parse both .rst and .md
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",

}

autosummary_generate = True
autodoc_typehints = "description"    # move hints into the description
autodoc_member_order = "bysource"


#autodoc_default_options = {
#    "members": True,
#    "undoc-members": False,
#    "show-inheritance": True,
#}

autosummary_generate = True
autosummary_imported_members = False
templates_path = ["_templates"]

# Autodoc sensible defaults (you can still override per-directive in .rst)
autodoc_default_options = {
    "members": True,          # we'll explicitly list members
    "undoc-members": False,
    "inherited-members": False,  # To supress external classes' members add :inherited-members: on class directives manually
    "show-inheritance": True,
    "special-members": "__call__",
}


napoleon_google_docstring = True
napoleon_numpy_docstring = True


# Allow Markdown index & pages
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist", "linkify"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

#html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "publish.md"]
html_theme = "sphinx_rtd_theme"  #"alabaster", "furo", "sphinx_rtd_theme", etc.
