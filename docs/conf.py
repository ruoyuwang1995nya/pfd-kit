from recommonmark.parser import CommonMarkParser
from importlib.metadata import version as get_version
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(1, os.path.abspath("../"))
import custom_directives

project = "pfd-kit"
copyright = "2024, Ruoyu Wang, Hongyu Wu"
author = "Ruoyu Wang, Hongyu Wu"
version = get_version("PFD-kit")
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_multiversion",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "recommonmark",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file parsers
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "rightsidebar": "true",
    "relbarbgcolor": "black"
}
html_static_path = ["_static", "images"]
# html_css_files = ['custom.css']

smv_tag_whitelist = r'^v*'  # Include tags starting with "v"
smv_branch_whitelist = r'^(main|release*)$'  # Include "main" and "release" branches

# Register the custom directives
def setup(app):
    custom_directives.setup(app)
