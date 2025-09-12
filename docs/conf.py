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
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser"
]

# templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


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

# Register the custom directives
def setup(app):
    custom_directives.setup(app)
