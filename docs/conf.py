# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "vectorose"
copyright = "2024, Benjamin Z. Rudski, Joseph Deering"
author = "Benjamin Z. Rudski, Joseph Deering"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.video",
    "sphinx_design",
    "sphinx_copybutton",
]
autoapi_dirs = ["../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

autoapi_keep_files = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "auto_examples/*.py",
    "auto_examples/*.ipynb",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "../resources/app_icons/icon_white_200x200.png"
html_favicon = "../resources/app_icons/icon_white_16x16.png"
html_title = "VectoRose Documentation"
html_short_title = "VectoRose Docs"

# -- Options for napoleon ----------------------------------------------------

# We want the return type to be presented inline
napoleon_use_rtype = False

# We want the types to be pre-processed
napoleon_preprocess_types = True
napoleon_custom_sections = [("Members", "params_style")]

# -- Options for intersphinx_mapping -----------------------------------------

# See https://gist.github.com/bskinn/0e164963428d4b51017cebdb6cda5209
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "trimesh": ("https://trimesh.org/", None),
    "pyvista": ("https://docs.pyvista.org/", None),
}

# -- Options for to-do -------------------------------------------------------
todo_include_todos = True

# -- Options for myst-nb -----------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "attrs_inline", "deflist"]
myst_dmath_double_inline = True
myst_heading_anchors = 4

# -- Options for sphinxcontrib-bibtex ----------------------------------------
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"
