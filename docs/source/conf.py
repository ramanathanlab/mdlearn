"""Documentation configuration file for Sphinx."""
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import annotations

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
import mdlearn  # noqa


# -- Project information -----------------------------------------------------

project = 'mdlearn'
author = (
    'Alexander Brace, '
    'Heng Ma, '
    'Austin Clyde, '
    'Debsindhu Bhowmik, '
    'Chakra Chennubhotla, '
    'Arvind Ramanathan'
)
now = datetime.datetime.now()
copyright = f'{now.year}, ' + author  # noqa A001

# The full version, including alpha/beta/rc tags
release = mdlearn.__version__
version = mdlearn.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinxcontrib.autodoc_pydantic',
]

# Autosummary settings
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Autodoc settings
# Need to figure these out. See https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_default_options
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []

html_context = {
    'conf_py_path': '/docs/source/',  # Path in the checkout to the docs root
}

# List of imports to mock when building the documentation.
autodoc_mock_imports = [
    'numpy',
    'h5py',
    'torch',
    'plotly',
    'pandas',
    'MDAnalysis',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: list[str] = []
