# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'py-initseq'
copyright = '2026, Enrico Foglia'
author = 'Enrico Foglia'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
 'sphinx.ext.autodoc',
 'sphinx.ext.autosummary',
 'sphinx.ext.napoleon',
 'autoapi.extension',
 ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for AutoAPI -----------------------------------------------------
autoapi_dirs = [str(PROJECT_ROOT/'src')]
 

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
