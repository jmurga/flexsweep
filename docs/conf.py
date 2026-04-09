import os
import sys

<<<<<<< HEAD
import os

os.environ.setdefault("FLEXSWEEP_BUILD_DOCS", "1")

import sys, pathlib
=======
os.environ.setdefault("FLEXSWEEP_BUILD_DOCS", "1")

import pathlib
import sys
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "Flex-sweep"
copyright = "2025, Jesús Murga-Moreno"
author = "Jesús Murga-Moreno, David Enard"

version = "2.0"
release = "2.0"


sys.path.insert(0, os.path.abspath(".."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
<<<<<<< HEAD
=======
    "sphinx.ext.napoleon",
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
]

<<<<<<< HEAD
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
=======
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

autodoc_mock_imports = ["tensorflow", "flexsweep.cnn"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "cms.rst"]
>>>>>>> ed421eb (pushing to 2.0. dann, recombination stratification normalization, custom stats, center/windows, outlier scan, partial cms, plotting)

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = True
html_static_path = []
