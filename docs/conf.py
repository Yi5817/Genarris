"""Sphinx configuration for Genarris documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version

# -- Path setup ----------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "Genarris"
copyright = "2025, Noa Marom Group"
author = "Yi Yang, Rithwik Tom"

try:
    release = _get_version("gnrs")
except PackageNotFoundError:
    release = "3.1.1"
version = ".".join(release.split(".")[:2])

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST-Parser (Markdown support) -------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "substitution",
    "amsmath",
    "dollarmath",
]
myst_heading_anchors = 3

# -- Autodoc -------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autoclass_content = "both"

# Mock imports for heavy dependencies that aren't needed at doc-build time
autodoc_mock_imports = [
    "torch",
    "numpy",
    "numba",
    "mpi4py",
    "ase",
    "spglib",
    "pymatgen",
    "matplotlib",
    "pyyaml",
    "tqdm",
    "h5py",
    "dscribe",
    "sklearn",
    "scipy",
    "mace",
    "fairchem",
    "aimnet",
    "yaml",
    "networkx",
    "gnrs.cgenarris",
    "gnrs.cgenarris.src",
    "gnrs.cgenarris.src.pygenarris_mpi",
    "gnrs.cgenarris.src.rpack",
    "gnrs.cgenarris.src.rpack.rigid_press",
    "gnrs.cgenarris.src.rpack.rigid_press.rigid_press",
    "gnrs.cgenarris.src.rpack.rigid_press.ase_interface",
    "gnrs.cgenarris.src.rpack.setup",
    "gnrs.cgenarris.src.setup_mpi",
]

# -- Napoleon (Google-style docstrings) ----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

# -- Togglebutton --------------------------------------------------------------
togglebutton_hint = "Show source code"
togglebutton_hint_hide = "Hide source code"

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_title = "Genarris"
html_logo = "assets/images/Genarris_logo.png"
html_favicon = "assets/images/Genarris_logo.png"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#4051b5",
        "color-brand-content": "#4051b5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7c8aff",
        "color-brand-content": "#7c8aff",
    },
    "source_repository": "https://github.com/Yi5817/Genarris",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Yi5817/Genarris",
            "html": '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
            'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
            "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-"
            "2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
            "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-"
            ".89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 "
            '2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82'
            ".44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95"
            ".29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 "
            '0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>',
            "class": "",
        },
    ],
}
