# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
from importlib.metadata import version as _version
from pathlib import Path

from tomllib import load as _load_toml

project = "elastic-ai.creator"
copyright = "2024, ies-ude (Intelligent Embedded System - University Duisburg-Essen)"
author = "es-ude"
release = _version("elasticai.creator")
version = ".".join(_version("elasticai.creator").split(".")[0:2])
html_title = "Elastic-AI.creator"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_book_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx_design",
    "autodoc2",
    "sphinxext.opengraph",
    "sphinxcontrib.plantuml",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.wavedrom",  # TODO: make wavedrom work to render waveforms
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# Default theme configuration
html_show_sourcelink = False


# Configure theme based on build type
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "content_footer_items": ["last-updated"],
    "repository_url": "https://github.com/es-ude/elastic-ai.creator/",
    "path_to_docs": "docs",
    "navigation_depth": 4,
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/es-ude/elastic-ai.creator/",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}


# only github flavored markdown
myst_gfm_only = False
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# allow mermaid usage like on github in markdown
myst_fence_as_directive = ["mermaid", "wavedrom"]

running_in_autobuild = os.getenv("SPHINX_AUTOBUILD", "NO") == "YES"


def find_builtin_plugins():
    plugin_path = Path(__file__).parent / "../elasticai/creator_plugins/"
    source_dir = Path(__file__).parent
    plugins = []
    for plugin in plugin_path.glob("*"):
        if (plugin / "__init__.py").exists():
            plugins.append(plugin)
    result = [
        {
            "path": str(plugin.relative_to(source_dir)),
            "module": f"elasticai.creator_plugins.{plugin.name}",
        }
        for plugin in plugins
    ]
    return result


find_builtin_plugins()

autodoc2_packages = [
    {
        "path": "../elasticai/creator",
        "module": "elasticai.creator",
    },
] + find_builtin_plugins()


def _build_regexes_for_all():
    """These regexes are considered to match for packages/modules that define an __all__ attribute.

    The __all__ attribute will be used to determine the public api of the package/module.
    This function is necessary because autodoc2 will try to build docs even for packages
    that do not define an __all__ attribute and pure namespace packages, however
    that will fail and doc build process aborts. Therefore we explicitly exclude these packages
    from the regexes.
    """
    project_root = Path(__file__).parents[1]

    def find_accessible_module_from_our_architecture():
        modules = []
        with (project_root / "tach.toml").open("r+b") as tach_config:
            architecture_description = _load_toml(tach_config)
        for module in architecture_description["modules"]:
            module_path = module["path"]
            if module_has_an_all_field(module_path):
                modules.append(module_path)
        return modules

    def module_has_an_all_field(module_path):
        file_path = project_root / "/".join(module_path.split("."))
        init_file = file_path / "__init__.py"
        if init_file.exists():
            with init_file.open("r") as f:
                for line in f:
                    if "__all__" in line:
                        return True
        return False

    modules = find_accessible_module_from_our_architecture()
    return modules


autodoc2_module_all_regexes = _build_regexes_for_all()
autodoc2_render_plugin = "myst"
autodoc2_hidden_objects = {"inherited", "private"}

myst_heading_anchors = 3
myst_heading_slug = True
