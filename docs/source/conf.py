from dataclasses import asdict

from sphinxawesome_theme import ThemeOptions, __version__
from sphinxawesome_theme.postprocess import Icons

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Forest3D"
author = "Josafat-Mattias Burmeister"
copyright = f"2024, {author}."
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxawesome_theme.highlighting",
    "sphinx_design",
    "sphinx_sitemap",
]

autodoc_type_aliases = {"ArrayLike": "ArrayLike"}
default_role = "literal"
napoleon_custom_sections = [
    ("Parameters for the DBSCAN clustering of trunk points", "params_style"),
    ("Parameters for the construction and maximum filtering of the canopy height model", "params_style"),
    ("Parameters for the matching of trunk positions and crown top positions", "params_style"),
    ("Parameters for the Watershed segmentation", "params_style"),
    ("Parameters for the region growing segmentation", "params_style"),
]
napoleon_use_ivar = True
nitpicky = True
python_maximum_signature_line_length = 88

# Global substitutions for reStructuredText files
substitutions = [
    f".. |product| replace:: {project}",
    f".. |current| replace:: {release}",
]
rst_prolog = "\n".join(substitutions)

templates_path = ["apidoc_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

theme_options = ThemeOptions(
    show_prev_next=True,
    awesome_external_links=True,
    main_nav_links={"Docs": "/forest3d/index"},
    extra_header_link_icons={
        "repository on GitHub": {
            "link": "https://github.com/josafatburmeister/forest3d",
            "icon": (
                '<svg height="26px" style="margin-top:-2px;display:inline" '
                'viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 '
                "14.853 20.608 1.087.2 1.483-.47 1.483-1.047 "
                "0-.516-.019-1.881-.03-3.693-6.04 "
                "1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 "  # noqa
                "2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 "
                "1.803.197-1.403.759-2.36 "
                "1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 "
                "0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 "
                "1.822-.584 5.972 2.226 "
                "1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 "
                "4.147-2.81 5.967-2.226 "
                "5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 3.457 "
                "2.232 5.828 0 "
                "8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 4.021 0 "
                "2.904-.027 5.247-.027 "
                "5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 44.2 "
                '22.647c0-11.996-9.726-21.72-21.722-21.72" '
                'fill="currentColor"/></svg>'
            ),
        },
    },
)

html_title = project
html_theme = "sphinxawesome_theme"
html_theme_options = asdict(theme_options)
html_last_updated_fmt = ""
html_use_index = False
html_domain_indices = False
html_copy_source = False
html_logo = ""
html_favicon = ""
html_permalinks_icon = Icons.permalinks_icon
html_baseurl = "https://josafatburmeister.github.io/forest3d/"
html_extra_path = ["robots.txt", "_redirects"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["autoclass.css"]