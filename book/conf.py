# Minimal Sphinx configuration for ReadTheDocs compatibility
# This is used only to satisfy ReadTheDocs requirements
# The actual build is done by Jupyter Book 2.0

project = 'FESTIM tutorial'
copyright = '2025, The FESTIM community'
author = 'The FESTIM community'

extensions = []

html_theme = 'sphinx_rtd_theme'
html_static_path = []

# Don't actually build anything with Sphinx since Jupyter Book handles it
exclude_patterns = ['**']