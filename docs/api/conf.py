import sys

sys.path.insert(0, "../../")

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.napoleon",
  "sphinx.ext.autosummary",
  "sphinx_ext_mystmd"
]
include_patterns = ["source/**"]
exclude_patterns = ["generated/**"]
master_doc = "source/index"
