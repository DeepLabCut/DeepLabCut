from __future__ import annotations

from pathlib import Path

import yaml

from tools.knowledge_indexing.api_index import API_ROOT_URI, EXCLUDED_MODULES


def test_constants_match_the_dev_docs_config():
    # These mirror plugins.api-autonav in dev-docs/mkdocs.yml. Drift there
    # silently repoints every generated api url at a page that does not exist.
    config = yaml.safe_load((Path(__file__).resolve().parents[3] / "dev-docs/mkdocs.yml").read_text(encoding="utf-8"))
    autonav = next(p["api-autonav"] for p in config["plugins"] if isinstance(p, dict) and "api-autonav" in p)

    assert autonav["api_root_uri"] == API_ROOT_URI
    assert tuple(autonav["exclude"]) == EXCLUDED_MODULES
