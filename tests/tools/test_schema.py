from __future__ import annotations

import pytest


def test_selector_result_forbids_extra_fields(selector):
    data = {
        "schema_version": 1,
        "plan": "fast",
        "pytest_paths": [],
        "functional_scripts": [],
        "reasons": [],
        "changed_files": [],
        "unexpected": "nope",
    }
    with pytest.raises(Exception):
        selector.SelectorResult.model_validate(data)
