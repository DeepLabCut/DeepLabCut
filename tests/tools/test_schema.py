from __future__ import annotations

import pytest
import pydantic


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
    with pytest.raises(pydantic.ValidationError):
        selector.SelectorResult.model_validate(data)
