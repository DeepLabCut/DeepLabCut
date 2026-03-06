from __future__ import annotations

import pytest


def test_validate_sha_accepts(selector):
    assert selector._validate_sha("x", "abc1234") == "abc1234"
    assert selector._validate_sha("x", "a" * 40) == "a" * 40


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "notasha",  # non-hex
        "123",  # too short
        "g" * 40,  # non-hex
        " " * 8,  # whitespace
    ],
)
def test_validate_sha_rejects(selector, bad):
    with pytest.raises(ValueError):
        selector._validate_sha("x", bad)
