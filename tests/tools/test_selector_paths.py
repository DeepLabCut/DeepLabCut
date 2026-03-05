from __future__ import annotations

import pytest


def test_normalize_relpath_basic(selector):
    assert selector._normalize_relpath("docs/index.md") == "docs/index.md"
    assert selector._normalize_relpath("docs\\index.md") == "docs/index.md"


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "   ",  # whitespace
        "/etc/passwd",  # absolute unix
        "C:/Windows/x",  # absolute windows
        "../secret.txt",  # traversal
        "docs/../../x",  # traversal inside
        "a\x00b",  # NUL
    ],
)
def test_normalize_relpath_rejects_bad(selector, bad):
    with pytest.raises(ValueError):
        selector._normalize_relpath(bad)
