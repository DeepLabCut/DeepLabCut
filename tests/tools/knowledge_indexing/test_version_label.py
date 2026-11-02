from __future__ import annotations

import argparse

import pytest

from tools.knowledge_indexing.__main__ import _version_label


@pytest.mark.parametrize("label", ["main", "3.0", "v3.0.1"])
def test_version_label_accepts(label: str):
    assert _version_label(label) == label


@pytest.mark.parametrize("label", ["", "../x", "has space", "-leading", "bad!"])
def test_version_label_rejects(label: str):
    with pytest.raises(argparse.ArgumentTypeError):
        _version_label(label)
