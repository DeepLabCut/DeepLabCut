#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for the annotated types in deeplabcut.core.config.validation."""

import pytest
from pydantic import ValidationError

from deeplabcut.core.config import DLCBaseConfig
from deeplabcut.core.config.validation import DefaultIfNone, Fraction


class ThresholdConfig(DLCBaseConfig):
    """Config exercising DefaultIfNone over a constrained type."""

    threshold: DefaultIfNone[Fraction] = 0.01
    name: DefaultIfNone[str] = "detector"


# ------------------------------------------------------------------
# DefaultIfNone
# ------------------------------------------------------------------


class TestDefaultIfNone:
    def test_none_falls_back_to_default(self):
        assert ThresholdConfig(threshold=None).threshold == 0.01

    def test_explicit_value_is_kept(self):
        assert ThresholdConfig(threshold=0.6).threshold == 0.6

    def test_falsy_value_is_kept(self):
        # A plain ``value or default`` would silently turn 0.0 into the default.
        assert ThresholdConfig(threshold=0.0).threshold == 0.0

    def test_applies_to_any_wrapped_type(self):
        assert ThresholdConfig(name=None).name == "detector"
        assert ThresholdConfig(name="ssdlite").name == "ssdlite"

    def test_wrapped_annotation_is_not_swallowed(self):
        # The generic alias must compose with Fraction, not replace it.
        with pytest.raises(ValidationError):
            ThresholdConfig(threshold=1.5)

    def test_none_is_normalized_on_assignment(self):
        # DLCBaseConfig sets validate_assignment=True, so the validator runs here too.
        cfg = ThresholdConfig(threshold=0.6)
        cfg.threshold = None
        assert cfg.threshold == 0.01

    def test_default_is_serialized_instead_of_none(self):
        # The null must not survive a load/save round trip.
        cfg = ThresholdConfig.from_dict({"threshold": None})
        assert cfg.to_dict()["threshold"] == 0.01
