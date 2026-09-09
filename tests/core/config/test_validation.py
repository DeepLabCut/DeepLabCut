#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for annotated types in deeplabcut.core.config.validation."""

import numpy as np
import pytest
from pydantic import ValidationError

from deeplabcut.core.config import DLCBaseConfig
from deeplabcut.core.config.validation import (
    BodypartPair,
    DefaultIfNone,
    Fraction,
    NDArrayInt,
    UniqueStrList,
    greater_than,
    less_than,
    unique_values,
    validate_crop_bounds,
)


class ThresholdConfig(DLCBaseConfig):
    """Config exercising DefaultIfNone over a constrained type."""

    threshold: DefaultIfNone[Fraction] = 0.01
    name: DefaultIfNone[str] = "detector"


class ListConfig(DLCBaseConfig):
    """Config exercising the list-based annotated types."""

    bodyparts: UniqueStrList = []
    skeleton: list[BodypartPair] = []
    indices: NDArrayInt = None


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


# ------------------------------------------------------------------
# greater_than / less_than
# ------------------------------------------------------------------


class TestComparisons:
    def test_greater_than_accepts_larger_value(self):
        greater_than(2.0, 1.0)

    def test_greater_than_rejects_smaller_value(self):
        with pytest.raises(ValueError, match="must be greater than"):
            greater_than(0.5, 1.0)

    def test_greater_than_is_strict(self):
        with pytest.raises(ValueError, match="must be greater than"):
            greater_than(1.0, 1.0)

    def test_less_than_accepts_smaller_value(self):
        less_than(1.0, 2.0, name="lower")

    def test_less_than_rejects_larger_value(self):
        with pytest.raises(ValueError, match="must be less than"):
            less_than(3.0, 2.0, name="lower")

    def test_less_than_is_strict(self):
        with pytest.raises(ValueError, match="must be less than"):
            less_than(2.0, 2.0, name="lower")

    def test_message_uses_names_when_given(self):
        with pytest.raises(ValueError) as excinfo:
            less_than(3, 2, name="x1", threshold_name="x2")
        message = str(excinfo.value)
        assert "x1 (3)" in message
        assert "x2 (2)" in message

    def test_message_uses_bare_values_when_unnamed(self):
        with pytest.raises(ValueError) as excinfo:
            greater_than(1, 2)
        assert str(excinfo.value) == "1 must be greater than 2"


# ------------------------------------------------------------------
# unique_values / UniqueStrList
# ------------------------------------------------------------------


class TestUniqueValues:
    def test_returns_input_when_unique(self):
        values = ["nose", "tail"]
        assert unique_values(values) == values

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError, match="must be unique"):
            unique_values(["nose", "nose"])

    def test_empty_list_is_allowed(self):
        assert unique_values([]) == []

    def test_enforced_through_annotated_type(self):
        assert ListConfig(bodyparts=["nose", "tail"]).bodyparts == ["nose", "tail"]
        with pytest.raises(ValidationError):
            ListConfig(bodyparts=["nose", "nose"])


# ------------------------------------------------------------------
# validate_crop_bounds
# ------------------------------------------------------------------


class TestValidateCropBounds:
    def test_all_none_is_allowed(self):
        validate_crop_bounds(x1=None, x2=None, y1=None, y2=None)

    def test_ordered_bounds_are_allowed(self):
        validate_crop_bounds(x1=0, x2=100, y1=0, y2=50)

    @pytest.mark.parametrize(
        "bounds",
        [
            {"x1": 0, "x2": None, "y1": None, "y2": None},
            {"x1": 0, "x2": 100, "y1": 0, "y2": None},
            {"x1": None, "x2": None, "y1": 0, "y2": 50},
        ],
        ids=["one-set", "three-set", "y-only"],
    )
    def test_partial_bounds_are_rejected(self, bounds):
        with pytest.raises(ValueError, match="all be set or all be omitted"):
            validate_crop_bounds(**bounds)

    def test_rejects_unordered_x(self):
        with pytest.raises(ValueError, match="x1.*must be less than.*x2"):
            validate_crop_bounds(x1=100, x2=0, y1=0, y2=50)

    def test_rejects_unordered_y(self):
        with pytest.raises(ValueError, match="y1.*must be less than.*y2"):
            validate_crop_bounds(x1=0, x2=100, y1=50, y2=0)

    def test_rejects_zero_width_crop(self):
        with pytest.raises(ValueError, match="must be less than"):
            validate_crop_bounds(x1=10, x2=10, y1=0, y2=50)


# ------------------------------------------------------------------
# BodypartPair
# ------------------------------------------------------------------


class TestBodypartPair:
    def test_accepts_a_pair(self):
        assert ListConfig(skeleton=[["nose", "tail"]]).skeleton == [["nose", "tail"]]

    @pytest.mark.parametrize("pair", [["nose"], ["nose", "tail", "ear"]], ids=["one", "three"])
    def test_rejects_wrong_length(self, pair):
        with pytest.raises(ValidationError, match="exactly two bodyparts"):
            ListConfig(skeleton=[pair])

    def test_rejects_duplicate_bodyparts(self):
        with pytest.raises(ValidationError, match="must be unique"):
            ListConfig(skeleton=[["nose", "nose"]])


# ------------------------------------------------------------------
# NDArrayInt
# ------------------------------------------------------------------


class TestNDArrayInt:
    def test_coerces_list_to_int_array(self):
        indices = ListConfig(indices=[1, 2, 3]).indices
        assert isinstance(indices, np.ndarray)
        assert indices.dtype == int
        np.testing.assert_array_equal(indices, [1, 2, 3])

    def test_coerces_nested_list(self):
        indices = ListConfig(indices=[[1, 2], [3, 4]]).indices
        assert indices.shape == (2, 2)

    def test_existing_array_is_passed_through_unchanged(self):
        # The coercion short-circuits on ndarray, so a float array keeps its dtype.
        array = np.array([1.5, 2.5])
        indices = ListConfig(indices=array).indices
        assert indices is array
        assert indices.dtype == np.float64
