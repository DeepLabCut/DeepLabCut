#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import numpy as np
import pytest
from deeplabcut.utils.visualization import safe_reshape


class TestSafeReshape:
    """Test cases for the safe_reshape function."""

    def test_valid_reshape(self):
        """Test that valid reshapes work correctly."""
        array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        result = safe_reshape(array, (2, 2, 2))
        expected = array.reshape((2, 2, 2))
        np.testing.assert_array_equal(result, expected)

    def test_valid_reshape_with_context(self):
        """Test that valid reshapes work correctly with context."""
        array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        result = safe_reshape(array, (2, 2, 3), "test context")
        expected = array.reshape((2, 2, 3))
        np.testing.assert_array_equal(result, expected)

    def test_invalid_reshape_too_small(self):
        """Test that reshape with too few elements raises proper error."""
        array = np.array([1, 2, 3, 4])  # 4 elements
        with pytest.raises(ValueError) as exc_info:
            safe_reshape(array, (2, 2, 2))  # Expects 8 elements
        
        error_msg = str(exc_info.value)
        assert "Cannot reshape array of size 4 into shape (2, 2, 2)" in error_msg
        assert "Expected 8 elements but got 4" in error_msg
        assert "missing data or inconsistent data structure" in error_msg

    def test_invalid_reshape_too_large(self):
        """Test that reshape with too many elements raises proper error."""
        array = np.array(range(20))  # 20 elements
        with pytest.raises(ValueError) as exc_info:
            safe_reshape(array, (2, 2, 2))  # Expects 8 elements
        
        error_msg = str(exc_info.value)
        assert "Cannot reshape array of size 20 into shape (2, 2, 2)" in error_msg
        assert "Expected 8 elements but got 20" in error_msg

    def test_invalid_reshape_with_context(self):
        """Test that reshape errors include context when provided."""
        array = np.array([1, 2, 3, 4])
        context = "ground truth data with 2 individuals and 2 bodyparts"
        
        with pytest.raises(ValueError) as exc_info:
            safe_reshape(array, (2, 2, 2), context)
        
        error_msg = str(exc_info.value)
        assert f"for {context}" in error_msg
        assert "Cannot reshape array of size 4 into shape (2, 2, 2)" in error_msg

    def test_empty_array(self):
        """Test that empty arrays are handled correctly."""
        array = np.array([])
        with pytest.raises(ValueError) as exc_info:
            safe_reshape(array, (1, 1), "empty array")
        
        error_msg = str(exc_info.value)
        assert "Cannot reshape array of size 0 into shape (1, 1)" in error_msg

    def test_original_error_scenario(self):
        """Test the specific error scenario from the issue."""
        # This reproduces the exact error: "cannot reshape array of size 8 into shape (2,4,2)"
        array = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # 8 elements
        with pytest.raises(ValueError) as exc_info:
            safe_reshape(array, (2, 4, 2))  # Expects 16 elements
        
        error_msg = str(exc_info.value)
        assert "Cannot reshape array of size 8 into shape (2, 4, 2)" in error_msg
        assert "Expected 16 elements but got 8" in error_msg

    def test_single_dimension(self):
        """Test reshaping to single dimension."""
        array = np.array([[1, 2], [3, 4]])
        result = safe_reshape(array, (4,))
        expected = array.reshape((4,))
        np.testing.assert_array_equal(result, expected)

    def test_multidimensional_input(self):
        """Test reshaping multidimensional input arrays."""
        array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2x2 = 8 elements
        result = safe_reshape(array, (4, 2))
        expected = array.reshape((4, 2))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("shape", [
        (2, 2, 2),
        (1, 8, 1),
        (4, 2),
        (8,),
    ])
    def test_various_valid_shapes(self, shape):
        """Test various valid reshape operations."""
        array = np.arange(8)
        result = safe_reshape(array, shape)
        expected = array.reshape(shape)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("array_size,shape", [
        (6, (2, 2, 2)),  # 6 elements into 8-element shape
        (10, (2, 2, 2)),  # 10 elements into 8-element shape 
        (4, (3, 3)),  # 4 elements into 9-element shape
        (0, (2, 2)),  # empty array into non-empty shape
    ])
    def test_various_invalid_shapes(self, array_size, shape):
        """Test various invalid reshape operations."""
        array = np.arange(array_size)
        with pytest.raises(ValueError) as exc_info:
            safe_reshape(array, shape)
        
        error_msg = str(exc_info.value)
        assert f"Cannot reshape array of size {array_size} into shape {shape}" in error_msg