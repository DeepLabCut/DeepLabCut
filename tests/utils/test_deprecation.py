#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import warnings

import pytest
from packaging.version import Version

from deeplabcut.utils.deprecation import (
    DLCDeprecationWarning,
    deprecated,
    renamed_parameter,
)

# ---------------------------------------------------------------------------
# @deprecated
# ---------------------------------------------------------------------------


def test_deprecated_emits_deprecation_warning():
    @deprecated()
    def old_fn():
        return 42

    with pytest.warns(DLCDeprecationWarning):
        result = old_fn()

    assert result == 42


def test_deprecated_warning_contains_function_name():
    @deprecated()
    def my_old_function():
        pass

    with pytest.warns(DLCDeprecationWarning, match="my_old_function"):
        my_old_function()


def test_deprecated_warning_contains_replacement():
    @deprecated(replacement="new_module.new_fn")
    def old_fn():
        pass

    with pytest.warns(DLCDeprecationWarning, match="new_module.new_fn"):
        old_fn()


def test_deprecated_warning_contains_since_and_removed_in():
    @deprecated(since="3.1", removed_in="4.0")
    def old_fn():
        pass

    with pytest.warns(DLCDeprecationWarning, match="3.1") as record:
        old_fn()

    assert "4.0" in str(record[0].message)


def test_deprecated_preserves_return_value_and_args():
    @deprecated()
    def add(a, b):
        return a + b

    with pytest.warns(DLCDeprecationWarning):
        assert add(2, 3) == 5


def test_deprecated_preserves_name_and_docstring():
    @deprecated(replacement="new_fn")
    def documented_fn():
        """Original docstring."""

    assert documented_fn.__name__ == "documented_fn"
    assert "Original docstring." in documented_fn.__doc__
    assert "Deprecated." in documented_fn.__doc__
    assert "new_fn" in documented_fn.__doc__


def test_deprecated_attaches_metadata():
    @deprecated(replacement="new_fn", since="3.1", removed_in="4.0")
    def old_fn():
        pass

    info = old_fn.__deprecated_info__
    assert info.kind == "callable"
    assert info.target.endswith("old_fn")
    assert info.replacement == "new_fn"
    assert info.since == Version("3.1")
    assert info.removed_in == Version("4.0")


def test_deprecated_invalid_since_raises():
    with pytest.raises(ValueError, match="Invalid version"):

        @deprecated(since="not-a-version")
        def old_fn():
            pass


def test_deprecated_invalid_removed_in_raises():
    with pytest.raises(ValueError, match="Invalid version"):

        @deprecated(removed_in="definitely-not-a-version")
        def old_fn():
            pass


def test_deprecated_removed_in_must_be_greater_than_since():
    with pytest.raises(ValueError, match="must be greater than"):

        @deprecated(since="4.0", removed_in="4.0")
        def old_fn():
            pass


# ---------------------------------------------------------------------------
# @renamed_parameter
# ---------------------------------------------------------------------------


def test_renamed_parameter_old_name_emits_warning():
    @renamed_parameter(old="in_random_order", new="shuffle")
    def fn(shuffle=False):
        return shuffle

    with pytest.warns(DLCDeprecationWarning):
        fn(in_random_order=True)


def test_renamed_parameter_old_name_is_forwarded():
    @renamed_parameter(old="in_random_order", new="shuffle")
    def fn(shuffle=False):
        return shuffle

    with pytest.warns(DLCDeprecationWarning):
        result = fn(in_random_order=True)

    assert result is True


def test_renamed_parameter_new_name_no_warning():
    @renamed_parameter(old="in_random_order", new="shuffle")
    def fn(shuffle=False):
        return shuffle

    # No warning should be emitted when using the current name.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DLCDeprecationWarning)
        result = fn(shuffle=True)

    assert result is True


def test_renamed_parameter_warning_contains_names():
    @renamed_parameter(old="videotype", new="extensions", since="3.2")
    def fn(extensions=None):
        return extensions

    with pytest.warns(DLCDeprecationWarning, match="videotype") as record:
        fn(videotype="mp4")

    message = str(record[0].message)
    assert "extensions" in message
    assert "3.2" in message


def test_renamed_parameter_preserves_name():
    @renamed_parameter(old="foo", new="bar")
    def my_fn(bar=None):
        """Docstring."""

    assert my_fn.__name__ == "my_fn"


def test_renamed_parameter_old_and_new_together_raise():
    @renamed_parameter(old="videotype", new="extensions")
    def fn(extensions=None):
        return extensions

    with pytest.raises(TypeError, match="both 'videotype' and 'extensions'"):
        fn(videotype="mp4", extensions="avi")


def test_renamed_parameter_attaches_metadata():
    @renamed_parameter(old="videotype", new="extensions", since="3.2")
    def fn(extensions=None):
        return extensions

    params = fn.__deprecated_params__
    assert len(params) == 1

    info = params[0]
    assert info.kind == "parameter"
    assert info.target.endswith("fn")
    assert info.old_parameter == "videotype"
    assert info.new_parameter == "extensions"
    assert info.since == Version("3.2")


def test_renamed_parameter_invalid_since_raises():
    with pytest.raises(ValueError, match="Invalid version"):

        @renamed_parameter(old="videotype", new="extensions", since="invalid-version")
        def fn(extensions=None):
            return extensions
