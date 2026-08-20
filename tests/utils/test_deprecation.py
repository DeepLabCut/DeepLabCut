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
import re
import warnings
from pathlib import Path

import pytest
from packaging.version import Version

import deeplabcut
from deeplabcut.core.deprecation import (
    DeprecationInfo,
    DeprecationRound,
    DeprecationRoundInfo,
    DLCDeprecationWarning,
    deprecated,
    renamed_parameter,
)

# Real rounds used by the tests below, so the suite exercises shipped data rather than
# fixtures. INIT_TF_DEPRECATION is the only round with a scheduled removal.
ROUND_WITH_REMOVAL = DeprecationRound.INIT_TF_DEPRECATION
ROUND_WITHOUT_REMOVAL = DeprecationRound.PARAMETER_CONSISTENCY

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
    round_info = ROUND_WITH_REMOVAL.value

    @deprecated(deprecation_round=ROUND_WITH_REMOVAL)
    def old_fn():
        pass

    with pytest.warns(DLCDeprecationWarning, match=str(round_info.since)) as record:
        old_fn()

    assert str(round_info.removed_in) in str(record[0].message)


def test_deprecated_warning_contains_pull_request_url():
    @deprecated(deprecation_round=ROUND_WITH_REMOVAL)
    def old_fn():
        pass

    with pytest.warns(DLCDeprecationWarning, match=ROUND_WITH_REMOVAL.value.url) as record:
        old_fn()

    assert "for more details" in str(record[0].message)


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
    round_info = ROUND_WITH_REMOVAL.value

    @deprecated(replacement="new_fn", deprecation_round=ROUND_WITH_REMOVAL)
    def old_fn():
        pass

    info = old_fn.__deprecated_info__
    assert info.kind == "callable"
    assert info.target.endswith("old_fn")
    assert info.replacement == "new_fn"
    assert info.deprecation_round is ROUND_WITH_REMOVAL
    assert info.since == round_info.since
    assert info.removed_in == round_info.removed_in
    assert info.url == round_info.url


def test_deprecated_without_round_has_no_version_metadata():
    @deprecated(replacement="new_fn")
    def old_fn():
        pass

    info = old_fn.__deprecated_info__
    assert info.deprecation_round is None
    assert info.since is None
    assert info.removed_in is None
    assert info.url is None


def test_deprecated_rejects_a_round_defined_outside_the_enum():
    """Rounds must be registered in ``DeprecationRound``, not built at the call site."""
    with pytest.raises(ValueError):

        @deprecated(deprecation_round=DeprecationRoundInfo(since="9.9", summary="ad-hoc"))
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
    @renamed_parameter(old="videotype", new="video_extensions", deprecation_round=ROUND_WITHOUT_REMOVAL)
    def fn(video_extensions=None):
        return video_extensions

    with pytest.warns(DLCDeprecationWarning, match="videotype") as record:
        fn(videotype="mp4")

    message = str(record[0].message)
    assert "video_extensions" in message
    assert str(ROUND_WITHOUT_REMOVAL.value.since) in message


def test_renamed_parameter_preserves_name():
    @renamed_parameter(old="foo", new="bar")
    def my_fn(bar=None):
        """Docstring."""

    assert my_fn.__name__ == "my_fn"


def test_renamed_parameter_old_and_new_together_raise():
    @renamed_parameter(old="videotype", new="video_extensions")
    def fn(video_extensions=None):
        return video_extensions

    with pytest.raises(TypeError, match="both 'videotype' and 'video_extensions'"):
        fn(videotype="mp4", video_extensions="avi")


def test_renamed_parameter_attaches_metadata():
    @renamed_parameter(old="videotype", new="video_extensions", deprecation_round=ROUND_WITHOUT_REMOVAL)
    def fn(video_extensions=None):
        return video_extensions

    params = fn.__deprecated_params__
    assert len(params) == 1

    info = params[0]
    assert info.kind == "parameter"
    assert info.target.endswith("fn")
    assert info.old_parameter == "videotype"
    assert info.new_parameter == "video_extensions"
    assert info.deprecation_round is ROUND_WITHOUT_REMOVAL
    assert info.since == ROUND_WITHOUT_REMOVAL.value.since


def test_renamed_parameter_new_not_in_signature_raises():
    with pytest.raises(ValueError, match="not a parameter"):

        @renamed_parameter(old="foo", new="nonexistent")
        def fn(bar=None):
            return bar


def test_new_not_in_signature_raises():
    """Applying a rename whose 'new' is not in the signature raises an error."""
    with pytest.raises(ValueError, match="not a parameter"):

        @renamed_parameter(old="old_name", new="new_name")
        def fn(not_new_name=None):
            return not_new_name


def test_old_still_in_signature_raises():
    """Applying a rename when the old name is still in the signature raises an error."""
    with pytest.raises(ValueError, match="still a parameter"):

        @renamed_parameter(old="old_name", new="new_name")
        def fn(old_name=None, new_name=None):
            return new_name


def test_renamed_parameter_chaining_raises():
    """Chaining renames A→B→C raises an error."""
    with pytest.raises(ValueError, match="chaining renames is not allowed"):

        @renamed_parameter(old="A", new="B")  # outer: A→B, but B is already deprecated to C
        @renamed_parameter(old="B", new="C")  # inner: B→C
        def fn(C=None):
            return C


def test_renamed_parameter_multiple_independent_renames():
    @renamed_parameter(old="batchsize", new="batch_size")
    @renamed_parameter(old="videotype", new="video_extensions")
    def fn(video_extensions=None, batch_size=None):
        return video_extensions, batch_size

    with pytest.warns(DLCDeprecationWarning):
        result = fn(videotype="mp4")
    assert result == ("mp4", None)

    with pytest.warns(DLCDeprecationWarning):
        result = fn(batchsize=4)
    assert result == (None, 4)


def test_renamed_parameter_positional_arg_unaffected():
    @renamed_parameter(old="in_random_order", new="shuffle")
    def fn(shuffle=False):
        return shuffle

    with warnings.catch_warnings():
        warnings.simplefilter("error", DLCDeprecationWarning)
        result = fn(True)

    assert result is True


def test_multiple_subsequent_renames_allowed():
    @renamed_parameter(old="oldestname", new="newest", deprecation_round=ROUND_WITHOUT_REMOVAL)
    @renamed_parameter(old="older_name", new="newest", deprecation_round=DeprecationRound.CONFIG_MODEL_MIGRATION)
    def fn(*, newest):
        return newest

    with pytest.warns(DLCDeprecationWarning):
        result = fn(oldestname=1)
    assert result == 1

    with pytest.warns(DLCDeprecationWarning):
        result = fn(older_name=2)
    assert result == 2


# ---------------------------------------------------------------------------
# DeprecationInfo
# ---------------------------------------------------------------------------


def test_callable_info_rejects_parameter_names():
    with pytest.raises(ValueError, match="cannot specify parameter names"):
        DeprecationInfo(kind="callable", target="func", old_parameter="old", new_parameter="new")


def test_parameter_info_requires_both_parameter_names():
    with pytest.raises(ValueError, match="require both"):
        DeprecationInfo(kind="parameter", target="func", old_parameter="old")


def test_parameter_info_rejects_replacement():
    with pytest.raises(ValueError, match="cannot specify 'replacement'"):
        DeprecationInfo(
            kind="parameter",
            target="func",
            replacement="not valid",
            old_parameter="old",
            new_parameter="new",
        )


# ---------------------------------------------------------------------------
# DeprecationRound / DeprecationRoundInfo
# ---------------------------------------------------------------------------


# An enum accepts any value, so a member assigned a bare version string would only fail
# later, when a marker reaches through .value.
def test_every_round_holds_a_round_info():
    assert list(DeprecationRound)

    for member in DeprecationRound:
        assert isinstance(member.value, DeprecationRoundInfo), member.name
        assert isinstance(member.value.since, Version), member.name


def test_round_url_points_at_the_pull_request():
    assert DeprecationRoundInfo(since="3.1", pull_request=1234).url.endswith("/pull/1234")
    assert DeprecationRoundInfo(since="3.1").url is None


def test_round_invalid_since_raises():
    with pytest.raises(ValueError, match="Invalid version"):
        DeprecationRoundInfo(since="not-a-version")


def test_round_invalid_removed_in_raises():
    with pytest.raises(ValueError, match="Invalid version"):
        DeprecationRoundInfo(since="3.1", removed_in="definitely-not-a-version")


def test_round_removed_in_must_be_greater_than_since():
    with pytest.raises(ValueError, match="must be greater than"):
        DeprecationRoundInfo(since="4.0", removed_in="4.0")


INLINE_ROUND = re.compile(r"deprecation_round\s*=\s*DeprecationRoundInfo\(")
ROUND_FIELD_ACCESS = re.compile(r"DeprecationRound\.\w+\.value\.")


def test_markers_reference_rounds_without_reaching_into_them():
    """Markers must name a ``DeprecationRound`` member, not build or unpack a round."""
    package_root = Path(deeplabcut.__file__).parent
    exempt = {package_root / "core" / "deprecation.py"}

    offenders = [
        f"{path.relative_to(package_root)}:{lineno}"
        for path in package_root.rglob("*.py")
        if path not in exempt
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if INLINE_ROUND.search(line) or ROUND_FIELD_ACCESS.search(line)
    ]

    assert not offenders, "Pass a DeprecationRound member instead: " + ", ".join(offenders)
