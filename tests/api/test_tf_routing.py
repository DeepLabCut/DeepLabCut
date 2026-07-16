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
"""Tests for deeplabcut/api/_tf_routing.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deeplabcut.api import _tf_routing as tf_routing
from deeplabcut.core.deprecation import DLCDeprecationWarning
from deeplabcut.core.engine import Engine

# ---------------------------------------------------------------------------
# _normalize_gputouse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gputouse, expected",
    [
        (0, "cuda:0"),
        (1, "cuda:1"),
        ("cuda:0", "cuda:0"),
        ("gpu:1", "cuda:1"),
        ("cpu", "cpu"),
    ],
)
def test_normalize_gputouse(gputouse, expected):
    assert tf_routing._normalize_gputouse(gputouse) == expected


# ---------------------------------------------------------------------------
# _resolve_legacy_kwargs
# ---------------------------------------------------------------------------


def test_resolve_legacy_kwargs_renames_deprecated_parameter():
    kwargs = {"batchsize": 8, "config": "cfg.yaml"}

    with pytest.warns(DLCDeprecationWarning, match="batchsize"):
        result = tf_routing._resolve_legacy_kwargs(
            kwargs,
            renamed_params={"batchsize": "batch_size"},
            dropped_params=[],
        )

    assert result == {"batch_size": 8, "config": "cfg.yaml"}
    assert "batchsize" not in result


def test_resolve_legacy_kwargs_rename_emits_warning():
    with pytest.warns(DLCDeprecationWarning, match="batchsize"):
        tf_routing._resolve_legacy_kwargs(
            {"batchsize": 8},
            renamed_params={"batchsize": "batch_size"},
            dropped_params=[],
        )


def test_resolve_legacy_kwargs_rename_raises_when_both_names_given():
    with pytest.raises(TypeError, match="Cannot specify both 'batchsize'"):
        tf_routing._resolve_legacy_kwargs(
            {"batchsize": 8, "batch_size": 4},
            renamed_params={"batchsize": "batch_size"},
            dropped_params=[],
        )


def test_resolve_legacy_kwargs_drops_tensorflow_only_parameters():
    kwargs = {"rescale": True, "config": "cfg.yaml"}

    with pytest.warns(DLCDeprecationWarning, match="rescale"):
        result = tf_routing._resolve_legacy_kwargs(
            kwargs,
            renamed_params={},
            dropped_params=["rescale"],
        )

    assert result == {"config": "cfg.yaml"}


def test_resolve_legacy_kwargs_drop_emits_warning():
    with pytest.warns(DLCDeprecationWarning, match="rescale"):
        tf_routing._resolve_legacy_kwargs(
            {"rescale": True},
            renamed_params={},
            dropped_params=["rescale"],
        )


def test_resolve_legacy_kwargs_normalize_gputouse_int():
    with pytest.warns(DLCDeprecationWarning, match="gputouse"):
        result = tf_routing._resolve_legacy_kwargs(
            {"gputouse": 1},
            renamed_params={},
            dropped_params=[],
            normalize_gputouse=True,
        )

    assert result == {"device": "cuda:1"}
    assert "gputouse" not in result


def test_resolve_legacy_kwargs_normalize_gputouse_gpu_prefix():
    with pytest.warns(DLCDeprecationWarning, match="gputouse"):
        result = tf_routing._resolve_legacy_kwargs(
            {"gputouse": "gpu:2"},
            renamed_params={},
            dropped_params=[],
            normalize_gputouse=True,
        )

    assert result == {"device": "cuda:2"}


def test_resolve_legacy_kwargs_normalize_gputouse_raises_when_both_given():
    with pytest.raises(TypeError, match="Cannot specify both 'gputouse'"):
        tf_routing._resolve_legacy_kwargs(
            {"gputouse": 1, "device": "cuda:1"},
            renamed_params={},
            dropped_params=[],
            normalize_gputouse=True,
        )


# ---------------------------------------------------------------------------
# _resolve_engine
# ---------------------------------------------------------------------------


def test_resolve_engine_uses_explicit_engine():
    engine = tf_routing._resolve_engine({"config": "cfg.yaml", "engine": Engine.PYTORCH})
    assert engine == Engine.PYTORCH


@patch("deeplabcut.generate_training_dataset.metadata.get_shuffle_engine", return_value=Engine.PYTORCH)
@patch("deeplabcut.core.config.utils.read_config", return_value={"project_path": "/tmp"})
def test_resolve_engine_from_shuffle_metadata(mock_read_config, mock_get_shuffle_engine):
    engine = tf_routing._resolve_engine(
        {
            "config": "cfg.yaml",
            "shuffle": 2,
            "trainingsetindex": 1,
            "modelprefix": "prefix",
        }
    )

    assert engine == Engine.PYTORCH
    mock_read_config.assert_called_once_with("cfg.yaml")
    mock_get_shuffle_engine.assert_called_once_with(
        {"project_path": "/tmp"},
        trainingsetindex=1,
        shuffle=2,
        modelprefix="prefix",
    )


@patch("deeplabcut.generate_training_dataset.metadata.get_shuffle_engine", return_value=Engine.PYTORCH)
@patch("deeplabcut.core.config.utils.read_config", return_value={"project_path": "/tmp"})
def test_resolve_engine_defaults_to_shuffle_one(mock_read_config, mock_get_shuffle_engine):
    engine = tf_routing._resolve_engine({"config": "cfg.yaml"})

    assert engine == Engine.PYTORCH
    mock_get_shuffle_engine.assert_called_once_with(
        {"project_path": "/tmp"},
        trainingsetindex=0,
        shuffle=1,
        modelprefix="",
    )


@patch("deeplabcut.generate_training_dataset.metadata.get_shuffle_engine")
@patch("deeplabcut.core.config.utils.read_config", return_value={"project_path": "/tmp"})
def test_resolve_engine_from_shuffles_list(mock_read_config, mock_get_shuffle_engine):
    mock_get_shuffle_engine.side_effect = [Engine.PYTORCH, Engine.PYTORCH]

    engine = tf_routing._resolve_engine({"config": "cfg.yaml", "shuffles": [1, 2]})

    assert engine == Engine.PYTORCH
    assert mock_get_shuffle_engine.call_count == 2


@patch("deeplabcut.generate_training_dataset.metadata.get_shuffle_engine", return_value=Engine.TF)
@patch("deeplabcut.core.config.utils.read_config", return_value={"project_path": "/tmp"})
def test_resolve_engine_accepts_legacy_shuffles_kwarg(mock_read_config, mock_get_shuffle_engine):
    engine = tf_routing._resolve_engine({"config": "cfg.yaml", "Shuffles": [2, 3]})

    assert engine == Engine.TF
    assert mock_get_shuffle_engine.call_count == 2
    mock_get_shuffle_engine.assert_any_call(
        {"project_path": "/tmp"},
        trainingsetindex=0,
        shuffle=2,
        modelprefix="",
    )
    mock_get_shuffle_engine.assert_any_call(
        {"project_path": "/tmp"},
        trainingsetindex=0,
        shuffle=3,
        modelprefix="",
    )


def test_resolve_engine_rejects_both_shuffles_and_shuffles():
    with pytest.raises(TypeError, match="Cannot specify both 'Shuffles'"):
        tf_routing._resolve_engine({"config": "cfg.yaml", "shuffles": [1], "Shuffles": [2]})


@patch("deeplabcut.generate_training_dataset.metadata.get_shuffle_engine")
@patch("deeplabcut.core.config.utils.read_config", return_value={"project_path": "/tmp"})
def test_resolve_engine_raises_when_shuffles_have_different_engines(mock_read_config, mock_get_shuffle_engine):
    mock_get_shuffle_engine.side_effect = [Engine.PYTORCH, Engine.TF]

    with pytest.raises(ValueError, match="All shuffles must have the same engine"):
        tf_routing._resolve_engine({"config": "cfg.yaml", "shuffles": [1, 2]})


@patch("deeplabcut.core.config.utils.read_config", return_value={"project_path": "/tmp"})
def test_resolve_engine_reads_config_from_kwargs(mock_read_config):
    with patch("deeplabcut.generate_training_dataset.metadata.get_shuffle_engine", return_value=Engine.PYTORCH):
        tf_routing._resolve_engine({"config": "other.yaml", "engine": Engine.PYTORCH})

    mock_read_config.assert_not_called()


# ---------------------------------------------------------------------------
# warn_deprecated_tensorflow
# ---------------------------------------------------------------------------


def test_warn_deprecated_tensorflow_emits_deprecation_warning():
    with pytest.warns(DLCDeprecationWarning, match="TensorFlow support is deprecated"):
        tf_routing.warn_deprecated_tensorflow()


# ---------------------------------------------------------------------------
# with_tensorflow_fallback
# ---------------------------------------------------------------------------


def test_with_tensorflow_fallback_routes_to_pytorch_fn():
    pytorch_fn = MagicMock(return_value="pytorch")

    @tf_routing.with_tensorflow_fallback(renamed_params={"batchsize": "batch_size"})
    def canonical_fn(*args, **kwargs):
        return pytorch_fn(*args, **kwargs)

    with (
        patch("deeplabcut.api._tf_routing._resolve_engine", return_value=Engine.PYTORCH),
        pytest.warns(DLCDeprecationWarning, match="batchsize"),
    ):
        result = canonical_fn("cfg.yaml", shuffle=1, batchsize=8)

    assert result == "pytorch"
    pytorch_fn.assert_called_once_with("cfg.yaml", shuffle=1, batch_size=8)


def test_with_tensorflow_fallback_routes_to_tensorflow_impl():
    tf_impl = MagicMock(return_value="tensorflow")

    @tf_routing.with_tensorflow_fallback
    def canonical_fn(*args, **kwargs):
        return "pytorch"

    with (
        patch("deeplabcut.api._tf_routing._resolve_engine", return_value=Engine.TF),
        patch("deeplabcut.api._tf_routing._get_tensorflow_impl", return_value=tf_impl),
        pytest.warns(DLCDeprecationWarning, match="TensorFlow support is deprecated"),
    ):
        result = canonical_fn("cfg.yaml", shuffle=1)

    assert result == "tensorflow"
    tf_impl.assert_called_once_with("cfg.yaml", shuffle=1)


def test_with_tensorflow_fallback_uses_custom_tensorflow_name():
    tf_impl = MagicMock(return_value="tensorflow")

    @tf_routing.with_tensorflow_fallback(tensorflow_name="legacy_fn_name")
    def canonical_fn(*args, **kwargs):
        return "pytorch"

    with (
        patch("deeplabcut.api._tf_routing._resolve_engine", return_value=Engine.TF),
        patch("deeplabcut.api._tf_routing._get_tensorflow_impl", return_value=tf_impl) as mock_get_impl,
        pytest.warns(DLCDeprecationWarning),
    ):
        canonical_fn("cfg.yaml")

    mock_get_impl.assert_called_once_with("legacy_fn_name", module=None)


def test_with_tensorflow_fallback_without_parentheses():
    pytorch_fn = MagicMock(return_value="pytorch")

    @tf_routing.with_tensorflow_fallback
    def bare_decorator_fn(*args, **kwargs):
        return pytorch_fn(*args, **kwargs)

    with patch("deeplabcut.api._tf_routing._resolve_engine", return_value=Engine.PYTORCH):
        result = bare_decorator_fn("cfg.yaml")

    assert result == "pytorch"


def test_with_tensorflow_fallback_drops_tensorflow_only_params_for_pytorch():
    pytorch_fn = MagicMock(return_value="pytorch")

    @tf_routing.with_tensorflow_fallback(dropped_params=["rescale"])
    def canonical_fn(*args, **kwargs):
        return pytorch_fn(*args, **kwargs)

    with (
        patch("deeplabcut.api._tf_routing._resolve_engine", return_value=Engine.PYTORCH),
        pytest.warns(DLCDeprecationWarning, match="rescale"),
    ):
        canonical_fn("cfg.yaml", rescale=True)

    pytorch_fn.assert_called_once_with("cfg.yaml")


def test_with_tensorflow_fallback_normalizes_gputouse_for_pytorch():
    pytorch_fn = MagicMock(return_value="pytorch")

    @tf_routing.with_tensorflow_fallback(normalize_gputouse=True)
    def canonical_fn(*args, **kwargs):
        return pytorch_fn(*args, **kwargs)

    with (
        patch("deeplabcut.api._tf_routing._resolve_engine", return_value=Engine.PYTORCH),
        pytest.warns(DLCDeprecationWarning, match="gputouse"),
    ):
        canonical_fn("cfg.yaml", gputouse=1)

    pytorch_fn.assert_called_once_with("cfg.yaml", device="cuda:1")


def test_with_tensorflow_fallback_strips_engine_before_calling_impl():
    tf_impl = MagicMock(return_value="tensorflow")

    @tf_routing.with_tensorflow_fallback
    def canonical_fn(*args, **kwargs):
        return "pytorch"

    with (
        patch("deeplabcut.api._tf_routing._resolve_engine", return_value=Engine.TF),
        patch("deeplabcut.api._tf_routing._get_tensorflow_impl", return_value=tf_impl),
        pytest.warns(DLCDeprecationWarning),
    ):
        canonical_fn("cfg.yaml", engine=Engine.TF)

    tf_impl.assert_called_once_with("cfg.yaml")


# ---------------------------------------------------------------------------
# with_tensorflow_fallback — custom `when` predicate
# ---------------------------------------------------------------------------


def test_with_tensorflow_fallback_when_routes_to_tf_if_predicate_true():
    tf_impl = MagicMock(return_value="tensorflow")

    @tf_routing.with_tensorflow_fallback(
        when=lambda *a, **kw: kw.get("model_name") == "dlcrnet",
        tensorflow_module="deeplabcut.tensorflow_compat.superanimal_inference",
        tensorflow_name="video_inference_superanimal_tf",
    )
    def canonical_fn(*args, **kwargs):
        return "pytorch"

    with (
        patch.object(tf_routing, "_get_tensorflow_impl", return_value=tf_impl) as mock_get_impl,
        pytest.warns(DLCDeprecationWarning),
    ):
        result = canonical_fn("some_path", model_name="dlcrnet")

    assert result == "tensorflow"
    mock_get_impl.assert_called_once_with(
        "video_inference_superanimal_tf",
        module="deeplabcut.tensorflow_compat.superanimal_inference",
    )


def test_with_tensorflow_fallback_when_routes_to_pt_if_predicate_false():
    pytorch_fn = MagicMock(return_value="pytorch")

    @tf_routing.with_tensorflow_fallback(
        when=lambda *a, **kw: kw.get("model_name") == "dlcrnet",
        dropped_params=["scale_list"],
    )
    def canonical_fn(*args, **kwargs):
        return pytorch_fn(*args, **kwargs)

    with pytest.warns(DLCDeprecationWarning, match="scale_list"):
        result = canonical_fn("some_path", model_name="hrnet_w32", scale_list=[200, 300])

    assert result == "pytorch"
    pytorch_fn.assert_called_once_with("some_path", model_name="hrnet_w32")


def test_with_tensorflow_fallback_when_receives_args_and_kwargs():
    tf_impl = MagicMock(return_value="tensorflow")

    captured_args = []
    captured_kwargs = {}

    def predicate(*a, **kw):
        captured_args.append(a)
        captured_kwargs.update(kw)
        return kw.get("force_tf", False)

    @tf_routing.with_tensorflow_fallback(when=predicate)
    def canonical_fn(*args, **kwargs):
        return "pytorch"

    with (
        patch.object(tf_routing, "_get_tensorflow_impl", return_value=tf_impl),
        pytest.warns(DLCDeprecationWarning),
    ):
        canonical_fn("arg1", "arg2", force_tf=True, extra="val")

    assert captured_args == [("arg1", "arg2")]
    assert captured_kwargs == {"force_tf": True, "extra": "val"}
    tf_impl.assert_called_once()


def test_with_tensorflow_fallback_when_takes_precedence_over_engine():
    tf_impl = MagicMock(return_value="tensorflow")

    @tf_routing.with_tensorflow_fallback(
        when=lambda *a, **kw: kw.get("model_name") == "dlcrnet",
    )
    def canonical_fn(*args, **kwargs):
        return "pytorch"

    # Even if _resolve_engine would return Engine.PYTORCH, the when predicate
    # should not call _resolve_engine at all when when is provided.
    with (
        patch("deeplabcut.api._tf_routing._resolve_engine") as mock_resolve,
        patch.object(tf_routing, "_get_tensorflow_impl", return_value=tf_impl),
        pytest.warns(DLCDeprecationWarning),
    ):
        result = canonical_fn("path", model_name="dlcrnet")

    assert result == "tensorflow"
    mock_resolve.assert_not_called()


def test_with_tensorflow_fallback_when_without_tensorflow_module_defaults():
    tf_impl = MagicMock(return_value="tensorflow")

    @tf_routing.with_tensorflow_fallback(
        when=lambda *a, **kw: True,
    )
    def canonical_fn(*args, **kwargs):
        return "pytorch"

    with (
        patch.object(tf_routing, "_get_tensorflow_impl", return_value=tf_impl) as mock_get_impl,
        pytest.warns(DLCDeprecationWarning),
    ):
        canonical_fn()

    mock_get_impl.assert_called_once_with("canonical_fn", module=None)
