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
"""Tests that each public API function in deeplabcut/api/pose_estimation.py
routes to the correct downstream PyTorch implementation with the correct
parameters.

Each test:
  1. Patches ``_resolve_engine`` to return ``Engine.PYTORCH`` (avoids filesystem access).
  2. Patches the downstream PyTorch function.
  3. Calls the public API wrapper.
  4. Asserts the downstream was called exactly once with the expected arguments.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deeplabcut.core.deprecation import DLCDeprecationWarning
from deeplabcut.core.engine import Engine

_PYTORCH = Engine.PYTORCH
_RESOLVE = "deeplabcut.api._tf_routing._resolve_engine"


# ---------------------------------------------------------------------------
# train_network
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_train_network_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value=None)
    with patch("deeplabcut.pose_estimation_pytorch.apis.train_network", mock_impl):
        from deeplabcut.api.pose_estimation import train_network

        train_network("cfg.yaml", shuffle=2, trainingsetindex=1, epochs=50, batch_size=8)

    mock_impl.assert_called_once_with(
        "cfg.yaml",
        shuffle=2,
        trainingsetindex=1,
        modelprefix="",
        device=None,
        snapshot_path=None,
        detector_path=None,
        load_head_weights=True,
        batch_size=8,
        epochs=50,
        save_epochs=None,
        detector_batch_size=None,
        detector_epochs=None,
        detector_save_epochs=None,
        display_iters=None,
        max_snapshots_to_keep=None,
        pose_threshold=0.1,
        pytorch_cfg_updates=None,
    )


# ---------------------------------------------------------------------------
# evaluate_network
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_evaluate_network_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value=None)
    with patch("deeplabcut.pose_estimation_pytorch.apis.evaluate_network", mock_impl):
        from deeplabcut.api.pose_estimation import evaluate_network

        evaluate_network("cfg.yaml", shuffles=(1, 2), plotting=True)

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        shuffles=(1, 2),
        trainingsetindex=0,
        snapshotindex=None,
        device=None,
        plotting=True,
        show_errors=True,
        transform=None,
        snapshots_to_evaluate=None,
        comparison_bodyparts=None,
        per_keypoint_evaluation=False,
        modelprefix="",
        detector_snapshot_index=None,
        pcutoff=None,
    )


# ---------------------------------------------------------------------------
# analyze_videos
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_analyze_videos_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value="scorer")
    with patch("deeplabcut.pose_estimation_pytorch.apis.analyze_videos", mock_impl):
        from deeplabcut.api.pose_estimation import analyze_videos

        analyze_videos("cfg.yaml", ["video.mp4"], save_as_csv=True, destfolder="/out")

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        videos=["video.mp4"],
        video_extensions=None,
        shuffle=1,
        trainingsetindex=0,
        save_as_csv=True,
        in_random_order=False,
        snapshot_index=None,
        detector_snapshot_index=None,
        device=None,
        destfolder="/out",
        batch_size=None,
        detector_batch_size=None,
        dynamic=(False, 0.5, 10),
        ctd_conditions=None,
        ctd_tracking=False,
        top_down_dynamic=None,
        modelprefix="",
        use_shelve=False,
        robust_nframes=False,
        transform=None,
        auto_track=True,
        n_tracks=None,
        animal_names=None,
        calibrate=False,
        identity_only=False,
        overwrite=False,
        cropping=None,
        save_as_df=False,
        show_gpu_memory=False,
        inference_cfg=None,
    )


# ---------------------------------------------------------------------------
# analyze_images
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_analyze_images_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value={})
    with patch("deeplabcut.pose_estimation_pytorch.analyze_images", mock_impl):
        from deeplabcut.api.pose_estimation import analyze_images

        analyze_images("cfg.yaml", "images/", shuffle=3, save_as_csv=True)

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        images="images/",
        frame_type=None,
        output_dir=None,
        shuffle=3,
        trainingsetindex=0,
        snapshot_index=None,
        detector_snapshot_index=None,
        modelprefix="",
        device=None,
        max_individuals=None,
        save_as_csv=True,
        progress_bar=True,
        plotting=False,
        pcutoff=None,
        bbox_pcutoff=None,
        plot_skeleton=True,
        ctd_conditions=None,
    )


# ---------------------------------------------------------------------------
# convert_detections2tracklets
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_convert_detections2tracklets_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value=None)
    with patch(
        "deeplabcut.pose_estimation_pytorch.apis.convert_detections2tracklets",
        mock_impl,
    ):
        from deeplabcut.api.pose_estimation import convert_detections2tracklets

        convert_detections2tracklets(
            "cfg.yaml",
            ["video.mp4"],
            track_method="box",
            destfolder="/out",
        )

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        videos=["video.mp4"],
        video_extensions=None,
        shuffle=1,
        trainingsetindex=0,
        overwrite=False,
        destfolder="/out",
        ignore_bodyparts=None,
        inferencecfg=None,
        modelprefix="",
        identity_only=False,
        track_method="box",
        snapshot_index=None,
        detector_snapshot_index=None,
    )


# ---------------------------------------------------------------------------
# extract_maps
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_extract_maps_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value={})
    with patch("deeplabcut.pose_estimation_pytorch.extract_maps", mock_impl):
        from deeplabcut.api.pose_estimation import extract_maps

        extract_maps("cfg.yaml", shuffle=1, indices=[0, 5])

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        shuffle=1,
        trainingsetindex=0,
        device=None,
        rescale=False,
        indices=[0, 5],
        extract_paf=True,
        modelprefix="",
        snapshot_index=None,
        detector_snapshot_index=None,
    )


# ---------------------------------------------------------------------------
# export_model
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_export_model_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value=None)
    with patch("deeplabcut.pose_estimation_pytorch.apis.export.export_model", mock_impl):
        from deeplabcut.api.pose_estimation import export_model

        export_model("cfg.yaml", shuffle=2, overwrite=True)

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        shuffle=2,
        trainingsetindex=0,
        snapshotindex=None,
        detector_snapshot_index=None,
        iteration=None,
        overwrite=True,
        wipe_paths=False,
        without_detector=False,
        modelprefix=None,
    )


# ---------------------------------------------------------------------------
# return_train_network_path
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_return_train_network_path_routes_to_pytorch_impl(_):
    from pathlib import Path

    mock_impl = MagicMock(return_value=(Path("a"), Path("b"), Path("c")))
    with patch(
        "deeplabcut.pose_estimation_pytorch.apis.utils.return_train_network_path",
        mock_impl,
    ):
        from deeplabcut.api.pose_estimation import return_train_network_path

        return_train_network_path("cfg.yaml", shuffle=2, trainingsetindex=1, modelprefix="pfx")

    mock_impl.assert_called_once_with(
        "cfg.yaml",
        shuffle=2,
        trainingsetindex=1,
        modelprefix="pfx",
    )


# ---------------------------------------------------------------------------
# return_evaluate_network_data
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_return_evaluate_network_data_raises_not_implemented_for_pytorch(_):
    from deeplabcut.api.pose_estimation import return_evaluate_network_data

    with pytest.raises(NotImplementedError):
        return_evaluate_network_data("cfg.yaml")


# ---------------------------------------------------------------------------
# create_tracking_dataset
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_create_tracking_dataset_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value="scorer")
    with patch(
        "deeplabcut.pose_estimation_pytorch.apis.create_tracking_dataset",
        mock_impl,
    ):
        from deeplabcut.api.pose_estimation import create_tracking_dataset

        create_tracking_dataset("cfg.yaml", ["video.mp4"], track_method="box")

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        videos=["video.mp4"],
        track_method="box",
        video_extensions=None,
        shuffle=1,
        trainingsetindex=0,
        destfolder=None,
        batch_size=None,
        detector_batch_size=None,
        cropping=None,
        modelprefix="",
        robust_nframes=False,
        n_triplets=1000,
    )


# ---------------------------------------------------------------------------
# analyze_time_lapse_frames
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_analyze_time_lapse_frames_emits_deprecation_warning(_):
    with patch("deeplabcut.pose_estimation_pytorch.analyze_images", MagicMock()):
        from deeplabcut.api.pose_estimation import analyze_time_lapse_frames

        with pytest.warns(DLCDeprecationWarning, match="deeplabcut.analyze_images"):
            analyze_time_lapse_frames("cfg.yaml", "dir/")


@patch(_RESOLVE, return_value=_PYTORCH)
def test_analyze_time_lapse_frames_routes_to_analyze_images(_):
    mock_impl = MagicMock(return_value={})
    with patch("deeplabcut.pose_estimation_pytorch.analyze_images", mock_impl):
        from deeplabcut.api.pose_estimation import analyze_time_lapse_frames

        with pytest.warns(DLCDeprecationWarning):
            analyze_time_lapse_frames("cfg.yaml", "dir/", shuffle=2, save_as_csv=True)

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        images="dir/",
        output_dir="dir/",
        shuffle=2,
        trainingsetindex=0,
        device=None,
        save_as_csv=True,
        modelprefix="",
    )


# ---------------------------------------------------------------------------
# extract_save_all_maps
# ---------------------------------------------------------------------------


@patch(_RESOLVE, return_value=_PYTORCH)
def test_extract_save_all_maps_routes_to_pytorch_impl(_):
    mock_impl = MagicMock(return_value=None)
    with patch("deeplabcut.pose_estimation_pytorch.extract_save_all_maps", mock_impl):
        from deeplabcut.api.pose_estimation import extract_save_all_maps

        extract_save_all_maps("cfg.yaml", shuffle=1, comparison_bodyparts=["nose"])

    mock_impl.assert_called_once_with(
        config="cfg.yaml",
        shuffle=1,
        trainingsetindex=0,
        comparison_bodyparts=["nose"],
        extract_paf=True,
        all_paf_in_one=True,
        device=None,
        rescale=False,
        indices=None,
        modelprefix="",
        snapshot_index=None,
        detector_snapshot_index=None,
        dest_folder=None,
    )


# ---------------------------------------------------------------------------
# visualize_scoremaps
# ---------------------------------------------------------------------------


def test_visualize_scoremaps_delegates_to_core_visualization():
    mock_fn = MagicMock(return_value="result")
    with patch("deeplabcut.core.visualization.visualize_scoremaps", mock_fn):
        from deeplabcut.api.pose_estimation import visualize_scoremaps

        image = np.zeros((100, 100, 3))
        scmap = np.zeros((100, 100, 5))
        result = visualize_scoremaps(image, scmap)

    mock_fn.assert_called_once_with(image, scmap)
    assert result == "result"


# ---------------------------------------------------------------------------
# visualize_locrefs
# ---------------------------------------------------------------------------


def test_visualize_locrefs_delegates_to_core_visualization():
    mock_fn = MagicMock(return_value="result")
    with patch("deeplabcut.core.visualization.visualize_locrefs", mock_fn):
        from deeplabcut.api.pose_estimation import visualize_locrefs

        image = np.zeros((100, 100, 3))
        scmap = np.zeros((100, 100, 5))
        locref_x = np.zeros((100, 100, 5))
        locref_y = np.zeros((100, 100, 5))
        result = visualize_locrefs(image, scmap, locref_x, locref_y, step=3, zoom_width=10)

    mock_fn.assert_called_once_with(
        image=image,
        scmap=scmap,
        locref_x=locref_x,
        locref_y=locref_y,
        step=3,
        zoom_width=10,
    )
    assert result == "result"


# ---------------------------------------------------------------------------
# visualize_paf
# ---------------------------------------------------------------------------


def test_visualize_paf_delegates_to_core_visualization():
    mock_fn = MagicMock(return_value="result")
    with patch("deeplabcut.core.visualization.visualize_paf", mock_fn):
        from deeplabcut.api.pose_estimation import visualize_paf

        image = np.zeros((100, 100, 3))
        paf = np.zeros((100, 100, 10))
        result = visualize_paf(image, paf, step=3, colors=["red"])

    mock_fn.assert_called_once_with(
        image=image,
        paf=paf,
        step=3,
        colors=["red"],
    )
    assert result == "result"
