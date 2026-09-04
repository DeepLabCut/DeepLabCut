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
# NOTE: @C-Achard 2026-09-04 : [CLEANUP-COLLECTEDDATA-FIXTURES] evaluation_dataframe_factory below duplicates the
# CollectedData frame builder that at least six test modules each roll their own version of,
# here with likelihood columns and its own index level names. Reuse target: the
# collected_data fixture in tests/utils/test_skeleton.py, once it is promoted to a shared conftest.
import logging
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from deeplabcut.utils import visualization


@pytest.fixture(autouse=True)
def cleanup_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def evaluation_dataframe_factory():
    """Build combined ground-truth and prediction evaluation data."""

    def create(
        *,
        scorer="human_scorer",
        model_name="model",
        gt_individuals=("animal",),
        pred_individuals=("individual0",),
        bodyparts=("nose", "tail"),
        gt_bodyparts=None,
        pred_bodyparts=None,
        unique_bodyparts=(),
        image="img001.png",
    ):
        if gt_bodyparts is None:
            gt_bodyparts = bodyparts
        if pred_bodyparts is None:
            pred_bodyparts = bodyparts

        gt_columns = pd.MultiIndex.from_product(
            [
                [scorer],
                gt_individuals,
                gt_bodyparts,
                ["x", "y"],
            ],
            names=[
                "scorer",
                "individuals",
                "bodyparts",
                "coords",
            ],
        )

        pred_columns = pd.MultiIndex.from_product(
            [
                [model_name],
                pred_individuals,
                pred_bodyparts,
                ["x", "y", "likelihood"],
            ],
            names=[
                "scorer",
                "individuals",
                "bodyparts",
                "coords",
            ],
        )
        unique_gt_columns = pd.MultiIndex.from_product(
            [
                [scorer],
                ["single"],
                unique_bodyparts,
                ["x", "y"],
            ],
            names=[
                "scorer",
                "individuals",
                "bodyparts",
                "coords",
            ],
        )

        unique_pred_columns = pd.MultiIndex.from_product(
            [
                [model_name],
                ["single"],
                unique_bodyparts,
                ["x", "y", "likelihood"],
            ],
            names=[
                "scorer",
                "individuals",
                "bodyparts",
                "coords",
            ],
        )

        columns = gt_columns.append(pred_columns).append(unique_gt_columns).append(unique_pred_columns)

        # Values only need to match the number of columns. Distinct values
        # make debugging and shape assertions easier.
        values = np.arange(
            1,
            len(columns) + 1,
            dtype=float,
        )

        # Set all prediction likelihoods to a valid confidence.
        for index, column in enumerate(columns):
            if column[-1] == "likelihood":
                values[index] = 0.9

        row_index = pd.MultiIndex.from_tuples(
            [
                (
                    "labeled-data",
                    "video",
                    image,
                )
            ],
            names=[
                "data_folder",
                "video",
                "image",
            ],
        )

        return {
            "df_combined": pd.DataFrame(
                [values],
                index=row_index,
                columns=columns,
            ),
            "scorer": scorer,
            "model_name": model_name,
            "gt_bodyparts": list(gt_bodyparts),
            "pred_bodyparts": list(pred_bodyparts),
            "unique_bodyparts": list(unique_bodyparts),
        }

    return create


@pytest.fixture
def single_animal_evaluation_data(
    evaluation_dataframe_factory,
):
    return evaluation_dataframe_factory()


@pytest.fixture
def mocked_evaluation_plotting(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(
        visualization.auxfun_videos,
        "imread",
        Mock(return_value=frame),
    )

    save_labeled_frame = Mock()
    monkeypatch.setattr(
        visualization,
        "save_labeled_frame",
        save_labeled_frame,
    )

    return {
        "frame": frame,
        "save_labeled_frame": save_labeled_frame,
    }


class TestPlotEvaluationResults:
    def test_single_animal_with_different_labels(
        self,
        tmp_path,
        capsys,
        single_animal_evaluation_data,
        mocked_evaluation_plotting,
    ):
        """A single animal must not be counted twice across scorers."""
        data = single_animal_evaluation_data
        df_combined = data["df_combined"]

        row = df_combined.iloc[0]

        assert row.index.get_level_values("individuals").unique().tolist() == ["animal", "individual0"]
        assert row[data["scorer"]].index.get_level_values("individuals").nunique() == 1
        assert row[data["model_name"]].index.get_level_values("individuals").nunique() == 1

        visualization.plot_evaluation_results(
            df_combined=df_combined,
            project_root=tmp_path,
            scorer=data["scorer"],
            model_name=data["model_name"],
            output_folder=tmp_path / "evaluation-results",
            in_train_set=False,
        )

        output = capsys.readouterr().out

        assert "DataFrame reshape failed" not in output
        mocked_evaluation_plotting["save_labeled_frame"].assert_called_once()

    @pytest.mark.parametrize(
        ("mode", "expected_shape"),
        [
            ("bodypart", (2, 1, 2)),
            ("individual", (1, 2, 2)),
        ],
    )
    def test_plot_evaluation_results_arranges_coordinates_by_mode(
        self,
        tmp_path,
        monkeypatch,
        single_animal_evaluation_data,
        mocked_evaluation_plotting,
        mode,
        expected_shape,
    ):
        data = single_animal_evaluation_data

        make_labeled_image = Mock(side_effect=lambda *, ax, **kwargs: ax)
        monkeypatch.setattr(
            visualization,
            "make_multianimal_labeled_image",
            make_labeled_image,
        )

        visualization.plot_evaluation_results(
            df_combined=data["df_combined"],
            project_root=tmp_path,
            scorer=data["scorer"],
            model_name=data["model_name"],
            output_folder=tmp_path / "evaluation-results",
            in_train_set=False,
            mode=mode,
        )

        kwargs = make_labeled_image.call_args.kwargs

        assert kwargs["coords_truth"].shape == expected_shape
        assert kwargs["coords_pred"].shape == expected_shape
        assert kwargs["probs_pred"].shape == (
            expected_shape[0],
            expected_shape[1],
            1,
        )

    def test_plot_evaluation_results_skips_individual_count_mismatch(
        self, tmp_path, capsys, evaluation_dataframe_factory, mocked_evaluation_plotting, caplog
    ):
        data = evaluation_dataframe_factory(
            gt_individuals=("animal",),
            pred_individuals=("individual0", "individual1"),
        )

        with caplog.at_level(logging.WARNING, logger=visualization.__name__):
            visualization.plot_evaluation_results(
                df_combined=data["df_combined"],
                project_root=tmp_path,
                scorer=data["scorer"],
                model_name=data["model_name"],
                output_folder=tmp_path / "evaluation-results",
                in_train_set=False,
            )

        assert "Individual count mismatch for img001.png" in caplog.text
        assert "Ground truth individual count: 1" in caplog.text
        assert "Predictions individual count: 2" in caplog.text

        mocked_evaluation_plotting["save_labeled_frame"].assert_not_called()

    def test_plot_evaluation_results_skips_bodypart_mismatch(
        self,
        tmp_path,
        capsys,
        evaluation_dataframe_factory,
        mocked_evaluation_plotting,
        caplog,
    ):
        data = evaluation_dataframe_factory(
            gt_bodyparts=("nose", "tail"),
            pred_bodyparts=("nose", "paw"),
        )

        with caplog.at_level(logging.WARNING, logger=visualization.__name__):
            visualization.plot_evaluation_results(
                df_combined=data["df_combined"],
                project_root=tmp_path,
                scorer=data["scorer"],
                model_name=data["model_name"],
                output_folder=tmp_path / "evaluation-results",
                in_train_set=False,
            )

        assert "Bodypart mismatch for img001.png" in caplog.text
        assert "nose" in caplog.text
        assert "tail" in caplog.text
        assert "paw" in caplog.text

        mocked_evaluation_plotting["save_labeled_frame"].assert_not_called()

    def test_plot_evaluation_results_skips_malformed_coordinates(
        self, tmp_path, capsys, single_animal_evaluation_data, mocked_evaluation_plotting, caplog
    ):
        data = single_animal_evaluation_data
        df_combined = data["df_combined"].copy()

        malformed_column = (
            data["model_name"],
            "individual0",
            "tail",
            "likelihood",
        )
        df_combined = df_combined.drop(columns=[malformed_column])

        with caplog.at_level(logging.WARNING, logger=visualization.__name__):
            visualization.plot_evaluation_results(
                df_combined=df_combined,
                project_root=tmp_path,
                scorer=data["scorer"],
                model_name=data["model_name"],
                output_folder=tmp_path / "evaluation-results",
                in_train_set=False,
            )

        assert "DataFrame reshape failed for img001.png" in caplog.text
        assert "Ground truth: 4 elements (expected 4)" in caplog.text
        assert "Predictions: 5 elements (expected 6)" in caplog.text

        mocked_evaluation_plotting["save_labeled_frame"].assert_not_called()

    def test_plot_evaluation_results_rejects_invalid_mode(
        self,
        tmp_path,
        single_animal_evaluation_data,
        mocked_evaluation_plotting,
    ):
        data = single_animal_evaluation_data

        with pytest.raises(
            ValueError,
            match="Invalid mode: invalid",
        ):
            visualization.plot_evaluation_results(
                df_combined=data["df_combined"],
                project_root=tmp_path,
                scorer=data["scorer"],
                model_name=data["model_name"],
                output_folder=tmp_path / "evaluation-results",
                in_train_set=False,
                mode="invalid",
            )

        mocked_evaluation_plotting["save_labeled_frame"].assert_not_called()

    @pytest.mark.parametrize(
        ("mode", "expected_shape", "expected_offset"),
        [
            ("bodypart", (2, 1, 2), 2),
            ("individual", (1, 2, 2), 1),
        ],
    )
    def test_arranges_unique_bodyparts_by_mode(
        self,
        tmp_path,
        monkeypatch,
        evaluation_dataframe_factory,
        mocked_evaluation_plotting,
        mode,
        expected_shape,
        expected_offset,
    ):
        data = evaluation_dataframe_factory(
            unique_bodyparts=("center", "anchor"),
        )

        make_labeled_image = Mock(side_effect=lambda *, ax, **kwargs: ax)
        monkeypatch.setattr(
            visualization,
            "make_multianimal_labeled_image",
            make_labeled_image,
        )

        visualization.plot_evaluation_results(
            df_combined=data["df_combined"],
            project_root=tmp_path,
            scorer=data["scorer"],
            model_name=data["model_name"],
            output_folder=tmp_path / "evaluation-results",
            in_train_set=False,
            plot_unique_bodyparts=True,
            mode=mode,
        )

        # One call for regular body parts and one for unique body parts.
        assert make_labeled_image.call_count == 2

        unique_kwargs = make_labeled_image.call_args_list[1].kwargs

        assert unique_kwargs["coords_truth"].shape == expected_shape
        assert unique_kwargs["coords_pred"].shape == expected_shape
        assert unique_kwargs["probs_pred"].shape == (
            expected_shape[0],
            expected_shape[1],
            1,
        )
        assert unique_kwargs["color_offset"] == expected_offset

        mocked_evaluation_plotting["save_labeled_frame"].assert_called_once()


@pytest.mark.parametrize(
    ("belongs_to_train", "prefix"),
    [
        (True, "Training"),
        (False, "Test"),
    ],
)
def test_save_labeled_frame_uses_expected_filename(
    tmp_path,
    belongs_to_train,
    prefix,
):
    output_folder = tmp_path / "evaluation-results"
    output_folder.mkdir()

    image_path = tmp_path / "labeled-data" / "video" / "img001.png"

    fig, _ = plt.subplots()

    try:
        visualization.save_labeled_frame(
            fig=fig,
            image_path=image_path,
            dest_folder=output_folder,
            belongs_to_train=belongs_to_train,
        )
    finally:
        plt.close(fig)

    expected = output_folder / f"{prefix}-video-img001.png"
    assert expected.is_file()


def test_make_multianimal_labeled_image_styles_bounding_boxes():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    coords_truth = np.array([[[5, 5], [10, 10]]])
    coords_pred = np.array([[[6, 6], [11, 11]]])
    probs_pred = np.array([[[0.9], [0.8]]])

    bounding_boxes = (
        np.array(
            [
                [1, 2, 10, 12],
                [3, 4, 8, 9],
            ]
        ),
        np.array([0.9, 0.4]),
    )

    fig, ax = plt.subplots()

    result = visualization.make_multianimal_labeled_image(
        frame=frame,
        coords_truth=coords_truth,
        coords_pred=coords_pred,
        probs_pred=probs_pred,
        colors=visualization.get_cmap(1),
        ax=ax,
        bounding_boxes=bounding_boxes,
        bboxes_cutoff=0.6,
    )

    assert result is ax
    assert len(ax.patches) == 2

    confident_box, uncertain_box = ax.patches

    assert confident_box.get_xy() == (1, 2)
    assert confident_box.get_width() == 10
    assert confident_box.get_height() == 12
    assert confident_box.get_linestyle() == "-"

    assert uncertain_box.get_xy() == (3, 4)
    assert uncertain_box.get_width() == 8
    assert uncertain_box.get_height() == 9
    assert uncertain_box.get_linestyle() == "--"


def test_make_multianimal_labeled_image_styles_keypoints_by_confidence():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    coords_truth = np.array(
        [
            [
                [5, 6],
                [10, 11],
            ]
        ],
        dtype=float,
    )
    coords_pred = np.array(
        [
            [
                [7, 8],
                [12, 13],
            ]
        ],
        dtype=float,
    )
    probs_pred = np.array(
        [
            [
                [0.9],
                [0.4],
            ]
        ],
        dtype=float,
    )

    fig, ax = plt.subplots()

    result = visualization.make_multianimal_labeled_image(
        frame=frame,
        coords_truth=coords_truth,
        coords_pred=coords_pred,
        probs_pred=probs_pred,
        colors=visualization.get_cmap(1),
        dotsize=8,
        alphavalue=0.5,
        pcutoff=0.6,
        ax=ax,
    )

    assert result is ax
    assert len(ax.images) == 1
    assert len(ax.lines) == 3

    truth, reliable, unreliable = ax.lines

    assert truth.get_marker() == "+"
    assert reliable.get_marker() == "."
    assert unreliable.get_marker() == "x"

    assert truth.get_markersize() == 8
    assert reliable.get_markersize() == 8
    assert unreliable.get_markersize() == 8

    assert truth.get_alpha() == 0.5
    assert reliable.get_alpha() == 0.5
    assert unreliable.get_alpha() == 0.5

    np.testing.assert_array_equal(
        np.column_stack((truth.get_xdata(), truth.get_ydata())),
        coords_truth[0],
    )
    np.testing.assert_array_equal(
        np.column_stack((reliable.get_xdata(), reliable.get_ydata())),
        coords_pred[0, :1],
    )
    np.testing.assert_array_equal(
        np.column_stack((unreliable.get_xdata(), unreliable.get_ydata())),
        coords_pred[0, 1:],
    )
