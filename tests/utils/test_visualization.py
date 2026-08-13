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
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from deeplabcut.utils import visualization


@pytest.fixture(autospec=True)
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

        columns = gt_columns.append(pred_columns)

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
        self,
        tmp_path,
        capsys,
        evaluation_dataframe_factory,
        mocked_evaluation_plotting,
    ):
        data = evaluation_dataframe_factory(
            gt_individuals=("animal",),
            pred_individuals=("individual0", "individual1"),
        )

        visualization.plot_evaluation_results(
            df_combined=data["df_combined"],
            project_root=tmp_path,
            scorer=data["scorer"],
            model_name=data["model_name"],
            output_folder=tmp_path / "evaluation-results",
            in_train_set=False,
        )

        output = capsys.readouterr().out

        assert "Individual count mismatch for img001.png" in output
        assert "Ground truth individual count: 1" in output
        assert "Predictions individual count: 2" in output

        mocked_evaluation_plotting["save_labeled_frame"].assert_not_called()

    def test_plot_evaluation_results_skips_bodypart_mismatch(
        self,
        tmp_path,
        capsys,
        evaluation_dataframe_factory,
        mocked_evaluation_plotting,
    ):
        data = evaluation_dataframe_factory(
            gt_bodyparts=("nose", "tail"),
            pred_bodyparts=("nose", "paw"),
        )

        visualization.plot_evaluation_results(
            df_combined=data["df_combined"],
            project_root=tmp_path,
            scorer=data["scorer"],
            model_name=data["model_name"],
            output_folder=tmp_path / "evaluation-results",
            in_train_set=False,
        )

        output = capsys.readouterr().out

        assert "Bodypart mismatch for img001.png" in output
        assert "nose" in output
        assert "tail" in output
        assert "paw" in output

        mocked_evaluation_plotting["save_labeled_frame"].assert_not_called()

    def test_plot_evaluation_results_skips_malformed_coordinates(
        self,
        tmp_path,
        capsys,
        single_animal_evaluation_data,
        mocked_evaluation_plotting,
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

        visualization.plot_evaluation_results(
            df_combined=df_combined,
            project_root=tmp_path,
            scorer=data["scorer"],
            model_name=data["model_name"],
            output_folder=tmp_path / "evaluation-results",
            in_train_set=False,
        )

        output = capsys.readouterr().out

        assert "DataFrame reshape failed for img001.png" in output
        assert "Ground truth: 4 elements (expected 4)" in output
        assert "Predictions: 5 elements (expected 6)" in output

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

    coords_truth = np.array(
        [
            [
                [5, 5],
                [10, 10],
            ]
        ]
    )
    coords_pred = np.array(
        [
            [
                [6, 6],
                [11, 11],
            ]
        ]
    )
    probs_pred = np.array(
        [
            [
                [0.9],
                [0.8],
            ]
        ]
    )

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

    try:
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
        assert ax.patches[0].get_linestyle() == "-"
        assert ax.patches[1].get_linestyle() == "--"
    finally:
        plt.close(fig)
