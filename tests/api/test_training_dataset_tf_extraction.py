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
"""Unit tests for TF pose-config extraction in create_training_dataset.

Verifies that ``engine=Engine.TF`` calls tensorflow_compat helpers and
``engine=Engine.PYTORCH`` does not.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import scipy.io as sio

import deeplabcut
from deeplabcut.core.config import ProjectConfig
from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation import (
    create_multianimaltraining_dataset,
)

_TRAINSET = "deeplabcut.generate_training_dataset.trainingsetmanipulation"
_MULTI = "deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation"
_TF_SINGLE = "deeplabcut.tensorflow_compat.dataset_management.create_single_animal"
_TF_MULTI = "deeplabcut.tensorflow_compat.dataset_management.create_multi_animal"


def _single_animal_cfg(project_path: Path) -> ProjectConfig:
    return ProjectConfig(
        Task="test",
        scorer="scorer",
        date="Jan1",
        multianimalproject=False,
        project_path=str(project_path),
        bodyparts=["nose", "tail"],
        TrainingFraction=[0.95],
        iteration=0,
        default_net_type="resnet_50",
        default_augmenter="imgaug",
        video_sets={},
    )


def _multi_animal_cfg(project_path: Path) -> ProjectConfig:
    return ProjectConfig(
        Task="test",
        scorer="scorer",
        date="Jan1",
        multianimalproject=True,
        identity=False,
        project_path=str(project_path),
        individuals=["ind1"],
        uniquebodyparts=[],
        multianimalbodyparts=["nose", "tail"],
        bodyparts="MULTI!",
        TrainingFraction=[0.95],
        iteration=0,
        default_net_type="dlcrnet_ms5",
        default_augmenter="imgaug",
        video_sets={},
        skeleton=[],
    )


def _labeled_dataframe(n_frames: int = 4) -> pd.DataFrame:
    """Minimal merged annotation table keyed by scorer (as used after merge)."""
    index = pd.MultiIndex.from_tuples(
        [(f"labeled-data/vid/img{i:03d}.png",) for i in range(n_frames)],
        names=["path"],
    )
    cols = pd.MultiIndex.from_product(
        [["scorer"], ["nose", "tail"], ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    return pd.DataFrame(0.0, index=index, columns=cols)


def _enter_single_animal_mocks(stack: ExitStack, cfg, labeled) -> MagicMock:
    mock_meta_cls = MagicMock()
    mock_meta_cls.path.return_value.exists.return_value = True
    stack.enter_context(patch(f"{_TRAINSET}.ProjectConfig.from_any", return_value=cfg))
    stack.enter_context(patch(f"{_TRAINSET}.merge_annotateddatasets", return_value=labeled))
    stack.enter_context(patch(f"{_TRAINSET}.format_training_data", return_value=([], [{"fake": True}])))
    stack.enter_context(patch(f"{_TRAINSET}.metadata.TrainingDatasetMetadata", mock_meta_cls))
    stack.enter_context(patch(f"{_TRAINSET}.metadata.update_metadata"))
    stack.enter_context(patch(f"{_TRAINSET}.auxiliaryfunctions.attempt_to_make_folder"))
    stack.enter_context(
        patch(
            f"{_TRAINSET}.auxiliaryfunctions.get_training_set_folder",
            return_value=Path("training-datasets/iteration-0/UnaugmentedDataSet_testJan1"),
        )
    )
    stack.enter_context(
        patch(
            f"{_TRAINSET}.auxiliaryfunctions.get_data_and_metadata_filenames",
            return_value=(Path("data.mat"), Path("meta.pickle")),
        )
    )
    stack.enter_context(
        patch(
            f"{_TRAINSET}.auxiliaryfunctions.get_model_folder",
            return_value=Path("dlc-models/iteration-0/testJan1-trainset95shuffle1"),
        )
    )
    stack.enter_context(patch(f"{_TRAINSET}.auxiliaryfunctions.save_metadata"))
    stack.enter_context(patch(f"{_TRAINSET}.auxiliaryfunctions.get_bodyparts", return_value=["nose", "tail"]))
    stack.enter_context(patch.object(sio, "savemat"))
    return mock_meta_cls


def test_create_training_dataset_tf_calls_tf_create_pose_config_files(tmp_path):
    cfg = _single_animal_cfg(tmp_path)
    labeled = _labeled_dataframe()
    mock_tf_create = MagicMock()

    with ExitStack() as stack:
        _enter_single_animal_mocks(stack, cfg, labeled)
        stack.enter_context(patch(f"{_TF_SINGLE}._tf_get_model_path", return_value="/fake/model"))
        stack.enter_context(patch(f"{_TF_SINGLE}._tf_create_pose_config_files", mock_tf_create))

        deeplabcut.create_training_dataset(
            config=str(tmp_path / "config.yaml"),
            num_shuffles=1,
            Shuffles=[1],
            userfeedback=False,
            engine=Engine.TF,
            net_type="resnet_50",
            augmenter_type="imgaug",
        )

    mock_tf_create.assert_called_once()
    kwargs = mock_tf_create.call_args.kwargs
    assert kwargs["net_type"] == "resnet_50"
    assert kwargs["bodyparts"] == ["nose", "tail"]
    assert kwargs["augmenter_type"] == "imgaug"


def test_create_training_dataset_pytorch_does_not_call_tf_helper(tmp_path):
    cfg = _single_animal_cfg(tmp_path)
    labeled = _labeled_dataframe()
    mock_tf_create = MagicMock()
    mock_pytorch_cfg = MagicMock()

    with ExitStack() as stack:
        _enter_single_animal_mocks(stack, cfg, labeled)
        stack.enter_context(patch(f"{_TF_SINGLE}._tf_create_pose_config_files", mock_tf_create))
        mock_make_pt = stack.enter_context(
            patch(
                "deeplabcut.pose_estimation_pytorch.config.make_pose_config.make_pytorch_pose_config",
                return_value=mock_pytorch_cfg,
            )
        )
        mock_make_test = stack.enter_context(
            patch(
                "deeplabcut.pose_estimation_pytorch.config.make_pose_config.make_pytorch_test_config",
            )
        )

        deeplabcut.create_training_dataset(
            config=str(tmp_path / "config.yaml"),
            num_shuffles=1,
            Shuffles=[1],
            userfeedback=False,
            engine=Engine.PYTORCH,
            net_type="resnet_50",
            augmenter_type="albumentations",
        )

    mock_tf_create.assert_not_called()
    mock_make_pt.assert_called_once()
    mock_make_test.assert_called_once()


def test_create_multianimaltraining_dataset_tf_calls_compat_helper(tmp_path):
    cfg = _multi_animal_cfg(tmp_path)
    labeled = _labeled_dataframe()
    mock_tf_create = MagicMock()
    mock_meta_cls = MagicMock()
    mock_meta_cls.path.return_value.exists.return_value = True

    with ExitStack() as stack:
        stack.enter_context(patch(f"{_MULTI}.ProjectConfig.from_any", return_value=cfg))
        stack.enter_context(patch(f"{_MULTI}.merge_annotateddatasets", return_value=labeled))
        stack.enter_context(patch(f"{_MULTI}.format_multianimal_training_data", return_value=[{"fake": True}]))
        stack.enter_context(patch(f"{_MULTI}.metadata.TrainingDatasetMetadata", mock_meta_cls))
        stack.enter_context(patch(f"{_MULTI}.metadata.update_metadata"))
        stack.enter_context(patch(f"{_MULTI}.auxiliaryfunctions.attempt_to_make_folder"))
        stack.enter_context(
            patch(
                f"{_MULTI}.auxiliaryfunctions.get_training_set_folder",
                return_value=Path("training-datasets/iteration-0/UnaugmentedDataSet_testJan1"),
            )
        )
        stack.enter_context(
            patch(
                f"{_MULTI}.auxiliaryfunctions.get_data_and_metadata_filenames",
                return_value=(Path("data.pickle"), Path("meta.pickle")),
            )
        )
        stack.enter_context(
            patch(
                f"{_MULTI}.auxiliaryfunctions.get_model_folder",
                return_value=Path("dlc-models/iteration-0/testJan1-trainset95shuffle1"),
            )
        )
        stack.enter_context(patch(f"{_MULTI}.auxiliaryfunctions.save_metadata"))
        stack.enter_context(patch(f"{_TF_SINGLE}._tf_get_model_path", return_value="/fake/model"))
        stack.enter_context(patch(f"{_TF_MULTI}._tf_create_multianimal_pose_config_files", mock_tf_create))
        stack.enter_context(patch(f"{_MULTI}.MakeInference_yaml"))
        stack.enter_context(patch("builtins.open", MagicMock()))
        stack.enter_context(patch("pickle.dump"))

        create_multianimaltraining_dataset(
            cfg,
            num_shuffles=1,
            Shuffles=[1],
            net_type="dlcrnet_ms5",
            userfeedback=False,
            engine=Engine.TF,
        )

    mock_tf_create.assert_called_once()
    # dlcrnet_ms5 is rewritten to resnet_50 for TF multi-stage
    assert mock_tf_create.call_args.kwargs["net_type"] == "resnet_50"
    assert mock_tf_create.call_args.kwargs["multi_stage"] is True
