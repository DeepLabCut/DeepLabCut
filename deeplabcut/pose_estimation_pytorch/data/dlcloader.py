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
"""Class implementing the Loader for DeepLabCut projects"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import re

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_pytorch.data.base import Loader
from deeplabcut.pose_estimation_pytorch.data.dataset import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.data.utils import read_image_shape_fast


class DLCLoader(Loader):
    """A Loader for DeepLabCut projects"""

    def __init__(
        self,
        config: str | Path,
        trainset_index: int = 0,
        shuffle: int = 0,
        modelprefix: str = "",
    ):
        """
        Args:
            config: path to the DeepLabCut project config
            trainset_index: the index of the TrainingsetFraction for which to load data
            shuffle: the index of the shuffle for which to load data
            modelprefix: the modelprefix for the shuffle
        """
        self._project_root = Path(config).parent
        self._project_config = af.read_config(str(config))
        self._shuffle = shuffle
        self._train_frac = self._project_config["TrainingFraction"][trainset_index]
        self._model_folder = af.get_model_folder(
            self._train_frac,
            shuffle,
            self._project_config,
            engine=Engine.PYTORCH,
            modelprefix=modelprefix,
        )
        self._evaluation_folder = af.get_evaluation_folder(
            trainFraction=self._train_frac,
            shuffle=shuffle,
            cfg=self._project_config,
            engine=Engine.PYTORCH,
            modelprefix=modelprefix,
        )
        super().__init__(
            self._project_root
            / self._model_folder
            / "train"
            / Engine.PYTORCH.pose_cfg_name
        )
        self._split = self.load_split(self._project_config, trainset_index, shuffle)
        self._df = self.load_ground_truth(self._project_config)
        self._dfs = {
            split: self.drop_duplicates(df)
            for split, df in self.split_data(self._df, self._split).items()
        }

    @property
    def df(self) -> pd.DataFrame:
        """Returns: The ground truth dataframe. Should not be modified."""
        return self._df

    @property
    def evaluation_folder(self) -> Path:
        """Returns: The path to the evaluation folder"""
        return self._project_root / self._evaluation_folder

    @property
    def project_path(self) -> Path:
        """Returns: The path to the DeepLabCut project"""
        return self._project_root

    @property
    def shuffle(self) -> int:
        """Returns: the shuffle being loaded"""
        return self._shuffle

    @property
    def train_fraction(self) -> float:
        """Returns: the fraction of the dataset used for training"""
        return self._train_frac

    def get_dataset_parameters(self) -> PoseDatasetParameters:
        """Retrieves dataset parameters based on the instance's configuration.

        Returns:
            An instance of the PoseDatasetParameters with the parameters set.
        """
        return PoseDatasetParameters(
            bodyparts=self.model_cfg["metadata"]["bodyparts"],
            unique_bpts=self.model_cfg["metadata"]["unique_bodyparts"],
            individuals=self.model_cfg["metadata"]["individuals"],
            with_center_keypoints=self.model_cfg.get("with_center_keypoints", False),
            color_mode=self.model_cfg.get("color_mode", "RGB"),
            cropped_image_size=self.model_cfg.get("output_size", (256, 256)),
        )

    def load_data(self, mode: str = "train") -> dict:
        """Loads DeepLabCut data into COCO-style annotations

        This function reads data from h5 file, split the data and returns it in
        COCO-like format

        Args:
            mode: mode indicating whether to use 'train' or 'test' data.

        Raises:
            AttributeError: if the specified mode (train or test) does not exist.

        Returns:
            the coco-style annotations
        """
        if mode not in ["train", "test"]:
            raise AttributeError(f"Unknown mode: {mode}")
        if mode not in self._dfs:
            raise ValueError(f"No split for: {mode} (found {self._dfs.keys()})")
        if self._dfs[mode] is None:
            raise ValueError(f"No data in {mode} split for this shuffle!")

        params = self.get_dataset_parameters()
        data = self.to_coco(str(self._project_root), self._dfs[mode], params)
        with_bbox = self._compute_bboxes(
            data["images"], data["annotations"], method="keypoints"
        )
        data["annotations"] = with_bbox
        return data

    @staticmethod
    def load_split(
        config: dict,
        trainset_index: int = 0,
        shuffle: int = 0,
    ) -> dict[str, list[int]]:
        """Loads the train/test split for a DeepLabCut shuffle

        Args:
            config: the DeepLabCut project config
            trainset_index: the TrainingsetFraction for which to load data
            shuffle: the index of the shuffle for which to load data

        Return:
            the {"train": [train_ids], "test": [test_ids]} data split
        """
        trainset_dir = Path(config["project_path"]) / af.get_training_set_folder(config)
        train_frac = int(100 * config["TrainingFraction"][trainset_index])
        shuffle_id = f"{config['Task']}_{train_frac}shuffle{shuffle}.pickle"
        doc_path = trainset_dir / f"Documentation_data-{shuffle_id}"

        with open(doc_path, "rb") as f:
            meta = pickle.load(f)

        train_ids = [int(i) for i in meta[1]]
        test_ids = [int(i) for i in meta[2]]

        return {"train": train_ids, "test": test_ids}

    @staticmethod
    def load_ground_truth(config: dict) -> pd.DataFrame:
        """Loads the ground truth dataset for a DeepLabCut project.

        Args:
            config: the DeepLabCut project configuration file

        Returns:
            the annotated DeepLabCut data for the current iteration

        Raises:
            ValueError: if the data contained in the ground truth HDF does not contain
                a dataframe.
        """
        trainset_dir = Path(config["project_path"]) / af.get_training_set_folder(config)
        dataset_path = f"CollectedData_{config['scorer']}.h5"
        df = pd.read_hdf(trainset_dir / dataset_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"The ground truth data in {trainset_dir} must contain a DataFrame! "
                f"Found {df}"
            )
        return df
    
    @staticmethod
    def load_predictions(bu_snapshot: Path, bu_predictions: Path, parameters: PoseDatasetParameters) -> pd.DataFrame:

        if bu_predictions is None:

            pred_path = Path(str(bu_snapshot).replace('dlc-models', 'evaluation-results')).parent.parent
            cfg = af.read_config(pred_path.parent.parent.parent / "config.yaml")
            scorer = af.get_scorer_name(
                    cfg=cfg,
                    shuffle=int(re.search(r'shuffle(\d+)', str(bu_snapshot)).group(1)),
                    trainFraction=int(re.search(r'trainset(\d+)', str(bu_snapshot)).group(1)) / 100,
                    engine=Engine.PYTORCH,
                    trainingsiterations=re.search(r'snapshot-(.+)\.pth', str(bu_snapshot)).group(1),
                    modelprefix="",
                )
            pred_file = pred_path / f"{scorer[0]}.h5"

            dlc_preds = pd.read_hdf(pred_file, key="df_with_missing")

            #FIXME: Implement the case where snapshot is loaded
            raise NotImplementedError("Need to implement the case with loaded snapshot")

        else:
            pred_path = bu_predictions.parent.parent
            dlc_preds = pd.read_hdf(bu_predictions, key="df_with_missing")
        
        predictions = {}
        for idx in dlc_preds.index.unique():
            img_path = pred_path.parent.parent / Path(*idx)
            keypoints = dlc_preds.loc[idx].values.reshape(-1,len(parameters.bodyparts),3)[:,:,:2]
            keypoints = keypoints[~np.isnan(keypoints).all(axis=-1).all(axis=-1)]
            predictions[str(img_path)] = keypoints

        return predictions
        

    @staticmethod
    def split_data(
        dlc_df: pd.DataFrame,
        split: dict[str, list[int]],
    ) -> dict[str, pd.DataFrame | None]:
        """
        Splits a DeepLabCut DataFrame into train/test dataframes

        Args:
            dlc_df: the dataframe containing the labeled data
            split: the train/test indices

        Returns:
            a dictionary containing the same keys as the split dictionary, where the
            values are the rows of dlc_df with index in the split, or None if there are
            no indices in that split
        """
        split_dfs = {}
        for k, indices in split.items():
            if len(indices) == 0:
                split_dfs[k] = None
            else:
                split_dfs[k] = dlc_df.iloc[indices]
        return split_dfs

    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Returns: the DataFrame with no duplicate rows"""
        return df[~df.index.duplicated(keep="first")]

    @staticmethod
    def to_coco(
        project_root: str | Path,
        df: pd.DataFrame,
        parameters: PoseDatasetParameters,
    ) -> dict:
        """Formerly Shaokai's function

        Args:
            project_root: the path to the project root
            df: the DLC-format annotation dataframe to convert to a COCO-format dict
            parameters: the parameters for pose estimation

        Returns:
            the coco format data
        """
        with_individuals = "individuals" in df.columns.names
        if (
            not with_individuals and
            (len(parameters.individuals) > 1 or len(parameters.unique_bpts) > 0)
        ):
            raise ValueError(
                "The DataFrame contains single-animal annotations (for a single, "
                "individual), but the parameters suggest this is a multi-animal project"
                f": {parameters} (with multiple individuals or unique bodyparts)"
            )

        categories = [
            {
                "id": 1,
                "name": "animals",
                "supercategory": "animal",
                "keypoints": parameters.bodyparts,
            },
        ]
        individuals = [idv for idv in parameters.individuals]
        if len(parameters.unique_bpts) > 0:
            individuals += ["single"]
            categories.append(
                {
                    "id": 2,
                    "name": "unique_bodypart",
                    "supercategory": "animal",
                    "keypoints": parameters.unique_bpts,
                }
            )

        anns, images = [], []
        base_path = Path(project_root)
        for idx, row in df.iterrows():
            image_id = len(images) + 1
            rel_path = Path(*idx) if isinstance(idx, tuple) else Path(idx)
            path = str(base_path / rel_path)
            _, height, width = read_image_shape_fast(path)
            images.append(
                {
                    "id": image_id,
                    "file_name": path,
                    "width": width,
                    "height": height,
                }
            )

            for idv_idx, idv in enumerate(individuals):
                category_id = 1
                individual_id = idv_idx
                if with_individuals:
                    if idv == "single":
                        category_id = 2
                        individual_id = -1
                    data = row.xs(idv, level="individuals")
                else:
                    data = row

                raw_keypoints = data.to_numpy().reshape((-1, 2))
                keypoints = np.zeros((len(raw_keypoints), 3))
                keypoints[:, :2] = raw_keypoints
                is_visible = ~pd.isnull(raw_keypoints).all(axis=1)
                keypoints[:, 2] = np.where(is_visible, 2, 0)
                num_keypoints = is_visible.sum()
                if num_keypoints > 0:
                    anns.append(
                        {
                            "id": len(anns) + 1,
                            "image_id": image_id,
                            "category_id": category_id,
                            "individual": idv,
                            "individual_id": individual_id,
                            "num_keypoints": num_keypoints,
                            "keypoints": keypoints,
                            "iscrowd": 0,
                        }
                    )

        return {"annotations": anns, "categories": categories, "images": images}
