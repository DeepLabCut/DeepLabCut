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

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

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
        self._resolutions = set()

        super().__init__(
            self._project_root
            / self._model_folder
            / "train"
            / Engine.PYTORCH.pose_cfg_name
        )
        self._dfs, image_sizes = self.load_ground_truth(
            self._project_config,
            trainset_index=trainset_index,
            shuffle=shuffle,
        )
        self._resolutions = self._resolutions.union(image_sizes)

    @property
    def df(self) -> pd.DataFrame:
        """Returns: The ground truth dataframe. Should not be modified."""
        return self._dfs["full"]

    def image_resolutions(self) -> set[tuple[int, int]]:
        """Returns: The collection of image resolutions present in the dataset"""
        return self._resolutions

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

    def load_ground_truth(
        self,
        config: dict,
        trainset_index: int,
        shuffle: int,
    ) -> tuple[dict[str, pd.DataFrame], set[tuple[int, int]]]:
        """Loads the ground truth dataset for a DeepLabCut project.

        Args:
            config: the DeepLabCut project configuration file
            trainset_index: the TrainingsetFraction for which to load data
            shuffle: the index of the shuffle for which to load data

        Returns: ground_truth_dataframes, image_resolutions
            ground_truth_dataframes: a dictionary containing the different DataFrames
                for the annotated DeepLabCut data for the current iteration
            image_resolutions: all possible image resolutions in the dataset

        Raises:
            ValueError: if the data contained in the ground truth HDF does not contain
                a dataframe.
        """
        trainset_dir = Path(config["project_path"]) / af.get_training_set_folder(config)
        dataset_path = f"CollectedData_{config['scorer']}.h5"
        train_frac = int(100 * config["TrainingFraction"][trainset_index])
        project_id = f"{config['Task']}_{config['scorer']}"
        dataset_file = trainset_dir / f"{project_id}{train_frac}shuffle{shuffle}"
        params = self.get_dataset_parameters()

        # as in TF DeepLabCut, load the training data from the .mat/.pickle file
        if config.get("multianimalproject", False):
            image_sizes, df_train = _load_pickle_dataset(
                dataset_file.with_suffix(".pickle"), config["scorer"], params=params,
            )
        else:
            image_sizes, df_train = _load_mat_dataset(
                dataset_file.with_suffix(".mat"), config["scorer"], params=params,
            )

        # load the full dataset file
        df = pd.read_hdf(trainset_dir / dataset_path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"The ground truth data in {trainset_dir} must contain a DataFrame! "
                f"Found {df}"
            )

        # load the data splits, check that there's nothing suspect
        splits = self.load_split(self._project_config, trainset_index, shuffle)
        dfs = self.split_data(df, splits)
        dfs["full"] = df
        # let's not validate for now
        # dfs = _validate_dataframes(dfs, df_train)
        return dfs, image_sizes

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
            rel_path = Path(*idx) if isinstance(idx, tuple) else Path(str(idx))
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
                is_visible = np.logical_and(
                    ~pd.isnull(raw_keypoints).all(axis=1),
                    np.logical_and(
                        np.logical_and(
                            0 < keypoints[..., 0],
                            keypoints[..., 0] < width,
                        ),
                        np.logical_and(
                            0 < keypoints[..., 1],
                            keypoints[..., 1] < height,
                        ),
                    )
                )
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


def _load_mat_dataset(
    file: Path,
    scorer: str,
    params: PoseDatasetParameters,
) -> tuple[set[tuple[int, int]], pd.DataFrame]:
    """Loads the training dataset stored as a .mat file

    Returns: images_sizes, dlc_dataset
        images_sizes: all possible images sizes in the dataset
        dlc_dataset: the dataset in a DLC-format DataFrame
    """
    if not params.max_num_animals == 1:
        raise RuntimeError(
            f"Cannot load a multi-animal pose dataset from a `.mat` file ({file})"
        )

    raw_data = sio.loadmat(str(file))
    dataset = raw_data["dataset"]
    num_images = dataset.shape[1]

    image_sizes = set()
    index, data = [], []
    for i in range(num_images):
        item = dataset[0, i]

        # add the image size
        c, h, w = item[1][0]
        image_sizes.add((h, w))

        # parse image path
        raw_path = item[0][0]
        if isinstance(raw_path, str):
            image_path = Path(raw_path).parts[-3:]
        else:
            image_path = tuple([p.strip() for p in raw_path])
        index.append(image_path)

        # parse data
        keypoints = np.zeros((1, params.num_joints, 2))
        keypoints.fill(np.nan)
        if len(item) >= 3:
            joints = item[2][0][0]
            for joint_id, x, y in joints:
                keypoints[0, joint_id, 0] = x
                keypoints[0, joint_id, 1] = y

            joint_id = joints[:, 0]
            if joint_id.size != 0:  # make sure joint ids are 0-indexed
                assert (joint_id < params.num_joints).any()
            joints[:, 0] = joint_id

        data.append(keypoints)

    dataframe = pd.DataFrame(
        data=np.stack(data, axis=0).reshape((num_images, -1)),
        index=pd.MultiIndex.from_tuples(index),
        columns=build_dlc_dataframe_columns(scorer, params, False),
    )
    dataframe = dataframe.sort_index(axis=0)
    return image_sizes, dataframe


def _load_pickle_dataset(
    file: Path,
    scorer: str,
    params: PoseDatasetParameters,
) -> tuple[set[tuple[int, int]], pd.DataFrame]:
    """Loads the training dataset stored as a .mat file

    Returns: images_sizes, dlc_dataset
        images_sizes: all possible images sizes in the dataset
        dlc_dataset: the dataset in a DLC-format DataFrame
    """
    with open(file, "rb") as f:
        raw_data = pickle.load(f)

    num_images = len(raw_data)
    image_sizes = set()
    index, data = [], []
    data_unique = None
    if params.num_unique_bpts > 0:
        data_unique = []

    for image_data in raw_data:
        # add image path
        index.append(image_data["image"])

        # add image size
        c, h, w = image_data["size"]
        image_sizes.add((h, w))

        # add keypoints
        keypoints = np.zeros((params.max_num_animals, params.num_joints, 2))
        keypoints.fill(np.nan)
        keypoints_unique = None
        for idv_idx, idv_bodyparts in image_data.get("joints", {}).items():
            if idv_idx < params.max_num_animals:
                for joint_id, x, y in idv_bodyparts:
                    bodypart = int(joint_id)
                    keypoints[idv_idx, bodypart, 0] = x
                    keypoints[idv_idx, bodypart, 1] = y

            elif idv_idx == params.max_num_animals and data_unique is not None and keypoints_unique is None:
                keypoints_unique = np.zeros((params.num_unique_bpts, 2))
                keypoints_unique.fill(np.nan)
                for joint_id, x, y in idv_bodyparts:
                    unique_bpt_id = int(joint_id) - params.num_joints
                    keypoints_unique[unique_bpt_id, 0] = x
                    keypoints_unique[unique_bpt_id, 1] = y

            else:
                raise ValueError(f"Malformed dataset: {params}, {image_data}")

        data.append(keypoints)
        if data_unique is not None:
            if keypoints_unique is None:
                keypoints_unique = np.zeros((params.num_unique_bpts, 2))
                keypoints_unique.fill(np.nan)
            data_unique.append(keypoints_unique)

    data = np.stack(data, axis=0).reshape((num_images, -1))
    if data_unique is not None:
        data_unique = np.stack(data_unique, axis=0).reshape((num_images, -1))
        data = np.concatenate([data, data_unique], axis=1)

    dataframe = pd.DataFrame(
        data=data,
        index=pd.MultiIndex.from_tuples(index),
        columns=build_dlc_dataframe_columns(scorer, params, False),
    )
    dataframe = dataframe.sort_index(axis=0)
    return image_sizes, dataframe


def _validate_dataframes(
    dfs: dict[str, pd.DataFrame], df_train: pd.DataFrame, strict: bool = False,
) -> dict[str, pd.DataFrame]:
    """Validates the training/test DataFrames

    Performs the following validation steps:
        1. Checks that the training data loaded from CollectedData.h5 matches the
            training data stored in the ".mat" or ".pickle" file.
        2. Checks that there are no duplicate entries in the DataFrames (if there are
            any, removes them)
        3. Checks that there is no data leak between the training and test set (if there
            is, prints a warning)

    Args:
        dfs: the "full" and split DataFrames loaded from the H5 file
        df_train: the training data loaded from the ".mat" or ".pickle" file
        strict: Whether to fail if the data does not pass validation (instead of
            attempting a fix).

    Returns:
        The validated and sanitized DataFrames

    Raises:
        ValueError: if strict and there is a small fixable error, or if there are images
        that are present in both the training and test set.
    """
    error = False

    # checks that all images in the .pickle/.mat file are in the HDF
    pickle_train_images = set(df_train.index)
    hdf_train_images = set(dfs["train"].index)
    missing_images = pickle_train_images - hdf_train_images
    extra_images = hdf_train_images - pickle_train_images
    if len(missing_images) > 0:
        error = True
        logging.debug(
            f"Found images in the dataset file which were not in H5: {missing_images}"
        )
    if len(extra_images) > 0:
        error = True
        logging.debug(
            f"Found images in the H5 file which were not in the dataset: {extra_images}"
        )

    # checks that the data is close for the similar images
    train_index = list(hdf_train_images.intersection(pickle_train_images))
    data_h5 = np.nan_to_num(dfs["full"].loc[train_index], nan=-1)
    data_pickle_mat = np.nan_to_num(df_train, nan=-1)
    if not np.isclose(data_h5, data_pickle_mat, atol=0.1).all():
        error = True
        logging.debug(
            "Found differences between the training-dataset HDF (.h5) data and the "
            "training data found. This might be the case if you refined your data "
            "after creating the dataset, and then created a new shuffle."
        )

    # checks that there are no duplicate entries
    dfs_clean = {}
    for split, df in dfs.items():
        dup = df.index.duplicated(keep="first")
        num_dup = dup.sum()
        if dup.sum() > 0:
            error = True
            logging.debug(f"Found {num_dup} duplicates in {split}: {df[dup].index}")
            dfs_clean[split] = df[~dup]
        else:
            dfs_clean[split] = df[~dup]

    # check for leaks
    if dfs["test"] is not None:
        train_images = set(dfs["train"].index)
        test_images = set(dfs["test"].index)
        leak = train_images.intersection(test_images)
        if len(leak) > 0:
            logging.warning(
                f"Found images both in the training and test set: {leak}! To resolve "
                "this issue please try the following:\n"
                f"  1. Check that each video is listed exactly once in your project's"
                f"`config.yaml`\n"
                f"  2. Make sure all of your videos have different names."
                f"  3. You can use `dropduplicatesinannotatinfiles` and "
                f"`comparevideolistsanddatafolders` to ensure that there are no more "
                f"duplicates"
                f"  3. Switch to a new iteration and create a fresh training dataset"
            )

    if error and strict:
        raise ValueError(f"Found errors when validating the dataset")

    return dfs


def build_dlc_dataframe_columns(
    scorer: str,
    parameters: PoseDatasetParameters,
    with_likelihood: bool,
) -> pd.MultiIndex:
    """Builds the columns for a DeepLabCut DataFrame

    Args:
        scorer: the scorer name
        parameters: the parameters for the project
        with_likelihood: whether the DataFrame contains pose likelihood

    Returns:
        the multi-index columns for the DataFrame
    """
    levels = ["scorer", "individuals", "bodyparts", "coords"]
    kpt_entries = ["x", "y"]
    if with_likelihood:
        kpt_entries.append("likelihood")

    columns = []
    for i in parameters.individuals:
        for b in parameters.bodyparts:
            columns += [(scorer, i, b, entry) for entry in kpt_entries]

    for unique_bpt in parameters.unique_bpts:
        columns += [(scorer, "single", unique_bpt, entry) for entry in kpt_entries]

    return pd.MultiIndex.from_tuples(columns, names=levels)
