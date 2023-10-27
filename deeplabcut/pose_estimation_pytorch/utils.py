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
from __future__ import annotations

import abc
import os

import numpy as np
import pandas as pd
import torch

from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    read_image_shape_fast,
)
from deeplabcut.pose_estimation_tensorflow.lib.trackingutils import (
    calc_bboxes_from_keypoints,
)
from deeplabcut.utils.auxiliaryfunctions import read_plainconfig

# Shaokai's function


def df_to_generic(proj_root: str, df: pd.DataFrame, image_id_offset: int = 0) -> dict:
    """
    Convert a pandas DataFrame containing pose estimation data to a dictionary in COCO format.

    Args:
        proj_root (str): The root directory of the project.
        df (pd.DataFrame): The DataFrame containing the pose estimation data.
        image_id_offset (int, optional): The offset to add to the image IDs. Defaults to 0.

    Returns:
        dict: A dictionary in COCO format containing the images, annotations, and categories.
    """
    try:
        individuals = df.columns.get_level_values(
            'individuals',
        ).unique().tolist()
    except KeyError:
        new_cols = pd.MultiIndex.from_tuples(
            [(col[0], 'animal', col[1], col[2]) for col in df.columns],
            names=['scorer', 'individuals', 'bodyparts', 'coords'],
        )
        df.columns = new_cols

        individuals = df.columns.get_level_values(
            'individuals',
        ).unique().tolist()

    unique_bpts = []

    if 'single' in individuals:
        unique_bpts.extend(
            df.xs('single', level='individuals', axis=1)
            .columns.get_level_values('bodyparts')
            .unique(),
        )
    multi_bpts = (
        df.xs(individuals[0], level='individuals', axis=1)
        .columns.get_level_values('bodyparts')
        .unique()
        .tolist()
    )

    coco_categories = []

    # assuming all individuals have the same name and same category id

    individual = individuals[0]

    category = {
        'name': individual,
        'id': 0,
        'supercategory': 'animal',
    }

    if individual == 'single':
        category['keypoints'] = unique_bpts
    else:
        category['keypoints'] = multi_bpts

    coco_categories.append(category)

    coco_images = []
    coco_annotations = []

    annotation_id = 0
    image_id = -1
    for _, file_name in enumerate(df.index):
        data = df.loc[file_name]

        # skipping all nan
        # if np.isnan(data.to_numpy()).all():
        #     continue

        image_id += 1

        for individual_id, individual in enumerate(individuals):
            category_id = 1  # 0 is for background by default
            try:
                kpts = (
                    data.xs(individual, level='individuals').to_numpy().reshape(
                        (-1, 2),
                    )
                )
            except:
                # somehow there are duplicates. So only use the first occurrence
                data = data.iloc[0]
                kpts = (
                    data.xs(individual, level='individuals').to_numpy().reshape(
                        (-1, 2),
                    )
                )

            keypoints = np.zeros((len(kpts), 3))

            keypoints[:, :2] = kpts

            is_visible = ~pd.isnull(kpts).all(axis=1)

            keypoints[:, 2] = np.where(is_visible, 2, 0)

            num_keypoints = is_visible.sum()

            bbox_margin = 20

            xmin, ymin, xmax, ymax = calc_bboxes_from_keypoints(
                [keypoints],
                slack=bbox_margin,
            )[0][:4]

            w = xmax - xmin
            h = ymax - ymin
            area = w * h
            bbox = np.nan_to_num([xmin, ymin, w, h])
            keypoints = np.nan_to_num(keypoints.flatten())

            annotation_id += 1
            annotation = {
                'image_id': image_id + image_id_offset,
                'num_keypoints': num_keypoints,
                'keypoints': keypoints,
                'id': annotation_id,
                'category_id': category_id,
                'individual': individual,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0,
            }

            # adds an annotation even if no keypoint is annotated for the current individual
            # This is not standard for COCO but is useful because each image will then have
            # the same number of annotations (i.e possible to train with batches without overcomplicating the code)
            coco_annotations.append(annotation)

        # I think width and height are important

        if isinstance(file_name, tuple):
            image_path = os.path.join(proj_root, *list(file_name))
        else:
            image_path = os.path.join(proj_root, file_name)

        _, height, width = read_image_shape_fast(image_path)

        image = {
            'file_name': image_path,
            'width': width,
            'height': height,
            'id': image_id + image_id_offset,
        }
        coco_images.append(image)

    ret_obj = {
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': coco_categories,
    }
    return ret_obj


def create_folder(path_to_folder):
    """Creates all folders contained in the path.

    Args:
        path_to_folder: Path to the folder that should be created
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)


def fix_seeds(seed: int):
    """
    Fixes seed for all random functions
    @param seed: int
        Seed to be fixed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq: The sequence to be checked.
        expected_type: Expected type of sequence items.
        seq_type: Expected sequence type.
    Returns:
        Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def get_pytorch_config(modelfolder):
    pytorch_config_path = os.path.join(
        modelfolder, 'train', 'pytorch_config.yaml',
    )
    pytorch_cfg = read_plainconfig(pytorch_config_path)

    return pytorch_cfg
