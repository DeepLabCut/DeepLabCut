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
import os

import numpy as np
import pandas as pd

from deeplabcut.modelzoo.generalized_data_converter.datasets.base_dlc import (
    BaseDLCPoseDataset,
)
from deeplabcut.modelzoo.generalized_data_converter.datasets.utils import (
    calc_bboxes_from_keypoints,
    read_image_shape_fast,
)


class MaDLCPoseDataset(BaseDLCPoseDataset):
    def __init__(self, proj_root, dataset_name, shuffle=1, modelprefix=""):
        super(MaDLCPoseDataset, self).__init__(
            proj_root, dataset_name, shuffle=shuffle, modelprefix=modelprefix
        )

    def _df2generic(self, df, image_id_offset=0):

        individuals = df.columns.get_level_values("individuals").unique().tolist()

        unique_bpts = []

        if "single" in individuals:
            unique_bpts.extend(
                df.xs("single", level="individuals", axis=1)
                .columns.get_level_values("bodyparts")
                .unique()
            )
        multi_bpts = (
            df.xs(individuals[0], level="individuals", axis=1)
            .columns.get_level_values("bodyparts")
            .unique()
            .tolist()
        )

        coco_categories = []

        # assuming all individuals have the same name and same category id

        individual = individuals[0]

        category = {
            "name": individual,
            "id": 0,
            "supercategory": "animal",
        }

        if individual == "single":
            category["keypoints"] = unique_bpts
        else:
            category["keypoints"] = multi_bpts

        coco_categories.append(category)

        coco_images = []
        coco_annotations = []

        annotation_id = 0
        image_id = -1
        for _, file_name in enumerate(df.index):
            data = df.loc[file_name]

            # skipping all nan
            if np.isnan(data.to_numpy()).all():
                continue

            image_id += 1

            for individual_id, individual in enumerate(individuals):
                category_id = 0
                try:
                    kpts = (
                        data.xs(individual, level="individuals")
                        .to_numpy()
                        .reshape((-1, 2))
                    )
                except:
                    # somehow there are duplicates. So only use the first occurrence
                    data = data.iloc[0]
                    kpts = (
                        data.xs(individual, level="individuals")
                        .to_numpy()
                        .reshape((-1, 2))
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
                    clip=True,
                )[0][:4]

                w = xmax - xmin
                h = ymax - ymin
                area = w * h
                bbox = np.nan_to_num([xmin, ymin, w, h])
                keypoints = np.nan_to_num(keypoints.flatten())

                annotation_id += 1
                annotation = {
                    "image_id": image_id + image_id_offset,
                    "num_keypoints": num_keypoints,
                    "keypoints": keypoints,
                    "id": annotation_id,
                    "category_id": category_id,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
                if np.sum(keypoints) != 0:
                    coco_annotations.append(annotation)

            # I think width and height are important

            if isinstance(file_name, tuple):
                image_path = os.path.join(self.proj_root, *list(file_name))
            else:
                image_path = os.path.join(self.proj_root, file_name)

            _, height, width = read_image_shape_fast(image_path)

            image = {
                "file_name": image_path,
                "width": width,
                "height": height,
                "id": image_id + image_id_offset,
            }
            coco_images.append(image)

        ret_obj = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories,
        }
        return ret_obj
