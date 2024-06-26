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
from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    parse_video_filenames,
)
from deeplabcut.modelzoo.generalized_data_converter.datasets.base import BasePoseDataset
from deeplabcut.modelzoo.generalized_data_converter.datasets.utils import (
    calc_bboxes_from_keypoints,
    read_image_shape_fast,
)
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions, conversioncode


def merge_annotateddatasets(cfg):
    """
    Merges all the h5 files for all labeled-datasets (from individual videos).

    This is a bit of a mess because of cross platform compatibility.

    Within platform comp. is straightforward. But if someone labels on windows and wants to train on a unix cluster or colab...
    """
    AnnotationData = []
    data_path = Path(os.path.join(cfg["project_path"], "labeled-data"))
    videos = cfg["video_sets"].keys()
    video_filenames = parse_video_filenames(videos)
    for filename in video_filenames:
        file_path = os.path.join(
            data_path / filename, f'CollectedData_{cfg["scorer"]}.h5'
        )
        try:
            data = pd.read_hdf(file_path)
            conversioncode.guarantee_multiindex_rows(data)
            if data.columns.levels[0][0] != cfg["scorer"]:
                print(
                    f"{file_path} labeled by a different scorer. This data will not be utilized in training dataset creation. If you need to merge datasets across scorers, see https://github.com/DeepLabCut/DeepLabCut/wiki/Using-labeled-data-in-DeepLabCut-that-was-annotated-elsewhere-(or-merge-across-labelers)"
                )
                continue
            AnnotationData.append(data)
        except FileNotFoundError:
            print(file_path, " not found (perhaps not annotated).")

    if not len(AnnotationData):
        print(
            "Annotation data was not found by splitting video paths (from config['video_sets']). An alternative route is taken..."
        )
        AnnotationData = conversioncode.merge_windowsannotationdataONlinuxsystem(cfg)
        if not len(AnnotationData):
            print("No data was found!")
            return

    AnnotationData = pd.concat(AnnotationData).sort_index()
    # When concatenating DataFrames with misaligned column labels,
    # all sorts of reordering may happen (mainly depending on 'sort' and 'join')
    # Ensure the 'bodyparts' level agrees with the order in the config file.
    if cfg.get("multianimalproject", False):
        (
            _,
            uniquebodyparts,
            multianimalbodyparts,
        ) = auxfun_multianimal.extractindividualsandbodyparts(cfg)
        bodyparts = multianimalbodyparts + uniquebodyparts
    else:
        bodyparts = cfg["bodyparts"]
    AnnotationData = AnnotationData.reindex(
        bodyparts, axis=1, level=AnnotationData.columns.names.index("bodyparts")
    )

    return AnnotationData


class MaDLCDataFrame(BasePoseDataset):

    def __init__(self, proj_root, dataset_name):
        super(MaDLCDataFrame, self).__init__()
        assert proj_root != None and dataset_name != None
        self.proj_root = proj_root
        self.dataset_name = dataset_name
        self.meta["dataset_name"] = dataset_name
        self.meta["proj_root"] = proj_root
        config_path = Path(proj_root) / "config.yaml"
        # read config
        cfg = auxiliaryfunctions.read_config(config_path)
        # get the train folder

        Data = merge_annotateddatasets(
            cfg,
        )

        # now with this data, we construct necessary generic data

        self.dlc_df = Data

        images = self.dlc_df.index

        ratio = 0.9

        df_train = self.dlc_df.iloc[: int(len(images) * ratio)]
        df_test = self.dlc_df.iloc[int(len(images) * ratio) :]

        self.coco_train = self._df2generic(df_train)

        offset = len(self.coco_train["images"])

        self.coco_test = self._df2generic(df_test, image_id_offset=offset)

        self.populate_generic()

    def populate_generic(self):

        self.generic_train_images = self.coco_train["images"]
        self.generic_test_images = self.coco_test["images"]
        self.generic_train_annotations = self.coco_train["annotations"]
        self.generic_test_annotations = self.coco_test["annotations"]

        self.meta["categories"] = self.coco_test["categories"][0]

        # to build maps for later analysis
        self._build_maps()

        print(f"Before checking trainset {self.meta['dataset_name']}")

        self.whether_anno_image_match(
            self.generic_train_images, self.generic_train_annotations
        )

        print(f"Before checking testset {self.meta['dataset_name']}")

        self.whether_anno_image_match(
            self.generic_test_images, self.generic_test_annotations
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


if __name__ == "__main__":
    dataset = MaDLCDataFrame("/mnt/md0/shaokai/daniel3mouse", "3mouse")
    dataset.summary()
