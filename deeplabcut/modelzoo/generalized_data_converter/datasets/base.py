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
import copy
import os
import warnings

import numpy as np

from deeplabcut.modelzoo.generalized_data_converter.conversion_table import (
    get_conversion_table,
)
from deeplabcut.modelzoo.generalized_data_converter.datasets.materialize import (
    mat_func_factory,
)


def raw_2_imagename_with_id(image):
    """
    raw image data has filename and id.
    we modify the imagename such that itis composed of
    both original imagename and image id
    """

    file_name = image["file_name"]
    image_name = file_name.split(os.sep)[-1]
    pre, suffix = image_name.split(".")
    image_id = image["id"]
    return f"{pre}_{image_id}.{suffix}"


def raw_2_imagename(image):
    """
    Only getting the imagename part from the image object
    """

    file_name = image["file_name"]
    image_name = file_name.split(os.sep)[-1]
    return image_name


class BasePoseDataset:
    """
    Dual representation of generic and raw data. For classes that inherits this class,
    the raw data is kept but generic data is populated so you have dual representation.
    """

    def __init__(self):
        # generic data is what all the manipulation is based on
        self.generic_train_images = []
        self.generic_test_images = []
        self.generic_train_annotations = []
        self.generic_test_annotations = []
        # These maps are very important for later analysis, including max_individuals
        # and trace back the original dataset etc.
        self.imageid2anno = {}
        self.dataset2images = {}
        self.imageid2filename = {}
        self.imageid2datasetname = {}
        self.datasetname2imageids = {}
        # meta keeps information for later analysis
        self.meta = {}
        # if conversion_table is None, dataset is not yet converted to super keypoints
        self.conversion_table = None

    def _build_maps(self):
        self.datasetname2imageids[self.meta["dataset_name"]] = set()

        total_annotations = (
            self.generic_train_annotations + self.generic_test_annotations
        )
        for anno in total_annotations:
            image_id = anno["image_id"]
            if image_id not in self.imageid2anno:
                self.imageid2anno[image_id] = []
            self.imageid2anno[image_id].append(anno)

        total_images = self.generic_train_images + self.generic_test_images
        for image in total_images:
            image_id = image["id"]
            self.imageid2datasetname[image_id] = self.meta["dataset_name"]
            file_name = image["file_name"]
            self.imageid2filename[image_id] = file_name
            self.datasetname2imageids[self.meta["dataset_name"]].add(image_id)

        # in DLC, even if you have more than one annotations in one image, it does not
        # mean it's a multi animal project
        max_num = 0
        for k in self.imageid2anno:
            max_num = max(len(self.imageid2anno[k]), max_num)

        self.meta["max_individuals"] = max_num
        self.meta["imageid2filename"] = self.imageid2filename

    def filter_by_pattern(self, pattern):

        keep_ids = []
        keep_train_images = []
        keep_test_images = []
        for img in self.generic_train_images + self.generic_test_images:
            print(img["file_name"])
            if pattern in img["file_name"]:

                image_id = img["id"]
                keep_ids.append(image_id)

        for image in self.generic_train_images:
            if image["id"] in keep_ids:
                keep_train_images.append(image["id"])

        self.generic_train_images = keep_train_images

        for image in self.generic_test_images:
            if image["id"] in keep_ids:
                keep_test_images.append(image["id"])

        self.generic_test_images = keep_test_images

        keep_train_annotations = []
        keep_test_annotations = []

        for anno in self.generic_train_annotations:
            if anno["image_id"] in keep_ids:
                keep_train_annotations.append(anno)

        self.generic_train_annotations = keep_train_annotations

        for anno in self.generic_test_annotations:
            if anno["image_id"] in keep_ids:
                keep_test_annotations.append(anno)

        self.generic_test_annotations = keep_test_annotations

    def summary(self):
        print(f'Summary of dataset {self.meta["dataset_name"]}')
        print("-------------")
        print(f'max num individuals  is {self.meta["max_individuals"]}')
        print(f"total keypoints : {len(self.meta['categories']['keypoints'])}")
        print(f"total train images : {len(self.generic_train_images)}")
        print(f"total test images : {len(self.generic_test_images)}")
        print(f"total train annotations : {len(self.generic_train_annotations)}")
        print(f"total test annotations : {len(self.generic_test_annotations)}")
        print("-------------")

    def populate_generic(self):
        raise NotImplementedError("Must implement this function")

    def materialize(
        self, proj_root, framework="coco", deepcopy=False, append_image_id=True
    ):
        mat_func = mat_func_factory(framework)
        self.meta["mat_datasets"] = {self.meta["dataset_name"]: self}
        self.meta["imageid2datasetname"] = self.imageid2datasetname
        mat_func(
            proj_root,
            self.generic_train_images,
            self.generic_test_images,
            self.generic_train_annotations,
            self.generic_test_annotations,
            self.meta,
            deepcopy=deepcopy,
            append_image_id=append_image_id,
        )

    def whether_anno_image_match(self, images, annotations):
        """
        Every image id should be annotated at least once
        There should not be any image that is not being annotated
        There should not be any annotation for beyond the set of given images
        """

        image_ids = set([image["id"] for image in images])

        annotation_image_ids = set([anno["image_id"] for anno in annotations])

        if image_ids != annotation_image_ids:
            print("images-annotations", image_ids - annotation_image_ids)
            print("len(images-annotatinos)", len(image_ids - annotation_image_ids))
            print("annotations-images", annotation_image_ids - image_ids)
            print("len(annotations-images)", len(annotation_image_ids - image_ids))
            warnings.warn("annotation and image ids do not match")

    def get_keypoints(self):
        # TODO make sure it's always one element in a list
        return self.meta["categories"]["keypoints"]

    def _proj(self, annotations, conversion_table):

        keypoints = self.get_keypoints()

        kpt2index = {kpt: kpt_id for kpt_id, kpt in enumerate(keypoints)}

        ret = []

        master2src = {}
        for kpt in keypoints:
            conv_kpt = conversion_table.convert(kpt)
            # sometimes a keypoint might not find its corresponding one from mastername
            if conv_kpt is not None:
                master2src[conv_kpt] = kpt

        master_keypoints = conversion_table.master_keypoints

        # need to change this in meta

        for anno in annotations:
            try:
                kpts = anno["keypoints"]
            except:
                print(anno)

            new_kpts = np.zeros(len(master_keypoints) * 3)
            new_num_kpts = len(master_keypoints)

            for master_kpt_id, master_kpt_name in enumerate(master_keypoints):
                # check whether the dataset has the corresponding keypoint
                if master_kpt_name not in master2src:
                    new_kpts[master_kpt_id * 3 : master_kpt_id * 3 + 3] = -1
                    continue

                src_kpt_name = master2src[master_kpt_name]
                src_kpt_id = kpt2index[src_kpt_name]
                new_kpts[master_kpt_id * 3 : master_kpt_id * 3 + 3] = kpts[
                    src_kpt_id * 3 : src_kpt_id * 3 + 3
                ]

            # skipping empty frames after conversion
            new_anno = copy.deepcopy(anno)
            new_anno["keypoints"] = new_kpts
            new_anno["num_keypoints"] = new_num_kpts
            ret.append(new_anno)

        return ret

    def adjust_bbox_and_area(self):
        """Called during conversion.

        This is to remove the impact of keypoints that are potentially environmental
        keypoints to the bbox and area calculation.
        """
        from .utils import calc_bboxes_from_keypoints

        for annotation in (
            self.generic_train_annotations + self.generic_test_annotations
        ):
            keypoints = annotation["keypoints"]
            bbox_margin = 20

            num_kpts = annotation["num_keypoints"]

            keypoints = np.array(keypoints).reshape((num_kpts, 3))

            mask = keypoints[:, 0] > 0
            keypoints = keypoints[mask]

            if keypoints.shape[0] == 0:
                continue

            xmin, ymin, xmax, ymax = calc_bboxes_from_keypoints(
                [keypoints],
                slack=bbox_margin,
                clip=True,
            )[0][:4]

            w = xmax - xmin
            h = ymax - ymin
            area = w * h
            bbox = np.nan_to_num([xmin, ymin, w, h])

            if "bbox" not in annotation:
                annotation["bbox"] = bbox
            if "area" not in annotation:
                annotation["area"] = area

    def project_with_conversion_table(self, table_path="", table_dict=None):
        """
        Replace the generic annotations with those that are in superset keypoint space

        """
        print(f'Converting {self.meta["dataset_name"]}')

        keypoints = self.get_keypoints()

        self.conversion_table = get_conversion_table(
            keypoints=keypoints, table_path=table_path, table_dict=table_dict
        )

        self.generic_train_annotations = self._proj(
            self.generic_train_annotations, self.conversion_table
        )

        self.generic_test_annotations = self._proj(
            self.generic_test_annotations, self.conversion_table
        )

        # all category id fixed to 1. So that it does not conflict with the background
        # category id
        for anno in self.generic_train_annotations + self.generic_test_annotations:
            anno["category_id"] = 1

        for img in self.generic_train_images + self.generic_test_images:
            img["source_dataset"] = self.meta["dataset_name"]

        self.adjust_bbox_and_area()
        self.meta["categories"]["keypoints"] = self.conversion_table.master_keypoints
        self.meta["categories"]["supercategory"] = "animal"
        self.meta["categories"]["name"] = "superanimal"

        # category id fixed to be 1, to avoid to conflict with background category id
        self.meta["categories"]["id"] = 1
