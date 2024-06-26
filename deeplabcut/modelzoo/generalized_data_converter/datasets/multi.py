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
import warnings

from deeplabcut.modelzoo.generalized_data_converter.datasets.materialize import (
    mat_func_factory,
)


class MultiSourceDataset:
    """
    Parameters:
    iid_ood_split:  {'iid' : ['dataset1', 'dataset2'],
                     'ood' : ['dataset3', 'dataset4'] }



    """

    def __init__(self, dataset_name, datasets, table_path):
        self.datasets = datasets
        #
        self.name2genericdataset = {}

        # useful maps for analysis
        self.imageid2filename = {}
        self.imageid2datasetname = {}
        self.datasetname2imageids = {}
        #
        self.dataset_name = dataset_name

        names = []
        for dataset in datasets:

            # Must project datasets to same keypoint space before merging
            if table_path != None:
                dataset.project_with_conversion_table(table_path)
            name = dataset.meta["dataset_name"]
            names.append(name)
            self.name2genericdataset[name] = dataset

        self.meta = {}
        self.meta["dataset_name"] = dataset_name
        # after conversion, all datasets have same categories
        self.meta["categories"] = dataset.meta["categories"]

        # map id from local scope to global
        self._update_imgids()

        (
            self.train_images,
            self.test_images,
            self.train_annotations,
            self.test_annotations,
        ) = self._merge_datasets(self.name2genericdataset)
        self.meta["name2genericdataset"] = self.name2genericdataset

        # only build maps after images are merged and ids are in global scope
        self._build_maps()

    def summary(self):
        print(f"Summary of dataset {self.dataset_name}")
        print("Decomposition of multi source datasets:")
        for dataset_name, dataset in self.name2genericdataset.items():
            n_images = len(dataset.generic_train_images) + len(
                dataset.generic_test_images
            )
            n_annotations = len(dataset.generic_train_annotations) + len(
                dataset.generic_test_annotations
            )
            print(f"{dataset_name} has {n_images} images, {n_annotations} annotations")

        print(f"total train images : {len(self.train_images)}")
        print(f"total test images : {len(self.test_images)}")

    def _build_maps(self):

        # shared by both scenarios

        species_set = set()
        for dataset_name, dataset in self.name2genericdataset.items():
            # I could of course do this during merge to save compute, but doing it here makes the logic cleaner to understand
            total_images = dataset.generic_train_images + dataset.generic_test_images

            for image in total_images:
                image_id = image["id"]
                image_name = image["file_name"]
                self.imageid2filename[image_id] = image_name

                self.imageid2datasetname[image_id] = dataset_name

                if dataset_name == "AwA-Pose":
                    species_set.add(image_name.split("/")[-1].split("_")[0])
        self.meta["imageid2datasetname"] = self.imageid2datasetname

        max_num = 0
        for dataset_name, dataset in self.name2genericdataset.items():
            max_num = max(max_num, dataset.meta["max_individuals"])
        self.meta["max_individuals"] = max_num
        dataset_name = self.meta["dataset_name"]
        print(f"Max individual in {dataset_name} is {max_num}")

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
            print("annotations-images", annotation_image_ids - image_ids)

            warnings.warn("annotation and image ids do not match")

            # This is constrain is too hard
            # assert len(annotation_image_ids - image_ids) == 0, "You can't have annotation on non-existed images"

    def _update_imgids(self):
        """
        update image ids for both image and annotation

        If datasets are merged, their image id, annotation id will conflict because they are defined within their own local scope. Therefore, we will need to put these ids in the global scope

        """

        from collections import defaultdict

        dataset_id_pool = defaultdict(set)
        all_datasets = self.name2genericdataset.values()

        total_number_images = 0
        total_number_annotations = 0
        for dataset in all_datasets:
            total_number_images += len(dataset.generic_train_images) + len(
                dataset.generic_test_images
            )
            total_number_annotations += len(dataset.generic_train_annotations) + len(
                dataset.generic_test_annotations
            )

        global_image_id_pool = set(range(total_number_images))
        global_annotation_id_pool = set(range(total_number_annotations))

        for dataset_name, dataset in self.name2genericdataset.items():

            local_image_id_map = defaultdict(int)
            local_anno_id_map = defaultdict(int)

            traintest_images = (
                dataset.generic_train_images + dataset.generic_test_images
            )
            traintest_annotations = (
                dataset.generic_train_annotations + dataset.generic_test_annotations
            )

            for img in traintest_images:

                new_image_id = global_image_id_pool.pop()
                local_image_id_map[img["id"]] = new_image_id
                img["id"] = new_image_id
                dataset_id_pool[dataset_name].add(img["id"])

            for anno in traintest_annotations:
                anno["image_id"] = local_image_id_map[anno["image_id"]]
                new_anno_id = global_annotation_id_pool.pop()
                local_anno_id_map[anno["id"]] = new_anno_id
                anno["id"] = new_anno_id

            self.whether_anno_image_match(traintest_images, traintest_annotations)

        from functools import reduce

        count = 0
        for k, v in dataset_id_pool.items():
            count += len(v)
        print("size of the summation", count)
        union = reduce(set.union, dataset_id_pool.values())
        print("size of the union", len(union))

    def _merge_datasets(self, name2dataset):
        """
        Merged datasets into common list

        # only do this when iid/ood split is done

        """

        merged_train_images = []
        merged_test_images = []
        merged_train_annotations = []
        merged_test_annotations = []

        for dataset_name, dataset in name2dataset.items():

            train_images = dataset.generic_train_images
            test_images = dataset.generic_test_images
            train_annotations = dataset.generic_train_annotations
            test_annotations = dataset.generic_test_annotations

            merged_train_images.extend(train_images)
            merged_test_images.extend(test_images)
            merged_train_annotations.extend(train_annotations)
            merged_test_annotations.extend(test_annotations)

        print("Checking merged dataset")

        merged_traintest_images = merged_train_images + merged_test_images
        merged_traintest_annotations = (
            merged_train_annotations + merged_test_annotations
        )

        self.whether_anno_image_match(
            merged_traintest_images, merged_traintest_annotations
        )

        return (
            merged_train_images,
            merged_test_images,
            merged_train_annotations,
            merged_test_annotations,
        )

    def __eq__(self, other_dataset):

        if isinstance(other_dataset, BasePoseDataset):

            train_images1 = set(map(raw_2_imagename_with_id, self.train_images))
            train_images2 = set(
                map(raw_2_imagename, other_dataset.generic_train_images)
            )

            test_images1 = set(map(raw_2_imagename_with_id, self.test_images))
            test_images2 = set(map(raw_2_imagename, other_dataset.generic_test_images))
            if train_images1 == train_images2 and test_images1 == test_images2:
                print(
                    f'dataset {self.meta["dataset_name"]} and {other_dataset.meta["dataset_name"]} are equivalent'
                )
                return True
            else:
                print(
                    f'dataset {self.meta["dataset_name"]} and {other_dataset.meta["dataset_name"]} are NOT equivalent'
                )
                return False

        else:
            return NotImplementedError("Not existed")

    def materialize(
        self,
        proj_root,
        framework="coco",
        train_all=False,
        deepcopy=False,
        full_image_path=True,
    ):

        # can't be set to true at the same time. This will cause bugs
        assert sum([train_all, full_image_path]) != 2

        mat_func = mat_func_factory(framework)

        self.meta["mat_datasets"] = self.name2genericdataset

        if train_all:
            # for pretrian phase, we can just train everything including the test part
            self.train_images += self.test_images
            self.train_annotations += self.test_annotations

        mat_func(
            proj_root,
            self.train_images,
            self.test_images,
            self.train_annotations,
            self.test_annotations,
            self.meta,
            deepcopy=deepcopy,
            full_image_path=full_image_path,
        )
