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

import numpy as np
import pandas as pd


class ConversionTableFromDict:
    def __init__(self, raw_table_dict):
        self.table_dict = raw_table_dict["conversion_table"]
        self.master_keypoints = raw_table_dict["master_keypoints"]

    def convert(self, kpt):
        if kpt not in self.table_dict:
            warnings.warn(
                f"{kpt} is defined in src space but not appeared in the conversion table"
            )
            return None
        else:
            return self.table_dict[kpt]


class ConversionTableFromCSV:
    """
    Base class only reads the table
    """

    def __init__(self, src_keypoints, table_path):
        self.table_path = table_path

        # sep removes leading and tailing white space
        df = pd.read_csv(table_path, sep=r"\s*,\s*")

        df.dropna(inplace=True, how="all")
        # drop the row is MasterName has nan in the row
        df = df.dropna(subset=["MasterName"])

        self.df = df

        self.src_keypoints = src_keypoints

        kpt_list = df.to_numpy()

        self.lookup_set = []

        for i in range(len(kpt_list)):
            kpts = np.array(kpt_list[i])
            # remove nan

            kpt_alias = set(kpts)

            for k in list(kpt_alias):
                if type(k) != str:
                    kpt_alias.remove(k)

            self.lookup_set.append(kpt_alias)

        target_keypoints = df["MasterName"].values

        # target_keypoints = target_keypoints[~np.isnan(target_keypoints.values)]

        self.master_keypoints = target_keypoints

        # paired when they both exist

        # following assumes that either it's 1vs.1 from src to target
        # or 1 vs. 0
        # it could be 1 vs. 2 in horse data
        self.table = {}
        for src_kpt in src_keypoints:
            for target_kpt in target_keypoints:

                src_kpt_id = self._search(src_kpt)
                target_kpt_id = self._search(target_kpt)

                if src_kpt_id == -1 or target_kpt_id == -1:
                    # if any one of them not exist in the set
                    # skip
                    continue
                if src_kpt_id == target_kpt_id:
                    self.table[src_kpt] = target_kpt

        self.check_inclusion()

    def _search(self, key):
        """
        return -1 if not found
        return kpt id if found

        """
        # [TODO] if it can be mapped to two, I can randomly return one
        for kpt_id in range(len(self.lookup_set)):
            if key in self.lookup_set[kpt_id]:
                return kpt_id
        return -1

    def check_inclusion(self):
        """
        check if conversion table covers
        every keypoint contained in src proj

        """
        count = 0
        print("src keypoints")
        print(self.src_keypoints)
        for kpt in self.src_keypoints:
            index = self._search(kpt)
            if index == -1:
                pass
            else:
                count += 1
        print(f"{count}/{len(self.src_keypoints)} keypoints will be converted")

    def convert(self, kpt):
        if kpt not in self.table:
            warnings.warn(
                f"{kpt} is defined in src space but not appeared in the conversion table"
            )
            return None
        else:
            return self.table[kpt]

    def get_subset(self, labname=""):

        bodyparts = self.df[labname]

        super_bodyparts = self.df["MasterName"]

        ret = []

        for bodypart in bodyparts:
            if bodypart in self.table:
                ret.append(self.table[bodypart])

        return ret


def get_conversion_table(keypoints=None, table_path="", table_dict=None):
    if table_path is not None and keypoints is not None:
        return ConversionTableFromCSV(keypoints, table_path)
    elif table_dict:
        return ConversionTableFromDict(table_dict)
    else:
        raise NotImplementedError("not supported")
