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
import os
import pandas as pd
from conftest import TEST_DATA_DIR
from deeplabcut.utils import conversioncode


def test_guarantee_multiindex_rows():
    df_unix = pd.read_hdf(os.path.join(TEST_DATA_DIR, "trimouse_calib.h5"))
    df_posix = df_unix.copy()
    df_posix.index = df_posix.index.str.replace("/", "\\")
    nrows = len(df_unix)
    for df in (df_unix, df_posix):
        conversioncode.guarantee_multiindex_rows(df)
        assert isinstance(df.index, pd.MultiIndex)
        assert len(df) == nrows
        assert df.index.nlevels == 3
        assert all(df.index.get_level_values(0) == "labeled-data")
        assert all(img.endswith(".png") for img in df.index.get_level_values(2))
