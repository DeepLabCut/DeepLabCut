"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deeplabcut.utils.auxfun_models import MODELTYPE_FILEPATH_MAP, Check4weights


class Check4WeightsTestCase(unittest.TestCase):
    def test_filepaths_for_modeltypes(self):
        with TemporaryDirectory() as tmpdir:
            with patch("deeplabcut.utils.auxfun_models.Downloadweights") as mocked_download:
                for modeltype, expected_path in MODELTYPE_FILEPATH_MAP.items():
                    actual_path, _ = Check4weights(modeltype, Path(tmpdir), 1)
                self.assertIn(str(expected_path), actual_path)
                if "efficientnet" in modeltype:
                    mocked_download.assert_called_with(
                        modeltype, tmpdir / expected_path.parent
                    )
                else:
                    mocked_download.assert_called_with(
                        modeltype, tmpdir / expected_path
                    )

    def test_bad_modeltype(self):
        actual_path, actual_num_shuffles = Check4weights(
            "dummymodel", "nonexistentpath", 1
        )
        self.assertEqual(actual_path, "nonexistentpath")
        self.assertEqual(actual_num_shuffles, -1)
