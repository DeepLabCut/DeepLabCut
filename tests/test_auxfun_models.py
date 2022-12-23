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


from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deeplabcut.utils.auxfun_models import MODELTYPE_FILEPATH_MAP, check_for_weights


class CheckForWeightsTestCase(unittest.TestCase):
    def test_filepaths_for_modeltypes(self):
        with TemporaryDirectory() as tmpdir:
            with patch(
                "deeplabcut.utils.auxfun_models.download_weights"
            ) as mocked_download:
                for modeltype, expected_path in MODELTYPE_FILEPATH_MAP.items():
                    actual_path, _ = check_for_weights(modeltype, Path(tmpdir), 1)
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
        actual_path, actual_num_shuffles = check_for_weights(
            "dummymodel", "nonexistentpath", 1
        )
        self.assertEqual(actual_path, "nonexistentpath")
        self.assertEqual(actual_num_shuffles, -1)
