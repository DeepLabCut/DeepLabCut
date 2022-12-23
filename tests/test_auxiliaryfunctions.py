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
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS


def test_find_analyzed_data(tmpdir_factory):
    import os
    import pytest

    fake_folder = tmpdir_factory.mktemp("videos")
    SUPPORTED_VIDEOS = ["avi"]
    n_ext = len(SUPPORTED_VIDEOS)

    SCORER = "DLC_dlcrnetms5_multi_mouseApr11shuffle1_5"
    WRONG_SCORER = "DLC_dlcrnetms5_multi_mouseApr11shuffle3_5"

    def _create_fake_file(filename):
        path = str(fake_folder.join(filename))
        with open(path, "w") as f:
            f.write("")
        return path

    for ind, ext in enumerate(SUPPORTED_VIDEOS):
        vname = "video" + str(ind)
        _ = _create_fake_file(vname + "." + ext)
        _ = _create_fake_file(vname + SCORER + ".pickle")
        _ = _create_fake_file(vname + SCORER + ".h5")

    for ind, ext in enumerate(SUPPORTED_VIDEOS):
        # test if existing models are found:
        assert auxiliaryfunctions.find_analyzed_data(
            fake_folder, "video" + str(ind), SCORER
        )

        # Test if nonexisting models are not found
        with pytest.raises(FileNotFoundError):
            auxiliaryfunctions.find_analyzed_data(
                fake_folder, "video" + str(ind), WRONG_SCORER
            )

        with pytest.raises(FileNotFoundError):
            auxiliaryfunctions.find_analyzed_data(
                fake_folder, "video" + str(ind), SCORER, filtered=True
            )


def test_get_list_of_videos(tmpdir_factory):
    fake_folder = tmpdir_factory.mktemp("videos")
    n_ext = len(SUPPORTED_VIDEOS)

    def _create_fake_file(filename):
        path = str(fake_folder.join(filename))
        with open(path, "w") as f:
            f.write("")
        return path

    fake_videos = []
    for ext in SUPPORTED_VIDEOS:
        path = _create_fake_file(f"fake.{ext}")
        fake_videos.append(path)

    # Add some other office files:
    path = _create_fake_file("fake.xls")
    path = _create_fake_file("fake.pptx")

    # Add a .pickle and .h5 files
    _ = _create_fake_file("fake.pickle")
    _ = _create_fake_file("fake.h5")

    # By default, all videos with common extensions are taken from a directory
    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder),
        videotype="",
    )
    assert len(videos) == n_ext

    # A list of extensions can also be passed in
    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder),
        videotype=SUPPORTED_VIDEOS,
    )
    assert len(videos) == n_ext

    for ext in SUPPORTED_VIDEOS:
        videos = auxiliaryfunctions.get_list_of_videos(
            str(fake_folder),
            videotype=ext,
        )
        assert len(videos) == 1

    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder),
        videotype="unknown",
    )
    assert not len(videos)

    videos = auxiliaryfunctions.get_list_of_videos(
        fake_videos,
        videotype="",
    )
    assert len(videos) == n_ext

    for video in fake_videos:
        videos = auxiliaryfunctions.get_list_of_videos([video], videotype="")
        assert len(videos) == 1

    for ext in SUPPORTED_VIDEOS:
        videos = auxiliaryfunctions.get_list_of_videos(
            fake_videos,
            videotype=ext,
        )
        assert len(videos) == 1
