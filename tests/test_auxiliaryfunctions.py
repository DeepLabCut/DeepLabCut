from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import SUPPORTED_VIDEOS


def test_get_list_of_videos(tmpdir_factory):
    fake_folder = tmpdir_factory.mktemp('videos')
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
        str(fake_folder), videotype="",
    )
    assert len(videos) == n_ext

    # A list of extensions can also be passed in
    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder), videotype=SUPPORTED_VIDEOS,
    )
    assert len(videos) == n_ext

    for ext in SUPPORTED_VIDEOS:
        videos = auxiliaryfunctions.get_list_of_videos(
            str(fake_folder), videotype=ext,
        )
        assert len(videos) == 1

    videos = auxiliaryfunctions.get_list_of_videos(
        str(fake_folder), videotype="unknown",
    )
    assert not len(videos)

    videos = auxiliaryfunctions.get_list_of_videos(
        fake_videos, videotype="",
    )
    assert len(videos) == n_ext

    for video in fake_videos:
        videos = auxiliaryfunctions.get_list_of_videos(
            [video], videotype=""
        )
        assert len(videos) == 1

    for ext in SUPPORTED_VIDEOS:
        videos = auxiliaryfunctions.get_list_of_videos(
            fake_videos, videotype=ext,
        )
        assert len(videos) == 1
