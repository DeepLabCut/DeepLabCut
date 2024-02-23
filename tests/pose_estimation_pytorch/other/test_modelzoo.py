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

import pytest

from deeplabcut.modelzoo.video_inference import video_inference_superanimal
from deeplabcut.utils import auxiliaryfunctions

examples_folder = os.path.join(
    auxiliaryfunctions.get_deeplabcut_path(),
    "modelzoo",
    "examples",
)

# requires videos to be in the examples folder
@pytest.mark.skip
@pytest.mark.parametrize(
    "video_paths, superanimal_name",
    [
        (f"{examples_folder}/black_dog.mp4", "superanimal_quadruped"),
        (f"{examples_folder}/black_dog.mp4", "superanimal_quadruped_hrnetw32"),
        (f"{examples_folder}/swear_mouse_tiny.mp4", "superanimal_topviewmouse"),
        (
            f"{examples_folder}/swear_mouse_tiny.mp4",
            "superanimal_topviewmouse_hrnetw32",
        ),
    ],
)
def test_video_inference_saves_file(video_paths, superanimal_name):
    video_inference_superanimal(
        video_paths,
        superanimal_name=superanimal_name,
    )
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    for video_path in video_paths:
        output_path = video_path.replace(".mp4", f"_labeled.mp4")
        assert os.path.exists(output_path), "Output video file does not exist"

        assert os.stat(output_path).st_size > 0, "Output video file is empty"
