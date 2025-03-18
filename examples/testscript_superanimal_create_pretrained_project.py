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
"""
Testscript for creating a pretrained project from a super animal model

"""
import glob
import shutil
from pathlib import Path

import deeplabcut

if __name__ == "__main__":
    superanimal_name = "superanimal_quadruped"
    working_dir = Path(__file__).resolve().parent
    video_dir = working_dir / "openfield-Pranav-2018-10-30/videos/m3v1mp4.mp4"
    project_name = "pretrained"

    deeplabcut.create_pretrained_project(
        project_name,
        "max",
        [str(video_dir)],
        engine=deeplabcut.Engine.PYTORCH,
    )

    dirs_to_delete = glob.glob(f"{working_dir}/{project_name}*")

    # Delete directories
    for directory in dirs_to_delete:
        shutil.rmtree(directory)

    print("Test passed!")
