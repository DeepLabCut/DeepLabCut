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
Testscript for super animal inference

"""
import deeplabcut
import os


if __name__ == "__main__":
    basepath = os.path.dirname(os.path.realpath(__file__))
    videoname = "reachingvideo1"
    video = [
        os.path.join(
            basepath, "Reaching-Mackenzie-2018-08-30", "videos", videoname + ".avi"
        )
    ]

    print("testing superanimal_topviewmouse")
    superanimal_name = "superanimal_topviewmouse"
    scale_list = [200, 300, 400]
    deeplabcut.video_inference_superanimal(
        video,
        superanimal_name,
        videotype=".avi",
        scale_list=scale_list,
    )

    print("testing superanimal_quadruped")
    superanimal_name = "superanimal_quadruped"
    deeplabcut.video_inference_superanimal(
        video,
        superanimal_name,
        videotype=".avi",
        scale_list=scale_list,
    )
