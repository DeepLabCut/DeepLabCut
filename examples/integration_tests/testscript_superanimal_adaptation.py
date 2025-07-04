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
Test script for super animal adaptation
"""
import deeplabcut
import os


if __name__ == "__main__":
    basepath = os.path.dirname(os.path.realpath(__file__))
    videoname = "m3v1mp4"
    video = os.path.join(
        basepath, "openfield-Pranav-2018-10-30", "videos", videoname + ".mp4"
    )
    video = deeplabcut.ShortenVideo(
        video,
        start="00:00:00",
        stop="00:00:01",
        outsuffix="short",
    )

    print("adaptation training for superanimal_topviewmouse")

    superanimal_name = "superanimal_topviewmouse"
    videotype = ".mp4"
    scale_list = [200, 300, 400]
    deeplabcut.video_inference_superanimal(
        [video],
        superanimal_name,
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        videotype=".mp4",
        video_adapt=True,
        scale_list=scale_list,
        pcutoff=0.1,
        adapt_iterations=50,
    )
