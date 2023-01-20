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

    print("adaptation training for superanimal_topviewmouse")

    superanimal_name = "superanimal_topviewmouse"
    videotype = ".mp4"
    scale_list = [200, 300, 400]
    deeplabcut.video_inference_superanimal(
        [video],
        superanimal_name,
        videotype=".mp4",
        video_adapt=True,
        scale_list=scale_list,
        pcutoff=0.1,
    )

    print("adaptation training for superanimal_quadruped")

    superanimal_name = "superanimal_quadruped"
    deeplabcut.video_inference_superanimal(
        [video],
        superanimal_name,
        videotype=".mp4",
        video_adapt=True,
        scale_list=scale_list,
        pcutoff=0.3,
    )
