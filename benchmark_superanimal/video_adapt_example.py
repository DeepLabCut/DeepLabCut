import deeplabcut.modelzoo.video_inference as modelzoo


def main():
    modelzoo.video_inference_superanimal(
        videos=["/mnt/md0/shaokai/DLCdev/3mice_video1_short.mp4"],
        superanimal_name="superanimal_topviewmouse_hrnetw32",
        video_adapt=True,
        max_individuals=3,
        pseudo_threshold=0.1,
        bbox_threshold=0.9,
        detector_epochs=4,
        pose_epochs=4,
    )


if __name__ == "__main__":
    main()
