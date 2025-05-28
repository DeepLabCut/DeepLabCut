import deeplabcut

cfg = "/home/max/tmp/riken_bandy-ti-2025-05-06/config.yaml"

deeplabcut.analyze_videos(
    config=cfg,
    videos=["/home/max/tmp/riken_bandy-ti-2025-05-06/videos/video.mp4"],
    shuffle=3,
    destfolder="/home/max/tmp/riken_bandy-ti-2025-05-06/videos/",
)
