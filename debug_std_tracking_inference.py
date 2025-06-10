import deeplabcut.pose_estimation_pytorch as dlc_torch

config = "/home/max/tmp/CTD_debug_project/trimice-dlc-2021-06-22/config.yaml"

dlc_torch.analyze_videos(
    config=config,
    shuffle=11,
    videos=["/home/max/tmp/CTD_debug_project/trimice-dlc-2021-06-22/videos/trimouse_10frames.mp4"],
    overwrite=True,
    #ctd_tracking=True,
)