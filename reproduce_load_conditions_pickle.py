import deeplabcut

config_path = "/home/max/tmp/CTD_debug_project/synthetic-data-niels-multi-animal/config.yaml"
videos = ["/home/max/tmp/CTD_debug_project/synthetic-data-niels-multi-animal/videos/video.mp4"]
trainset_index = 0
shuffle_index = 2
device="cpu"

print(f"Analyzing videos for shuffle {shuffle_index}")
video_kwargs = dict(
    videos=videos, shuffle=shuffle_index, trainingsetindex=trainset_index
)
# deeplabcut.analyze_videos(
#     str(config_path), **video_kwargs, device=device, auto_track=True
# )

deeplabcut.create_video_with_all_detections(
    str(config_path), **video_kwargs
)

deeplabcut.create_labeled_video(str(config_path), **video_kwargs)