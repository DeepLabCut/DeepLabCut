import deeplabcut

method = "CTD"

if method == "CTD":
    cfg = "/home/max/tmp/riken_bandy-ti-2025-05-06/config.yaml"
    images = [
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame343.png",
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame382.png",
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame653.png",
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame686.png"
    ]
    shuffle = 3
elif method == "TD":
    cfg = "/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/config.yaml"
    images = [
        "/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/labeled-data/output_clip/img003.png",
        "/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/labeled-data/output_clip/img055.png",
        "/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/labeled-data/output_clip/img132.png",
        "/home/max/Work/DeepLabCut-Projects/uk_first_results/uk_results/uk_har-maxim-2024-11-04/labeled-data/output_clip/img144.png"
    ]
    shuffle = 0
elif method == "BU":
    cfg = "/home/max/tmp/riken_bandy-ti-2025-05-06/config.yaml"
    images = [
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame343.png",
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame382.png",
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame653.png",
        "/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame686.png"
    ]
    shuffle = 1



deeplabcut.analyze_images(
    config=cfg,
    images=images,
    shuffle=shuffle,
    plotting=True,
)
