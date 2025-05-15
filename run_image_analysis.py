import deeplabcut

cfg = "/home/max/tmp/riken_bandy-ti-2025-05-06/config.yaml"

deeplabcut.analyze_images(
    config=cfg,
    images=["/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/frame343.png"],
    shuffle=3,
    destfolder="/home/max/tmp/riken_bandy-ti-2025-05-06/images_inference/",
)
