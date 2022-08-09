class DLCParams:
    VIDEOTYPES = [
        "",
        "avi",
        "mp4",
        "mov",
    ]

    NNETS = [
        "dlcrnet_ms5",
        "resnet_50",
        "resnet_101",
        "resnet_152",
        "mobilenet_v2_1.0",
        "mobilenet_v2_0.75",
        "mobilenet_v2_0.5",
        "mobilenet_v2_0.35",
        "efficientnet-b0",
        "efficientnet-b3",
        "efficientnet-b6",
    ]

    IMAGE_AUGMENTERS = ["default", "tensorpack", "imgaug"]

    FRAME_EXTRACTION_ALGORITHMS = ["kmeans", "uniform"]

    OUTLIER_EXTRACTION_ALGORITHMS = ["jump", "fitting", "uncertain", "manual"]

    TRACKERS = ["ellipse", "box", "skeleton"]
