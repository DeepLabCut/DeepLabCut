"""TODO"""
import torch

import deeplabcut.pose_estimation_pytorch.models as dlc_models


def _get_keypoints(number_of_joints: int = 4, axis: int = 2):
    keypoints_torch = torch.Tensor(number_of_joints, axis)

    return keypoints_torch, number_of_joints


def test_generate_heatmaps():
    keypoints_torch, number_of_joints = _get_keypoints()
    image_size = (256, 256)
    sigma = 5
    heatmap_size = (64, 64)
    heatmaps = dlc_models._generate_heatmaps(
        keypoints_torch, heatmap_size, image_size, sigma=sigma
    )
    assert heatmaps.shape == (number_of_joints, heatmap_size[0], heatmap_size[1])


# Create fake config dict for testing purposes
def write_config(multianimal: bool) -> dict:
    cfg_file = {}
    if multianimal:  # parameters specific to multianimal project
        cfg_file["multianimalproject"] = multianimal
        cfg_file["identity"] = False
        cfg_file["individuals"] = ["individual1", "individual2", "individual3"]
        cfg_file["multianimalbodyparts"] = ["bodypart1", "bodypart2", "bodypart3"]
        cfg_file["uniquebodyparts"] = []
        cfg_file["bodyparts"] = "MULTI!"
        cfg_file["skeleton"] = [
            ["bodypart1", "bodypart2"],
            ["bodypart2", "bodypart3"],
            ["bodypart1", "bodypart3"],
        ]
        cfg_file["default_augmenter"] = "multi-animal-imgaug"
        cfg_file["default_net_type"] = "dlcrnet_ms5"
        cfg_file["default_track_method"] = "ellipse"
    else:
        cfg_file["multianimalproject"] = False
        cfg_file["bodyparts"] = ["bodypart1", "bodypart2", "bodypart3", "objectA"]
        cfg_file["skeleton"] = [["bodypart1", "bodypart2"], ["objectA", "bodypart3"]]
        cfg_file["default_augmenter"] = "default"
        cfg_file["default_net_type"] = "resnet_50"

    # common parameters:
    cfg_file["Task"] = "test"
    cfg_file["scorer"] = "experimenter"
    cfg_file["video_sets"] = "placeholder"
    cfg_file["project_path"] = "fake\\path"
    cfg_file["date"] = "Oct30"
    cfg_file["cropping"] = False
    cfg_file["start"] = 0
    cfg_file["stop"] = 1
    cfg_file["numframes2pick"] = 20
    cfg_file["TrainingFraction"] = [0.95]
    cfg_file["iteration"] = 0
    cfg_file["snapshotindex"] = -1
    cfg_file["x1"] = 0
    cfg_file["x2"] = 640
    cfg_file["y1"] = 277
    cfg_file["y2"] = 624
    cfg_file[
        "batch_size"
    ] = 8  # batch size during inference (video - analysis); see https://www.biorxiv.org/content/early/2018/10/30/457242
    cfg_file["corner2move2"] = (50, 50)
    cfg_file["move2corner"] = True
    cfg_file["skeleton_color"] = "black"
    cfg_file["pcutoff"] = 0.6
    cfg_file["dotsize"] = 12  # for plots size of dots
    cfg_file["alphavalue"] = 0.7  # for plots transparency of markers
    cfg_file["colormap"] = "rainbow"  # for plots type of colormap

    return cfg_file
