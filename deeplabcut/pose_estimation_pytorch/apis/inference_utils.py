import torch
import numpy as np
from skimage.transform import resize
from deeplabcut.pose_estimation_pytorch.models import DETECTORS, PREDICTORS, PoseModel
from deeplabcut.pose_estimation_pytorch.models.predictors import BasePredictor
from deeplabcut.pose_estimation_pytorch.models.detectors import BaseDetector

from typing import Union


def get_predictions_bottom_up(
    model: PoseModel, predictor: BasePredictor, images: torch.Tensor
) -> np.array:
    """Gets the predicted coordinates tensor for a bottom_up approach

    Model and images should already be on the same device

    Args:
        model (PoseModel): bottom-up model
        predictor (BasePredictor): predictor used to regress keypoints coordinates and scores
        images (torch.Tensor): input images (should already be normalised and formatted if needed),
                                shape (batch_size, 3, height, width)

    Returns:
        np.array: predictions tensor of shape (batch_size, num_animals, num_keypoints, 3)
    """

    output = model(images)
    shape_image = images.shape
    scale_factor = (
        shape_image[2] / output[0].shape[2],
        shape_image[3] / output[0].shape[3],
    )
    predictions = predictor(output, scale_factor)

    return predictions.cpu().numpy()


def get_predictions_top_down(
    detector: BaseDetector,
    top_down_predictor: BasePredictor,
    model: PoseModel,
    pose_predictor: BasePredictor,
    images: torch.Tensor,
    max_num_animals: int,
    num_keypoints: int,
    device: Union[torch.device, str],
) -> np.array:
    """
    TODO probably quite bad design, most arguments could be stored somewhere else
    Gets the predicted coordinates tensor for a bottom_up approach

    Detector, Model and images should already be on the same device

    Args:
        detector (BaseDetector): detector used to detect bboxes, should be in eval mode
        top_down_predictor (BasePredictor): Given the bboxes and the cropped keypoints coordinates, outputs the regressed keypoints
        model (PoseModel): pose model
        pose_predictor (BasePredictor): predictor used to regress keypoints coordinates and scores in the cropped images
        images (torch.Tensor): input images (should already be normalised and formatted if needed),
                                shape (batch_size, 3, height, width)
        max_num_animals (int) : maximum number of animals to predict
        num_keypoints (int) : number of keypoints per animal in the dataset
        device (Union[torch.device, str]): device everything should be on

    Returns:
        np.array: predictions tensor of shape (batch_size, num_animals, num_keypoints, 3)
    """
    batch_size = images.shape[0]

    output_detector = detector(images)

    boxes = torch.zeros((batch_size, max_num_animals, 4))
    for b, item in enumerate(output_detector):
        boxes[b][: min(max_num_animals, len(item["boxes"]))] = item["boxes"][
            :max_num_animals
        ]  # Boxes should be sorted by scores, only keep the maximum number allowed
    boxes = boxes.int()
    cropped_kpts_total = torch.full(
        (batch_size, max_num_animals, num_keypoints, 3), -1.0
    )

    for b in range(batch_size):
        for j, box in enumerate(boxes[b]):
            if (box == 0.0).all():
                continue
            cropped_image = (
                images[b][:, box[1] : box[3], box[0] : box[2]]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )  # needs to be (h,w,c) for resizing
            cropped_image = resize(cropped_image, (256, 256))  # TODO: hardcoded for now
            cropped_image = (
                torch.tensor(cropped_image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            )
            heatmaps = model(cropped_image)

            scale_factors_cropped = (
                cropped_image.shape[2] / heatmaps[0].shape[2],
                cropped_image.shape[3] / heatmaps[0].shape[3],
            )

            cropped_kpts = pose_predictor(heatmaps, scale_factors_cropped)
            cropped_kpts_total[b, j, :] = cropped_kpts[0, 0]

    final_predictions = top_down_predictor(boxes, cropped_kpts_total)

    return final_predictions.cpu().numpy()
