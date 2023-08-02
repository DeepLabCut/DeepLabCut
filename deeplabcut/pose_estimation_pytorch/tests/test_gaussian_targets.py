import pytest
import torch
from deeplabcut.pose_estimation_pytorch.models.target_generators import gaussian_targets


@pytest.mark.parametrize(
    "batch_size, num_keypoints, image_size",
    [(2, 2, (64, 64)), (1, 5, (48, 64)), (15, 50, (64, 48))],
)
def test_gaussian_target_generation(
    batch_size: int, num_keypoints: int, image_size: tuple, num_animals=1
):
    # generate annotations
    annotations = {
        "keypoints": torch.randint(
            1, min(image_size), (batch_size, num_animals, num_keypoints, 2)
        )
    }  # batch size, num animals, num keypoints, 2 for x,y
    # generate predictions
    prediction = [
        torch.rand((batch_size, num_keypoints, image_size[0], image_size[1]))
    ]  # batch size, num keypoints , imageh, imagew

    # generate heatmap
    output = gaussian_targets.GaussianGenerator(5.0, num_keypoints, 17)
    output = torch.tensor(
        output(annotations, prediction, image_size)["heatmaps"].reshape(
            batch_size, num_keypoints, image_size[0] * image_size[1]
        )
    )

    # get coords of max value of the heatmap
    gaus_max = torch.argmax(output, dim=2)

    # get unraveled coords
    x = gaus_max % image_size[1]
    y = gaus_max // image_size[1]

    # get heatmap center tensor
    predict_kp = torch.stack((x, y), dim=-1)
    # Remove num_animals dimension - only one animal is supported
    annotations["keypoints"] = torch.squeeze(annotations["keypoints"], dim=1)

    # compare heatmap center to annotation
    assert torch.eq(annotations["keypoints"], predict_kp).all().item()
