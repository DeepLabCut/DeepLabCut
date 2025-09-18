#!/usr/bin/env python3
"""
Test script for superanimal_humanbody with torchvision detector
"""

from deeplabcut.pose_estimation_pytorch.apis.utils import (
    TORCHVISION_DETECTORS,
    get_filtered_coco_detector_inference_runner,
)
from deeplabcut.pose_estimation_pytorch.models.detectors.filtered_detector import (
    FilteredDetector,
)
from deeplabcut.pose_estimation_pytorch.modelzoo import load_super_animal_config


def test_torchvision_detector():
    """Test that the torchvision detector works with superanimal_humanbody"""
    for detector_name in TORCHVISION_DETECTORS:

        # Load the superanimal_humanbody config
        superanimal_config = load_super_animal_config(
            super_animal="superanimal_humanbody",
            model_name="rtmpose_x",
            detector_name=detector_name,
        )
        print("Config loaded successfully!")

        # Test loading the torchvision detector directly
        print("\nTesting torchvision detector loading...")
        entry = TORCHVISION_DETECTORS[detector_name]
        weights = entry["weights"]
        coco_detector = entry["fn"](weights=weights, box_score_thresh=0.6)
        coco_detector.eval()
        print("Torchvision detector loaded successfully!")

        # Test loading the FilteredDetector
        COCO_PERSON = 1  # COCO class ID for person
        person_detector = FilteredDetector(coco_detector, class_id=COCO_PERSON)
        person_detector.eval()
        print("Filtered detector loaded successfully!")

        _ = get_filtered_coco_detector_inference_runner(
            model_name=detector_name,
            category_id=COCO_PERSON,
            batch_size=1,
            model_config=superanimal_config,
        )
        print("Filtered detector runner created successfully!")

    print(
        "\n✅ All tests passed! The torchvision detector integration is working correctly."
    )
    return True


if __name__ == "__main__":
    print("Testing superanimal_humanbody with torchvision detector...")
    success = test_torchvision_detector()
    if success:
        print(
            "\n✅ Test passed! The torchvision detector works with superanimal_humanbody"
        )
    else:
        print(
            "\n❌ Test failed! There's an issue with the torchvision detector integration"
        )
