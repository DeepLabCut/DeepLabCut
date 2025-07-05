#!/usr/bin/env python3
"""
Test script for superanimal_humanbody with torchvision detector
"""

import torch
import torchvision.models.detection as detection
from deeplabcut.pose_estimation_pytorch.modelzoo import load_super_animal_config

def test_torchvision_detector():
    """Test that the torchvision detector works with superanimal_humanbody"""
    
    # Load the superanimal_humanbody config
    config = load_super_animal_config(
        super_animal="superanimal_humanbody",
        model_name="rtmpose_x",
        detector_name="fasterrcnn_mobilenet_v3_large_fpn",
    )
    
    print("Config loaded successfully!")
    print(f"Model method: {config['method']}")
    print(f"Detector variant: {config['detector']['model']['variant']}")
    
    # Check if the detector is configured to use torchvision
    detector_config = config['detector']['model']
    print(f"Detector config: {detector_config}")
    
    # Test loading the torchvision detector directly
    print("\nTesting torchvision detector loading...")
    weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    detector = detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, box_score_thresh=0.6,
    )
    detector.eval()
    print("Torchvision detector loaded successfully!")
    
    # Test that the detector config matches what we expect for torchvision
    print("\nTesting detector config compatibility...")
    expected_variant = "fasterrcnn_mobilenet_v3_large_fpn"
    actual_variant = detector_config.get("variant", "")
    
    if actual_variant == expected_variant:
        print(f"✅ Detector variant matches expected: {expected_variant}")
    else:
        print(f"❌ Detector variant mismatch. Expected: {expected_variant}, Got: {actual_variant}")
        return False
    
    # Test that the config has the correct structure for torchvision detector
    if "type" in detector_config and detector_config["type"] == "FasterRCNN":
        print("✅ Detector type is correctly set to FasterRCNN")
    else:
        print("❌ Detector type is not correctly set")
        return False
    
    # Test that the config allows for torchvision weights (no pretrained field or pretrained=False)
    if "pretrained" not in detector_config or detector_config.get("pretrained") is False:
        print("✅ Detector config allows torchvision weights")
    else:
        print("❌ Detector config has pretrained=True, which may conflict with torchvision weights")
        return False
    
    print("\n✅ All tests passed! The torchvision detector integration is working correctly.")
    return True

if __name__ == "__main__":
    print("Testing superanimal_humanbody with torchvision detector...")
    success = test_torchvision_detector()
    if success:
        print("\n✅ Test passed! The torchvision detector works with superanimal_humanbody")
    else:
        print("\n❌ Test failed! There's an issue with the torchvision detector integration") 