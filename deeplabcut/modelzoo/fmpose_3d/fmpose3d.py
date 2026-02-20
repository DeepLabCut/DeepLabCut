"""
DeepLabCut2.0-3.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

from fmpose3d import (
    FMPose3DInference,
    FMPose3DConfig,
    SupportedModel,
)


def get_fmpose3d_inference_api(
    model_type: SupportedModel = "fmpose3d_humans",
    snapshot_path: str | None = None,
    device: str | None = None,
    config_kwargs: dict = {},
    ) -> FMPose3DInference:
    """
    Get a FMPose3DInference API for a given model type and snapshot path.
    
    Args:
        model_type: one of the supported model types: "fmpose3d_humans", "fmpose3d_animals",
        snapshot_path: The path to the snapshot file. If None, FMPose3D will download the default snapshot.
        device: The device to use. If None, the device will be inferred from the environment.
        config_kwargs: Additional keyword arguments to pass to the FMPose3DConfig.
    Returns:
        FMPose3DInference: An FMPose3DInference API runner.
    
    Example Usages
    ```python
    # Initialize the API (downloads the default weights automatically from huggingface)
    fmpose = get_fmpose3d_inference_api(
        model_type="fmpose3d_animals",
        device="cuda:0",
    )

    # Run inference on an image
    predictions_3d = fmpose.predict(source="path/to/image.jpg") # or (H, W, 3) numpy array

    # Lift 2d predictions to 3d
    keypoints_2d = np.random.rand(num_frames, num_joints, 2)
    predictions_3d = fmpose.pose_3d(keypoints_2d=keypoints_2d)
    ```
    """
    model_config = FMPose3DConfig(model_type=model_type, **config_kwargs)   
    fmpose3d_api = FMPose3DInference(
        model_config,
        model_weights_path=snapshot_path,
        device=device
    )
    return fmpose3d_api