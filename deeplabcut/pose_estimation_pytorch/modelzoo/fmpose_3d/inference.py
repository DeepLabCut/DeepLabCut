#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import json
import logging
from pathlib import Path

import numpy as np

from deeplabcut.modelzoo.utils import get_superanimal_colormaps
from deeplabcut.pose_estimation_pytorch.apis.videos import (
    VideoIterator,
    create_df_from_prediction,
)
from deeplabcut.pose_estimation_pytorch.modelzoo.fmpose_3d.fmpose3d import (
    FMPOSE3D_MODEL_METADATA,
    get_fmpose3d_inference_api,
)
from deeplabcut.utils import auxiliaryfunctions_3d
from deeplabcut.utils.make_labeled_video import create_video

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return json.JSONEncoder.default(self, obj)


def _pose2d_to_dlc_predictions(
    pose_2d,
    max_individuals: int,
    num_bodyparts: int,
) -> list[dict[str, np.ndarray]]:
    """Convert FMPose3D 2D output to DLC per-frame prediction format."""
    all_kpts = np.asarray(pose_2d.keypoints)
    all_scores = np.asarray(pose_2d.scores)
    if all_kpts.ndim != 4 or all_scores.ndim != 3:
        raise ValueError(
            "Expected pose_2d keypoints/scores shaped as (num_persons, num_frames, num_bodyparts, {2 or score})."
        )

    num_frames = all_kpts.shape[1]
    num_persons = all_kpts.shape[0]
    per_frame: list[dict[str, np.ndarray]] = []
    for frame_idx in range(num_frames):
        n_det = min(num_persons, max_individuals)
        bodyparts_array = np.zeros((max_individuals, num_bodyparts, 3))
        bodyparts_array[:n_det, :, :2] = all_kpts[:n_det, frame_idx, :num_bodyparts, :2]
        bodyparts_array[:n_det, :, 2] = all_scores[:n_det, frame_idx, :num_bodyparts]
        per_frame.append({"bodyparts": bodyparts_array})
    return per_frame


# NOTE:  i_o; pandas; kpt_refactor;  this function may need to change in the future, to improve dataframe
#  i/o migration to validated keypoint schemas (parquet)
def _poses3d_to_dataframe(poses_3d: list[np.ndarray], df_2d, scorer_3d: str):
    """Create and fill a 3D dataframe using the shared auxiliary helper."""
    df_3d, scorer_3d, bodyparts = auxiliaryfunctions_3d.create_empty_df(df_2d, scorer_3d, "3d")
    n_frames = len(poses_3d)
    n_bodyparts = len(bodyparts)
    arr = np.full((n_frames, n_bodyparts, 3), np.nan, dtype=float)

    for frame_idx, pose in enumerate(poses_3d):
        pose_np = np.asarray(pose)
        if pose_np.ndim == 3:
            if pose_np.shape[0] == 0:
                continue
            pose_np = pose_np[0]
        if pose_np.ndim != 2 or pose_np.shape[-1] != 3:
            continue

        n = min(n_bodyparts, pose_np.shape[0])
        arr[frame_idx, :n] = pose_np[:n]

    xyz_cols = [(scorer_3d, bp, coord) for bp in bodyparts for coord in ("x", "y", "z")]
    df_3d.loc[:, xyz_cols] = arr.reshape(n_frames, -1)

    return df_3d


def _video_inference_fmpose3d(
    video_paths: str | Path | list[str | Path],
    model_name: str,
    max_individuals: int = 1,
    pcutoff: float = 0.1,
    batch_size: int = 1,
    dest_folder: str | Path | None = None,
    device: str | None = None,
    create_labeled_video: bool = True,
    cropping: list[int] | None = None,
    include_3d_in_return: bool = False,
) -> dict:
    """Perform FMPose3D video inference with a lightweight DLC loop."""
    import torch
    from tqdm import tqdm

    if max_individuals != 1:
        logger.warning(
            "FMPose3D 3D lifting currently supports only one individual. "
            "Clamping max_individuals=%s to 1 for this pipeline.",
            max_individuals,
        )
    max_individuals = 1

    if device is None or device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if isinstance(video_paths, (str, Path)):
        video_paths = [video_paths]

    if model_name not in FMPOSE3D_MODEL_METADATA:
        raise ValueError(
            f"Unsupported FMPose3D model '{model_name}'. "
            "Use one of: " + ", ".join(sorted(FMPOSE3D_MODEL_METADATA.keys()))
        )
    metadata = FMPOSE3D_MODEL_METADATA[model_name]
    model_cfg = metadata.build_model_cfg(max_individuals)
    bodyparts = list(metadata.bodyparts)
    num_bodyparts = metadata.num_bodyparts
    superanimal_name = metadata.superanimal_name

    api = get_fmpose3d_inference_api(model_type=model_name, device=device)

    dest_folder = Path(video_paths[0]).parent if dest_folder is None else Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    if create_labeled_video:
        superanimal_colormaps = get_superanimal_colormaps()
        colormap = superanimal_colormaps[superanimal_name]

    dlc_scorer = f"DLC_{model_name}"
    results = {}

    for video_path in video_paths:
        print(f"Processing video {video_path} with {model_name}")
        video = VideoIterator(video_path, cropping=cropping)
        vid_w, vid_h = video.dimensions

        predictions_2d: list[dict[str, np.ndarray]] = []
        all_poses_3d: list[np.ndarray] = []
        warned_multi_person_2d = False

        def _process_batch(
            frames: list[np.ndarray],
            predictions_2d=predictions_2d,
            all_poses_3d=all_poses_3d,
        ) -> None:
            nonlocal warned_multi_person_2d
            pose_2d = api.prepare_2d(source=np.stack(frames))
            num_detected = int(np.asarray(pose_2d.keypoints).shape[0])
            if num_detected > 1 and not warned_multi_person_2d:
                logger.warning(
                    "Multiple 2D detections (%s) were found, but FMPose3D 3D lifting uses only the first individual.",
                    num_detected,
                )
                warned_multi_person_2d = True
            predictions_2d.extend(
                _pose2d_to_dlc_predictions(
                    pose_2d,
                    max_individuals=max_individuals,
                    num_bodyparts=num_bodyparts,
                )
            )
            try:
                pose_3d = api.pose_3d(
                    keypoints_2d=pose_2d.keypoints,
                    image_size=pose_2d.image_size,
                )
                all_poses_3d.extend(np.asarray(pose_3d.poses_3d))
            except ValueError as e:
                logger.info("Skipping 3D lifting for batch due to invalid 2D result: %s", e)
                all_poses_3d.extend([np.zeros((0, num_bodyparts, 3)) for _ in frames])

        batch: list[np.ndarray] = []
        for frame in tqdm(video, desc="FMPose3D inference"):
            batch.append(frame)
            if len(batch) == batch_size:
                _process_batch(batch)
                batch.clear()
        if batch:
            _process_batch(batch)

        output_prefix = f"{Path(video_path).stem}_{dlc_scorer}"
        output_h5 = dest_folder / f"{output_prefix}.h5"

        print(f"Saving 2D results to {dest_folder}")
        df = create_df_from_prediction(
            predictions=predictions_2d,
            dlc_scorer=dlc_scorer,
            multi_animal=True,
            model_cfg=model_cfg,
            output_path=dest_folder,
            output_prefix=output_prefix,
        )
        scorer_3d = f"{dlc_scorer}_3d"
        df_3d = _poses3d_to_dataframe(all_poses_3d, df, scorer_3d)
        output_3d_h5 = dest_folder / f"{output_prefix}_3d.h5"
        df_3d.to_hdf(output_3d_h5, key="df_with_missing", mode="w", format="table")
        print(f"3D dataframe saved to {output_3d_h5}")

        if include_3d_in_return:
            results[video_path] = {
                "df_2d": df,
                "df_3d": df_3d,
            }
        else:
            results[video_path] = df

        output_json = dest_folder / f"{output_prefix}.json"
        with open(output_json, "w") as f:
            json.dump(predictions_2d, f, cls=NumpyEncoder)

        poses_3d_serialisable = [pose.tolist() if isinstance(pose, np.ndarray) else pose for pose in all_poses_3d]
        output_3d_json = dest_folder / f"{output_prefix}_3d.json"
        with open(output_3d_json, "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "bodyparts": bodyparts,
                    "poses_3d": poses_3d_serialisable,
                },
                f,
            )
        print(f"3D predictions saved to {output_3d_json}")

        if create_labeled_video:
            bbox = cropping
            if cropping is None:
                bbox = (0, vid_w, 0, vid_h)
            output_video = dest_folder / f"{output_prefix}_labeled.mp4"
            create_video(
                video_path,
                output_h5,
                pcutoff=pcutoff,
                fps=video.fps,
                bbox=bbox,
                cmap=colormap,
                output_path=output_video,
                plot_bboxes=False,
                bboxes_list=[],
                bboxes_pcutoff=0.0,
            )
            print(f"Video with predictions was saved as {output_video}")

    return results
