""" Benchmarking maDLC datasets - inference

This script can be used to run inference on the test images of a DeepLabCut project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from deeplabcut.pose_estimation_pytorch import PoseDatasetParameters
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.utils.visualization import make_labeled_images_from_dataframe
from ruamel.yaml import YAML
from tqdm import tqdm

from projects import MA_DLC_BENCHMARKS
from utils import Project, Shuffle


def run_inference_on_all_images(
    project: Project,
    snapshot: Path,
    save_as_csv: bool,
    plot: bool,
    detector_snapshot: Path | None = None,
) -> None:
    pytorch_config_path = snapshot.parent / "pytorch_config.yaml"
    with open(pytorch_config_path, "r") as file:
        pytorch_config = YAML(typ="safe", pure=True).load(pytorch_config_path)

    parameters = PoseDatasetParameters(
        bodyparts=pytorch_config["metadata"]["bodyparts"],
        unique_bpts=pytorch_config["metadata"]["unique_bodyparts"],
        individuals=pytorch_config["metadata"]["individuals"],
        with_center_keypoints=pytorch_config.get("with_center_keypoints", False),
        color_mode=pytorch_config.get("color_mode", "RGB"),
        cropped_image_size=pytorch_config.get("output_size", (256, 256)),
    )
    shuffle_name = snapshot.parent.parent.name
    test_data_dir = project.root / "test-images" / project.name / "labeled-data"
    video_folders = [p for p in test_data_dir.iterdir() if p.is_dir()]
    images = []
    for video_folder in video_folders:
        images += [
            p  # f"labeled-data/{video_folder.name}/{p.name}"
            for p in video_folder.iterdir()
            if p.suffix == ".png"
        ]

    runner, detector_runner = get_inference_runners(
        model_config=pytorch_config,
        snapshot_path=str(snapshot),
        max_individuals=parameters.max_num_animals,
        num_bodyparts=parameters.num_joints,
        num_unique_bodyparts=parameters.num_unique_bpts,
        with_identity=False,  # TODO: implement
        transform=None,
        detector_path=str(detector_snapshot),
        detector_transform=None,
    )

    pose_inputs = [str(i) for i in images]
    if detector_runner is not None:
        print("Running detection")
        bbox_predictions = detector_runner.inference(images=tqdm(pose_inputs))
        pose_inputs = list(zip(pose_inputs, bbox_predictions))

    print("Running pose prediction")
    predictions = runner.inference(tqdm(pose_inputs))
    poses = np.array([p["bodyparts"] for p in predictions])
    poses = poses[..., :3]

    if detector_snapshot is None:
        scorer = f"{shuffle_name}-{snapshot.stem}"
    else:
        scorer = f"{shuffle_name}-{detector_snapshot.stem}-{snapshot.stem}"

    output_path = (
        project.root
        / "test-images"
        / project.name
        / "evaluation-results"
        / f"iteration-{project.iteration}"
        / shuffle_name
        / "benchmark"
        / f"{scorer}.h5"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)

    index = pd.MultiIndex.from_tuples(
        [(f"labeled-data", f"{i.parent.name}", f"{i.name}") for i in images],
        names=["dir", "video", "image"],
    )
    columns = pd.MultiIndex.from_product(
        [
            [scorer],
            project.cfg["individuals"],
            project.cfg["multianimalbodyparts"],
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    poses = poses.reshape(len(images), -1)
    if parameters.num_unique_bpts > 0:
        unique_columns = pd.MultiIndex.from_product(
            [
                [scorer],
                ["single"],
                parameters.unique_bpts,
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        columns = columns.append(unique_columns)
        unique_poses = np.array([p["unique_bodyparts"] for p in predictions])
        unique_poses = unique_poses[..., :3]
        unique_poses = unique_poses.reshape(len(images), -1)
        poses = np.concatenate([poses, unique_poses], axis=1)

    df = pd.DataFrame(poses, index=index, columns=columns)
    df.to_hdf(output_path, key="df_with_missing")
    if save_as_csv:
        df.to_csv(output_path.with_suffix(".csv"))

    if plot:
        test_config_path = str(
            project.root / "test-images" / project.name / "config.yaml"
        )
        with open(test_config_path, "r") as file:
            test_config = YAML(typ="safe", pure=True).load(file)

        image_output_folder = output_path.parent / "images"
        image_output_folder.mkdir(exist_ok=True)
        for video in video_folders:
            index_filter = [v == video.name for v in df.index.get_level_values("video")]
            col_filter = [
                c in ("x", "y") and bpt not in ("dfin1", "dfin2")
                for c, bpt in zip(
                    df.columns.get_level_values("coords"),
                    df.columns.get_level_values("bodyparts"),
                )
            ]
            df_video = df.loc[index_filter, col_filter]
            plot_output_folder = image_output_folder / video.name
            make_labeled_images_from_dataframe(
                df_video,
                test_config,
                destfolder=str(plot_output_folder),
                scale=1.0,
                dpi=200,
                keypoint="+",
                draw_skeleton=False,
                color_by="bodypart",
            )


def main(
    shuffle: Shuffle,
    snapshot_indices: int | list[int] | None = None,
    detector_snapshot_indices: int | list[int] | None = None,
    save_as_csv: bool = False,
    plot: bool = False,
) -> None:
    """

    Args:
        shuffle:
        snapshot_indices:
        detector_snapshot_indices:
        save_as_csv:
        plot:

    Returns:

    """
    if isinstance(snapshot_indices, int):
        snapshot_indices = [snapshot_indices]
    if isinstance(detector_snapshot_indices, int):
        detector_snapshot_indices = [detector_snapshot_indices]

    detectors = [None]
    if shuffle.pytorch_cfg.get("method", "bu").lower() == "td":
        detectors = shuffle.snapshots(detector=True)
        if detector_snapshot_indices is not None:
            detectors = [detectors[idx] for idx in detector_snapshot_indices]
        print(f"Running inference with detectors: {[s.name for s in detectors]}")

    snapshots = shuffle.snapshots()
    if snapshot_indices is not None:
        snapshots = [snapshots[idx] for idx in snapshot_indices]
    print(f"Running inference with snapshots: {[s.name for s in snapshots]}")

    for detector in detectors:
        for snapshot in snapshots:
            run_inference_on_all_images(
                shuffle.project, snapshot, save_as_csv, plot, detector
            )


if __name__ == "__main__":
    main(
        shuffle=Shuffle(
            project=MA_DLC_BENCHMARKS["trimouse"],
            index=0,
            train_fraction=0.95,
        ),
        snapshot_indices=None,
        detector_snapshot_indices=-1,
        save_as_csv=False,
        plot=False,
    )
