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
from __future__ import annotations

import shutil
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, for CI/CD on Windows

import cv2
import deeplabcut
import deeplabcut.utils.auxiliaryfunctions as af
import numpy as np
import pandas as pd
from deeplabcut.compat import Engine
from deeplabcut.generate_training_dataset import get_existing_shuffle_indices
from PIL import Image


def log_step(message: Any) -> None:
    print(100 * "-")
    print(str(message))
    print(100 * "-")


def cleanup(test_path: Path) -> None:
    if test_path.exists():
        shutil.rmtree(test_path)


@dataclass(frozen=True)
class SyntheticProjectParameters:
    multianimal: bool
    num_bodyparts: int
    num_frames: int = 10
    num_individuals: int = 1
    num_unique: int = 0
    identity: bool = False
    frame_shape: tuple[int, int] = (480, 640)

    def bodyparts(self) -> list[str]:
        return [i for i in string.ascii_lowercase[: self.num_bodyparts]]

    def unique(self) -> list[str]:
        return [f"unique_{i}" for i in string.ascii_lowercase[: self.num_unique]]

    def individuals(self) -> list[str]:
        return [f"animal_{i}" for i in range(self.num_individuals)]


def sample_pose_random(
    gen: np.random.Generator,
    num_individuals: int,
    num_bodyparts: int,
    num_unique: int,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Fully random pose sampling"""
    xs = gen.choice(img_w, size=(num_individuals, num_bodyparts), replace=False)
    ys = gen.choice(img_h, size=(num_individuals, num_bodyparts), replace=False)
    pose = np.stack([xs, ys], axis=-1)

    image_data = pose.reshape(-1)
    if num_unique > 0:
        unique_pose = np.stack(
            [
                gen.choice(img_w, size=(1, num_unique), replace=False),
                gen.choice(img_h, size=(1, num_unique), replace=False),
            ],
            axis=-1,
        )
        image_data = np.concatenate([image_data, unique_pose.reshape(-1)])
    return image_data


def sample_pose_from_center(
    center_xs: np.ndarray,
    center_ys: np.ndarray,
    num_individuals: int,
    num_bodyparts: int,
    num_unique: int,
    radius: int = 25,
) -> np.ndarray:
    """Sample keypoints from the center of each individual"""
    pose = np.zeros((num_individuals, num_bodyparts, 2))
    for i, (xc, yc) in enumerate(zip(center_xs, center_ys)):
        if i < num_individuals:
            x_start, x_end = xc - radius + 1, xc + radius - 1
            y_start, y_end = yc - radius + 1, yc + radius - 1
            pose[i, :, 0] = np.linspace(start=x_start, stop=x_end, num=num_bodyparts)
            pose[i, :, 1] = np.linspace(start=y_start, stop=y_end, num=num_bodyparts)

    image_data = pose.reshape(-1)
    if num_unique > 0:
        xc, yc = center_xs[-1], center_ys[-1]
        x_start, x_end = xc - radius + 1, xc + radius - 1
        y_start, y_end = yc - radius + 1, yc + radius - 1
        unique_pose = np.zeros((1, num_unique, 2))
        unique_pose[0, :, 0] = np.linspace(start=x_start, stop=x_end, num=num_unique)
        unique_pose[0, :, 1] = np.linspace(start=y_start, stop=y_end, num=num_unique)
        image_data = np.concatenate([image_data, unique_pose.reshape(-1)])
    return image_data


def gen_fake_data(
    scorer: str,
    video_name: str,
    params: SyntheticProjectParameters,
) -> pd.DataFrame:
    kpt_entries = ["x", "y"]
    col_names = ["scorer", "individuals", "bodyparts", "coords"]
    col_values = []
    for i in params.individuals():
        for b in params.bodyparts():
            col_values += [(scorer, i, b, entry) for entry in kpt_entries]

    for unique_bpt in params.unique():
        col_values += [(scorer, "single", unique_bpt, entry) for entry in kpt_entries]

    index_data = []
    pose_data = []
    gen = np.random.default_rng(seed=0)

    # sample starting points for each individual
    img_h, img_w = params.frame_shape[:2]
    radius = 8
    center_xs = gen.choice(
        np.arange(radius, img_w - radius),
        size=params.num_individuals + 1,  # in case unique bodyparts
        replace=False,
    )
    center_ys = gen.choice(
        np.arange(radius, img_h - radius),
        size=params.num_individuals + 1,  # in case unique bodyparts
        replace=False,
    )

    for frame_index in range(params.num_frames):
        index_data.append(("labeled-data", video_name, f"img{frame_index:04}.png"))
        pose_data.append(
            sample_pose_from_center(
                center_xs,
                center_ys,
                num_individuals=params.num_individuals,
                num_bodyparts=params.num_bodyparts,
                num_unique=params.num_unique,
                radius=radius,
            )
        )
        mvt_x = gen.integers(low=-1, high=4, size=center_xs.size)
        mvt_y = gen.integers(low=-1, high=4, size=center_ys.size)
        center_xs = np.clip(center_xs + mvt_x, radius, img_w - radius)
        center_ys = np.clip(center_ys + mvt_y, radius, img_h - radius)

    pose = np.stack(pose_data)
    pose[params.num_frames // 2, :] = np.nan  # add missing row in a frame
    for idv in range(params.num_individuals):
        idv_start = 2 * params.num_bodyparts * idv
        idv_end = 2 * params.num_bodyparts * (idv + 1)
        if params.num_frames > idv + 1:
            pose[idv + 1, idv_start:idv_end] = np.nan

    for bpt in range(params.num_bodyparts):
        frame_idx = 1 + params.num_individuals + bpt
        idv_idx = bpt % params.num_individuals
        offset = 2 * params.num_bodyparts * idv_idx
        bpt_start, bpt_end = 2 * bpt + offset, 2 * (bpt + 1) + offset
        if params.num_frames + 1 > frame_idx:
            pose[frame_idx, bpt_start:bpt_end] = np.nan

    return pd.DataFrame(
        pose,
        index=pd.MultiIndex.from_tuples(index_data),
        columns=pd.MultiIndex.from_tuples(col_values, names=col_names),
    )


def gen_fake_image(
    project_root: Path,
    row: pd.Series,
    params: SyntheticProjectParameters,
    radius: int = 5,
):
    img_h, img_w = params.frame_shape
    image_array = np.zeros((*params.frame_shape, 3), dtype=np.uint8)
    for i, idv in enumerate(params.individuals()):
        r = int(255 * (i + 1) / params.num_individuals)
        if "individuals" in row.index.names:
            idv_data = row.droplevel("scorer").loc[idv]
        else:
            idv_data = row.droplevel("scorer")

        keypoints = idv_data.to_numpy().reshape((-1, 2))
        if not np.all(np.isnan(keypoints)):
            idv_center = np.nanmean(keypoints, axis=0)
            x, y = int(idv_center[0]), int(idv_center[1])
            xmin, xmax = max(0, x - radius), min(img_w - 1, x + radius)
            ymin, ymax = max(0, y - radius), min(img_h - 1, y + radius)
            image_array[ymin:ymax, xmin:xmax, 0] = r

            for j, bpt in enumerate(params.bodyparts()):
                g = int(255 * (j + 1) / params.num_bodyparts)

                bpt_data = idv_data.loc[bpt]
                if np.all(~pd.isnull(bpt_data)):
                    x, y = int(bpt_data.x), int(bpt_data.y)
                    xmin, xmax = max(0, x - radius), min(img_w - 1, x + radius)
                    ymin, ymax = max(0, y - radius), min(img_h - 1, y + radius)
                    image_array[ymin:ymax, xmin:xmax, 0] = r
                    image_array[ymin:ymax, xmin:xmax, 1] = g

    if params.num_unique > 0:
        unique_data = row.droplevel("scorer").loc["single"]
        for i, unique_bpt in enumerate(params.unique()):
            bpt_data = unique_data.loc[unique_bpt]
            if np.all(~pd.isnull(bpt_data)):
                x, y = int(bpt_data.x), int(bpt_data.y)
                xmin, xmax = max(0, x - radius), min(img_w - 1, x + radius)
                ymin, ymax = max(0, y - radius), min(img_h - 1, y + radius)
                image_array[ymin:ymax, xmin:xmax, 2] = int(
                    255 * (i + 1) / params.num_unique
                )

    img = Image.fromarray(image_array)
    img.save(project_root / Path(*row.name))


def generate_video_from_images(image_dir: Path, output_video: Path) -> None:
    images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix == ".png"]
    images = sorted(images, key=lambda f: f.stem)
    if len(images) == 0:
        return

    height, width, channels = cv2.imread(str(images[0])).shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(output_video), fourcc, 10, (width, height))
    for img_path in images:
        img = cv2.imread(str(img_path))
        out.write(img)
    out.release()


def create_fake_project(path: Path, params: SyntheticProjectParameters) -> None:
    if path.exists():
        print(f"[DEBUG] Path exists: {path} (is_dir={path.is_dir()}, is_file={path.is_file()})")
        raise ValueError(f"Cannot create a fake project at an existing path")

    scorer = "synthetic"
    video_name = "cat"
    path.mkdir(parents=True, exist_ok=False)
    config = {
        "Task": "synthetic",
        "scorer": scorer,
        "date": "Nov11",
        "multianimalproject": params.multianimal,
        "identity": params.identity,
        "project_path": str(path / "config.yaml"),
        "TrainingFraction": [0.8],
        "iteration": 0,
        "default_net_type": "resnet_50",
        "default_augmenter": "default",
        "default_track_method": "ellipse",
        "snapshotindex": "all",
        "batch_size": 8,
        "pcutoff": 0.6,
        "video_sets": {
            str(path / "videos" / video_name): {
                "crop": (0, params.frame_shape[1], 0, params.frame_shape[0]),
            },
        },
        "start": 0,
        "stop": 1,
        "numframes2pick": 10,
        "dotsize": 4,
        "alphavalue": 1.0,
        "colormap": "rainbow",
    }
    if not params.multianimal:
        config["bodyparts"] = params.bodyparts()
        assert params.num_individuals == 1
        assert params.num_unique == 0
    else:
        config["bodyparts"] = "MULTI!"
        config["multianimalbodyparts"] = params.bodyparts()
        config["uniquebodyparts"] = params.unique()
        config["individuals"] = params.individuals()

    af.write_config(str(path / "config.yaml"), config)
    image_dir = path / "labeled-data" / video_name
    image_dir.mkdir(parents=True, exist_ok=False)

    df = gen_fake_data(
        scorer=scorer,
        video_name=video_name,
        params=params,
    )
    print("SYNTHETIC DATA:")
    print(df)
    print("\n")
    if not params.multianimal:
        df.columns = df.columns.droplevel("individuals")

    df.to_hdf(image_dir / f"CollectedData_{scorer}.h5", key="df_with_missing")
    df.to_csv(image_dir / f"CollectedData_{scorer}.csv")

    for idx in range(params.num_frames):
        gen_fake_image(path, df.iloc[idx], params=params, radius=5)

    output_video = path / "videos" / "video.mp4"
    output_video.parent.mkdir(exist_ok=True)
    generate_video_from_images(image_dir, output_video)


def copy_project_for_test() -> Path:
    data_path = Path.cwd() / "openfield-Pranav-2018-10-30"
    test_path = Path.cwd() / "pytorch-testscript1234-openfield-Pranav-2018-10-30"
    if not test_path.exists():
        shutil.copytree(data_path, test_path)

    project_config = af.read_config(str(test_path / "config.yaml"))
    videos = list(project_config["video_sets"].keys())
    video = videos[0]
    crop = project_config["video_sets"][video]
    project_config["video_sets"] = {str(test_path / "videos" / "m3v1mp4.mp4"): crop}
    af.write_config(str(test_path / "config.yaml"), project_config)
    return test_path


def run(
    config_path: Path,
    train_fraction: float,
    trainset_index: int,
    net_type: str,
    videos: list[str],
    device: str,
    engine: Engine = Engine.PYTORCH,
    pytorch_cfg_updates: dict | None = None,
    create_labeled_videos: bool = False,
) -> None:
    times = [time.time()]
    log_step(f"Testing with net type {net_type}")
    log_step("Creating the training dataset")
    deeplabcut.create_training_dataset(
        str(config_path), net_type=net_type, engine=engine
    )
    existing_shuffles = get_existing_shuffle_indices(
        config_path, train_fraction=train_fraction, engine=engine
    )
    shuffle_index = existing_shuffles[-1]

    log_step(
        f"Starting training for train_frac {train_fraction}, shuffle {shuffle_index}"
    )
    deeplabcut.train_network(
        config=str(config_path),
        shuffle=shuffle_index,
        trainingsetindex=trainset_index,
        device=device,
        pytorch_cfg_updates=pytorch_cfg_updates,
    )
    times.append(time.time())
    log_step(f"Train time: {times[-1] - times[-2]} seconds")

    log_step(
        f"Starting evaluation for train_frac {train_fraction}, shuffle {shuffle_index}"
    )
    deeplabcut.evaluate_network(
        config=str(config_path),
        Shuffles=[shuffle_index],
        trainingsetindex=trainset_index,
        device=device,
        plotting=True,
        per_keypoint_evaluation=True,
    )
    times.append(time.time())
    log_step(f"Evaluation time: {times[-1] - times[-2]} seconds")

    if len(videos) > 0:
        log_step(f"Analyzing videos for {train_fraction}, shuffle {shuffle_index}")
        video_kwargs = dict(
            videos=videos, shuffle=shuffle_index, trainingsetindex=trainset_index
        )
        deeplabcut.analyze_videos(
            str(config_path), **video_kwargs, device=device, auto_track=False
        )
        times.append(time.time())
        log_step(f"Video analysis time: {times[-1] - times[-2]} seconds")
        log_step(f"Total test time: {times[-1] - times[0]} seconds")

        cfg = af.read_config(config_path)
        if cfg.get("multianimalproject"):
            if create_labeled_videos:
                deeplabcut.create_video_with_all_detections(
                    str(config_path), **video_kwargs
                )

            # relaxed tracking parameters
            deeplabcut.convert_detections2tracklets(
                str(config_path),
                **video_kwargs,
                inferencecfg=dict(
                    boundingboxslack=10,
                    iou_threshold=0.2,
                    max_age=5,
                    method="m1",
                    min_hits=1,
                    minimalnumberofconnections=2,
                    pafthreshold=0.1,
                    pcutoff=0.1,
                    topktoretain=3,
                    variant=0,
                    withid=False,
                ),
            )
            deeplabcut.stitch_tracklets(str(config_path), **video_kwargs, min_length=3)

        if create_labeled_videos:
            log_step(f"Making labeled video, {train_fraction}, shuffle={shuffle_index}")
            results = deeplabcut.create_labeled_video(
                config=str(config_path),
                videos=videos,
                shuffle=shuffle_index,
                trainingsetindex=trainset_index,
            )
            assert all(results), f"Failed to create some labeled video for {videos}"


if __name__ == "__main__":
    create_fake_project(
        path=Path("synthetic-data-niels"),
        params=SyntheticProjectParameters(
            multianimal=True,
            num_bodyparts=4,
            num_individuals=3,
            num_unique=1,
            num_frames=50,
            frame_shape=(128, 256),
        ),
    )
