""" Benchmarking maDLC datasets

TODO: Document data format

In a first step, create_dataset=True was used to create the training dataset files and
the pytorch configurations for the models. The data augmentation parameters were then
updated for the shuffle that I wanted to train. This also included adding the following:

```
logger:
 type: 'WandbLogger'
 project_name: 'dlc3-ff5f2af-fish'
 run_name: 'dekr-w32-shuffle3'
```

Which specifies to log the run to wandb, (including the project and with which name each
shuffle should be logged).

Then run with `create_dataset=False, train=True` to train the models.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import ruamel.yaml as yaml
import wandb

import deeplabcut
import deeplabcut.pose_estimation_pytorch.apis as api
from deeplabcut.pose_estimation_pytorch import DLCLoader
from deeplabcut.pose_estimation_pytorch.apis.utils import (
    build_inference_transform,
    get_runners,
)
from deeplabcut.utils.visualization import make_labeled_images_from_dataframe


@dataclass
class ProjectConfig:
    data_root: Path
    name: str
    iteration: int
    shuffle_prefix: str

    def snapshot_path(
        self,
        train_percentage: int,
        shuffle: int,
        num_epochs: int,
    ) -> Path:
        return (
            self.data_root
            / self.name
            / "dlc-models"
            / f"iteration-{self.iteration}"
            / f"{self.shuffle_prefix}-trainset{train_percentage}shuffle{shuffle}"
            / "train"
            / f"snapshot-{num_epochs}.pt"
        )


@dataclass
class DataParameters:
    model_prefix: str = ""
    output_folder: str = "videos"
    shuffle: int = 0
    trainset_index: int = 0
    net_types: tuple[str, ...] = ("dekr_w18", "dekr_w32", "dekr_w48")


@dataclass
class RunParameters:
    create_dataset: bool = False
    train: bool = False
    evaluate: bool = False
    analyze_videos: bool = False
    track: bool = False
    create_labeled_video: bool = False
    device: str = "cuda:0"


@dataclass
class TrainParameters:
    batch_size: int = 16
    display_iters: int = 500
    epochs: int | None = None
    save_epochs: int | None = None
    snapshot_path: Path | None = None
    detector_max_epochs: int | None = None
    detector_save_epochs: int | None = None

    def train_kwargs(self) -> dict:
        kwargs = {
            "batch_size": self.batch_size,
            "display_iters": self.display_iters,
        }
        if self.epochs is not None:
            kwargs["epochs"] = self.epochs
        if self.save_epochs is not None:
            kwargs["save_epochs"] = self.save_epochs
        if self.snapshot_path is not None:
            kwargs["snapshot_path"] = str(self.snapshot_path)
        if self.detector_max_epochs is not None:
            kwargs["detector_max_epochs"] = self.detector_max_epochs
        if self.detector_save_epochs is not None:
            kwargs["detector_save_epochs"] = self.detector_save_epochs
        return kwargs


@dataclass
class EvalParameters:
    snapshotindex: int | list[int] | str | None = (None,)
    plotting: str | bool = False
    show_errors: bool = True

    def eval_kwargs(self) -> dict:
        return {
            "plotting": self.plotting,
            "show_errors": self.show_errors,
        }


def run_inference_on_all_images(
    project: ProjectConfig,
    snapshot: Path,
    plot: bool,
) -> None:
    warnings.simplefilter("ignore", yaml.error.UnsafeLoaderWarning)

    with open(project.data_root / project.name / "config.yaml", "r") as file:
        config = yaml.load(file)

    pytorch_config_path = snapshot.parent / "pytorch_config.yaml"
    with open(pytorch_config_path, "r") as file:
        pytorch_config = yaml.load(file)

    loader = DLCLoader(
        project_root=str(project.data_root / project.name),
        model_config_path=str(pytorch_config_path),
    )
    parameters = loader.get_dataset_parameters()

    shuffle_name = snapshot.parent.parent.name

    video_folders = [
        p
        for p in (
            project.data_root / (project.name + "-test-images") / "labeled-data"
        ).iterdir()
        if p.is_dir()
    ]
    images = []
    for video_folder in video_folders:
        images += [
            p  # f"labeled-data/{video_folder.name}/{p.name}"
            for p in video_folder.iterdir()
            if p.suffix == ".png"
        ]

    transform_cfg = {
        "auto_padding": {
            "pad_width_divisor": 32,
            "pad_height_divisor": 32,
        },
        "normalize_images": True,
        "resize": False,
    }
    transform = build_inference_transform(transform_cfg, augment_bbox=True)
    runner, _ = get_runners(
        pytorch_config=pytorch_config,
        snapshot_path=str(snapshot),
        max_individuals=parameters.max_num_animals,
        num_bodyparts=parameters.num_joints,
        num_unique_bodyparts=parameters.num_unique_bpts,
        with_identity=False,  # TODO: implement
        transform=transform,
        detector_path=None,  # TODO: Fix for top-down models
        detector_transform=None,
    )
    predictions = runner.inference([str(i) for i in images])
    poses = np.array([p["bodyparts"] for p in predictions])

    output_path = (
        project.data_root
        / (project.name + "-test-images")
        / "evaluation-results"
        / f"iteration-{project.iteration}"
        / shuffle_name
        / "benchmark"
        / f"{shuffle_name}-{snapshot.stem}.h5"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)

    index = pd.MultiIndex.from_tuples(
        [(f"labeled-data", f"{i.parent.name}", f"{i.name}") for i in images],
        names=["dir", "video", "image"],
    )
    columns = pd.MultiIndex.from_product(
        [
            [shuffle_name],
            config["individuals"],
            config["multianimalbodyparts"],
            ["x", "y", "likelihood"],
        ],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )
    df = pd.DataFrame(poses.reshape(len(images), -1), index=index, columns=columns)
    df.to_hdf(output_path, "df_with_missing")
    if plot:
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
                config,
                destfolder=str(plot_output_folder),
                scale=1.0,
                dpi=200,
                keypoint="+",
                draw_skeleton=False,
                color_by="bodypart",
            )


def main(
    project_root: Path,
    iteration: int,
    data_parameters: DataParameters,
    run_parameters: RunParameters,
    train_parameters: TrainParameters,
    eval_parameters: EvalParameters,
) -> None:
    project = project_root.name
    cfg = project_root / "config.yaml"

    print("Training on dataset:")
    print(f"  Project        {project}")
    print(f"  Iteration      {iteration}")
    print(f"  Shuffle        {data_parameters.shuffle}")
    print(f"  Model prefix   {data_parameters.model_prefix}")

    # Configuration
    videos = str(project_root / "videos")
    deeplabcut.auxiliaryfunctions.edit_config(
        str(cfg),
        {"iteration": iteration},
    )

    if run_parameters.create_dataset:
        deeplabcut.create_training_model_comparison(
            str(cfg),
            trainindex=data_parameters.trainset_index,
            num_shuffles=1,
            net_types=list(data_parameters.net_types),
        )

    if run_parameters.train:
        api.train_network(
            str(cfg),
            shuffle=data_parameters.shuffle,
            trainingsetindex=data_parameters.trainset_index,
            transform=None,
            modelprefix=data_parameters.model_prefix,
            device=run_parameters.device,
            **train_parameters.train_kwargs(),
        )

    if run_parameters.evaluate:
        snapshot_indices = eval_parameters.snapshotindex
        if isinstance(snapshot_indices, int) or isinstance(snapshot_indices, str):
            snapshot_indices = [snapshot_indices]

        for idx in snapshot_indices:
            api.evaluate_network(
                config=str(cfg),
                shuffles=[data_parameters.shuffle],
                trainingsetindex=data_parameters.trainset_index,
                snapshotindex=idx,
                device=run_parameters.device,
                transform=None,
                modelprefix=data_parameters.model_prefix,
                **eval_parameters.eval_kwargs(),
            )

    if run_parameters.analyze_videos:
        api.analyze_videos(
            config=str(cfg),
            videos=videos,
            videotype=".mp4",
            trainingsetindex=data_parameters.trainset_index,
            destfolder=str(project_root / data_parameters.output_folder),
            snapshotindex=5,
            device=run_parameters.device,
            modelprefix=data_parameters.model_prefix,
            batchsize=train_parameters.batch_size,
            transform=None,
            overwrite=False,
            auto_track=False,
        )

    if run_parameters.track:
        api.convert_detections2tracklets(
            config=str(cfg),
            videos=videos,
            videotype=".mp4",
            modelprefix=data_parameters.model_prefix,
            destfolder=str(project_root / data_parameters.output_folder),
            track_method="box",
        )
        deeplabcut.stitch_tracklets(
            str(cfg),
            [videos],
            shuffle=1,
            trainingsetindex=data_parameters.trainset_index,
            destfolder=str(project_root / data_parameters.output_folder),
            modelprefix=data_parameters.model_prefix,
            save_as_csv=True,
            track_method="box",
        )

    if run_parameters.create_labeled_video:
        deeplabcut.create_labeled_video(
            config=str(cfg),
            videos=[videos],
            videotype="mp4",
            trainingsetindex=data_parameters.trainset_index,
            color_by="individual",  # bodypart, individual
            modelprefix=data_parameters.model_prefix,
            destfolder=str(project_root / data_parameters.output_folder),
            track_method="box",
        )


if __name__ == "__main__":
    benchmarks = {
        "trimouse": ProjectConfig(
            data_root=Path("/home/datasets"),
            name="trimice-dlc-2021-06-22",
            iteration=1,
            shuffle_prefix="trimiceJun22",
        ),
        "fish": ProjectConfig(
            data_root=Path("/home/datasets"),
            name="fish-dlc-2021-05-07",
            iteration=4,
            shuffle_prefix="fishMay7",
        ),
        "parenting": ProjectConfig(
            data_root=Path("/home/datasets"),
            name="pups-dlc-2021-03-24",
            iteration=1,
            shuffle_prefix="pupsMar24",
        ),
    }

    for name, project in benchmarks.items():
        if wandb.run is not None:  # TODO: Finish wandb run in DLC
            wandb.finish()

        print(f"Running {name}")
        data_parameters = DataParameters(
            model_prefix="",
            output_folder=f"videos-iter{project.iteration}",
            shuffle=0,
            trainset_index=0,
            net_types=(
                "dekr_w18",
                "dekr_w18",
                "dekr_w18",
                "dekr_w32",
                "dekr_w32",
                "dekr_w32",
            ),
        )
        run_parameters = RunParameters(
            create_dataset=False,
            train=True,
            evaluate=True,
            analyze_videos=False,
            track=False,
            create_labeled_video=False,
            device="cuda:0",
        )
        train_parameters = TrainParameters(
            batch_size=2,
            epochs=125,
            save_epochs=25,
        )

        try:
            main(
                project_root=(project.data_root / project.name),
                iteration=project.iteration,
                data_parameters=data_parameters,
                run_parameters=run_parameters,
                train_parameters=train_parameters,
                eval_parameters=EvalParameters(
                    snapshotindex="all",
                    plotting=False,
                ),
            )

            if run_parameters.train:
                for num_epochs in range(
                    train_parameters.save_epochs,
                    train_parameters.epochs + 1,
                    train_parameters.save_epochs,
                ):
                    snapshot = project.snapshot_path(
                        train_percentage=95,
                        shuffle=data_parameters.shuffle,
                        num_epochs=num_epochs,
                    )
                    run_inference_on_all_images(
                        project,
                        snapshot=snapshot,
                        plot=(num_epochs == train_parameters.epochs),
                    )
        except Exception as err:
            print(f"Failed to run {project}: {err}")
