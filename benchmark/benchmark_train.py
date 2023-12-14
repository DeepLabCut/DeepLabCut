"""Training models on DLC benchmark datasets

In a first step, shuffles can be created for your projects (pass an empty list and no
shuffles are created).

Then you can train models using RunParameters. I usually create the shuffles first,
modify the PyTorch configuration files to add a logger and modify the data augmentation
for whatever I'm doing, and then start my training runs. A logger can be added with:
```
logger:
 type: 'WandbLogger'
 project_name: 'dlc3-ff5f2af-fish'
 run_name: 'dekr-w32-shuffle3'
```

Which specifies to log the run to wandb, (including the project and with which name each
shuffle should be logged).

For single animal projects, benchmark splits were created using the
`create_train_test_splits.py` file. This script creates a JSON file for DLC projects
specifying train/test indices, which can then be passed in the ShuffleCreationParameters
to create new shuffles with the splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import wandb

import deeplabcut
import deeplabcut.pose_estimation_pytorch.apis as api

from projects import MA_DLC_BENCHMARKS, SA_DLC_BENCHMARKS
from utils import Shuffle


@dataclass
class TrainParameters:
    """Parameters to train models"""
    batch_size: int = 16
    display_iters: int = 500
    epochs: int | None = 100
    save_epochs: int | None = 25
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
    """Parameters for evaluation"""
    snapshotindex: int | list[int] | str | None = (None,)
    plotting: str | bool = False
    show_errors: bool = True

    def eval_kwargs(self) -> dict:
        return {
            "plotting": self.plotting,
            "show_errors": self.show_errors,
        }


@dataclass
class VideoAnalysisParameters:
    """Parameters to run video analysis"""
    videos: list[str]
    videotype: str
    output_folder: str = ""


@dataclass
class RunParameters:
    """Parameters on what to run for each shuffle"""
    shuffle: Shuffle
    train: bool = False
    evaluate: bool = False
    analyze_videos: bool = False
    track: bool = False
    create_labeled_video: bool = False
    device: str = "cuda:0"
    train_params: TrainParameters | None = None
    eval_params: EvalParameters | None = None
    video_analysis_params: VideoAnalysisParameters | None = None

    def __post_init__(self):
        if (
            (self.analyze_videos is None or self.track or self.create_labeled_video)
            and self.video_analysis_params is None
        ):
            raise ValueError(f"Must specify video_analysis_params")


def run_dlc(parameters: RunParameters) -> None:
    """Runs DeepLabCut 3.0 API methods

    Args:
        parameters: the parameters specifying what to run, and which parameters to use
    """
    if parameters.train:
        api.train_network(
            str(parameters.shuffle.project.config_path()),
            shuffle=parameters.shuffle.index,
            trainingsetindex=parameters.shuffle.trainset_index,
            transform=None,
            modelprefix=parameters.shuffle.model_prefix,
            device=parameters.device,
            **parameters.train_params.train_kwargs(),
        )

    if parameters.evaluate:
        snapshot_indices = parameters.eval_params.snapshotindex
        if isinstance(snapshot_indices, int) or isinstance(snapshot_indices, str):
            snapshot_indices = [snapshot_indices]

        for idx in snapshot_indices:
            api.evaluate_network(
                config=str(parameters.shuffle.project.config_path()),
                shuffles=[parameters.shuffle.index],
                trainingsetindex=parameters.shuffle.trainset_index,
                snapshotindex=idx,
                device=parameters.device,
                transform=None,
                modelprefix=parameters.shuffle.model_prefix,
                **parameters.eval_params.eval_kwargs(),
            )

    if parameters.analyze_videos:
        destfolder = parameters.shuffle.project.path / parameters.video_analysis_params.output_folder
        api.analyze_videos(
            config=str(parameters.shuffle.project.config_path()),
            videos=parameters.video_analysis_params.videos,
            videotype=parameters.video_analysis_params.videotype,
            trainingsetindex=parameters.shuffle.trainset_index,
            destfolder=str(destfolder),
            snapshotindex=5,
            device=parameters.device,
            modelprefix=parameters.shuffle.model_prefix,
            batchsize=parameters.train_params.batch_size,
            transform=None,
            overwrite=False,
            auto_track=False,
        )

    if parameters.track:
        destfolder = parameters.shuffle.project.path / parameters.video_analysis_params.output_folder
        api.convert_detections2tracklets(
            config=str(parameters.shuffle.project.config_path()),
            videos=parameters.video_analysis_params.videos,
            videotype=".mp4",
            modelprefix=parameters.shuffle.model_prefix,
            destfolder=str(destfolder),
            track_method="box",
        )
        deeplabcut.stitch_tracklets(
            str(parameters.shuffle.project.config_path()),
            videos=parameters.video_analysis_params.videos,
            shuffle=1,
            trainingsetindex=parameters.shuffle.trainset_index,
            destfolder=str(destfolder),
            modelprefix=parameters.shuffle.model_prefix,
            save_as_csv=True,
            track_method="box",
        )

    if parameters.create_labeled_video:
        destfolder = parameters.shuffle.project.path / parameters.video_analysis_params.output_folder
        deeplabcut.create_labeled_video(
            config=str(parameters.shuffle.project.config_path()),
            videos=parameters.video_analysis_params.videos,
            videotype="mp4",
            trainingsetindex=parameters.shuffle.trainset_index,
            color_by="individual",  # bodypart, individual
            modelprefix=parameters.shuffle.model_prefix,
            destfolder=str(destfolder),
            track_method="box",
        )
    return


def main(runs: list[RunParameters]) -> None:
    """Runs benchmarking scripts for DeepLabCut

    Args:
        runs:
    """
    for run in runs:
        run.shuffle.project.update_iteration_in_config()

        if wandb.run is not None:  # TODO: Finish wandb run in DLC
            wandb.finish()

        print(f"Running {run.shuffle}")
        try:
            run_dlc(run)
        except Exception as err:
            print(f"Failed to run {run}: {err}")
            raise err


if __name__ == "__main__":
    main(
        runs=[
            RunParameters(
                shuffle=Shuffle(
                    project=SA_DLC_BENCHMARKS["fly"],
                    train_fraction=0.8,
                    index=1,
                    model_prefix="",
                ),
                train=True,
                evaluate=True,
                analyze_videos=False,
                track=False,
                create_labeled_video=False,
                device="cuda:0",
                train_params=TrainParameters(
                    batch_size=8, epochs=125, save_epochs=25,
                ),
                eval_params=EvalParameters(
                    snapshotindex="all", plotting=False
                )
            ),
        ]
    )
