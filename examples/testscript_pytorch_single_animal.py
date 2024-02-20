""" Testscript for single animal PyTorch projects """
import time
from pathlib import Path
from typing import Any

import deeplabcut
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.compat import Engine
from deeplabcut.generate_training_dataset import get_existing_shuffle_indices


def run(
    config_path: Path,
    train_fraction: float,
    trainset_index: int,
    net_type: str,
    videos: list[str],
    device: str,
    train_kwargs: dict,
    engine: Engine = Engine.PYTORCH,
    create_labeled_videos: bool = False,
) -> None:
    times = [time.time()]
    log_step(f"Testing with net type {net_type}")
    log_step("Creating the training dataset")
    deeplabcut.create_training_dataset(str(config_path), net_type=net_type, engine=engine)
    existing_shuffles = get_existing_shuffle_indices(
        config_path, train_fraction=train_fraction, engine=engine
    )
    shuffle_index = existing_shuffles[-1]

    log_step(f"Starting training for train_frac {train_fraction}, shuffle {shuffle_index}")
    deeplabcut.train_network(
        config=str(config_path),
        shuffle=shuffle_index,
        trainingsetindex=trainset_index,
        device=device,
        **train_kwargs,
    )
    times.append(time.time())
    log_step(f"Train time: {times[-1] - times[-2]} seconds")

    log_step(f"Starting evaluation for train_frac {train_fraction}, shuffle {shuffle_index}")
    deeplabcut.evaluate_network(
        config=str(config_path),
        Shuffles=[shuffle_index],
        trainingsetindex=trainset_index,
        device=device,
    )
    times.append(time.time())
    log_step(f"Evaluation time: {times[-1] - times[-2]} seconds")

    log_step(f"Analyzing videos for {train_fraction}, shuffle {shuffle_index}")
    deeplabcut.analyze_videos(
        config=str(config_path),
        videos=videos,
        shuffle=shuffle_index,
        trainingsetindex=trainset_index,
        device=device,
    )
    times.append(time.time())
    log_step(f"Video analysis time: {times[-1] - times[-2]} seconds")
    log_step(f"Total test time: {times[-1] - times[0]} seconds")

    if create_labeled_videos:
        log_step(f"Creating a labeled video for {train_fraction}, shuffle {shuffle_index}")
        deeplabcut.create_labeled_video(
            config=str(config_path),
            videos=videos,
            shuffle=shuffle_index,
            trainingsetindex=trainset_index,
        )


def main(
    net_types: list[str],
    epochs: int = 1,
    save_epochs: int = 1,
    batch_size: int = 1,
    device: str = "cpu",
    create_labeled_videos: bool = False,
) -> None:
    engine = Engine.PYTORCH
    project_path = Path.cwd() / "openfield-Pranav-2018-10-30"
    config_path = project_path / "config.yaml"
    cfg = af.read_config(config_path)
    trainset_index = 0
    train_frac = cfg["TrainingFraction"][trainset_index]
    for net_type in net_types:
        try:
            run(
                config_path=config_path,
                train_fraction=train_frac,
                trainset_index=trainset_index,
                net_type=net_type,
                videos=[str(project_path / "videos" / "m3v1mp4.mp4")],
                device=device,
                train_kwargs=dict(
                    display_iters=1,
                    epochs=epochs,
                    save_epochs=save_epochs,
                    batch_size=batch_size,
                ),
                engine=engine,
                create_labeled_videos=create_labeled_videos,
            )
        except Exception as err:
            log_step(f"FAILED TO RUN {net_type}")
            log_step(str(err))
            log_step("Continuing to next model")
            raise err


def log_step(message: Any) -> None:
    print(100 * "-")
    print(str(message))
    print(100 * "-")


if __name__ == "__main__":
    main(
        net_types=["resnet_50", "hrnet_w18", "hrnet_w32"],
        batch_size=8,
        epochs=1,
        save_epochs=1,
        device="cpu",  # "cpu", "cuda:0", "mps"
        create_labeled_videos=False,
    )
