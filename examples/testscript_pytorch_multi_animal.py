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
""" Testscript for single animal PyTorch projects """
from pathlib import Path

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.compat import Engine

from utils import cleanup, create_fake_project, log_step, run


def main(
    net_types: list[str],
    epochs: int = 1,
    save_epochs: int = 1,
    batch_size: int = 1,
    device: str = "cpu",
    create_labeled_videos: bool = False,
    delete_after_test_run: bool = False,
) -> None:
    project_path = Path("../synthetic-data-niels-multi-animal").resolve()
    config_path = project_path / "config.yaml"

    create_fake_project(
        path=project_path,
        multianimal=True,
        num_bodyparts=3,
        num_frames=20,
        num_individuals=2,
        num_unique=4,
        identity=False,
        frame_shape=(128, 256),
    )

    engine = Engine.PYTORCH
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
                videos=[],
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

    if delete_after_test_run:
        cleanup(project_path)


if __name__ == "__main__":
    main(
        net_types=["resnet_50", "dekr_w18"],  # , "hrnet_w18", "hrnet_w32"],
        batch_size=8,
        epochs=1,
        save_epochs=1,
        device="cpu",  # "cpu", "cuda:0", "mps"
        create_labeled_videos=False,
        delete_after_test_run=True,
    )
