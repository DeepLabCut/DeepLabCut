"""Fine-tuning SuperAnimal models"""
from __future__ import annotations

import pickle
from pathlib import Path

import deeplabcut
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset import TrainingDatasetMetadata
from deeplabcut.modelzoo.utils import create_conversion_table
from deeplabcut.pose_estimation_pytorch import DLCLoader


def create_shuffles(config_path: Path, super_animal: str):
    """
    Iteration 0: the three data splits given by Shaokai
    Iteration 1: trained model on these data splits
        Shuffle 1001: Split 1,   1% data
        Shuffle 1005: Split 1,   5% data
        Shuffle 1010: Split 1,  10% data
        Shuffle 1050: Split 1,  50% data
        Shuffle 1100: Split 1, 100% data

        Shuffle 2001: Split 2,   1% data
        Shuffle 2005: Split 2,   5% data
        Shuffle 2010: Split 2,  10% data
        Shuffle 2050: Split 2,  50% data
        Shuffle 2100: Split 2, 100% data

        Shuffle 3001: Split 3,   1% data
        Shuffle 3005: Split 3,   5% data
        Shuffle 3010: Split 3,  10% data
        Shuffle 3050: Split 3,  50% data
        Shuffle 3100: Split 3, 100% data
    """
    project_path = config_path.parent
    split_folder = (
        project_path /
        "training-datasets" /
        "iteration-0" /
        "UnaugmentedDataSet_openfieldAug20"
    )

    data_splits = [1, 2, 3]
    frac_train_data_used = [0.01, 0.05, 0.1, 0.5, 1]

    shuffles = []
    train_indices = []
    test_indices = []

    for split in data_splits:
        split_name = f"Documentation_data-openfield_95shuffle{split - 1}.pickle"
        path_metadata = split_folder / split_name
        with open(path_metadata, "rb") as f:
            _, train_idx, test_idx, _ = pickle.load(f)

        num_train_images = len(train_idx)
        for frac in frac_train_data_used:
            shuffle_idx = (1000 * split) + int(100 * frac)
            num_samples = int(frac * num_train_images)  # as done by Shaokai

            shuffles.append(shuffle_idx)
            train_indices.append(train_idx[:num_samples])
            test_indices.append(test_idx)

    cfg = af.read_config(config_path)
    weight_init = WeightInitialization.build(cfg, super_animal, with_decoder=True)
    deeplabcut.create_training_dataset(
        str(config_path),
        Shuffles=shuffles,
        trainIndices=train_indices,
        testIndices=test_indices,
        userfeedback=False,
        weight_init=weight_init,
        engine=Engine.PYTORCH,
    )


def create_transfer_learning_shuffles(
    config_path: Path,
    net_type: str,
    super_animal: str | None,
):
    project_path = config_path.parent
    split_folder = (
        project_path /
        "training-datasets" /
        "iteration-0" /
        "UnaugmentedDataSet_openfieldAug20"
    )

    data_splits = [1, 2, 3]
    frac_train_data_used = [0.01, 0.05, 0.1, 0.5, 1]

    weight_init = None
    if super_animal is not None:
        weight_init = WeightInitialization(dataset=super_animal, with_decoder=False)

    shuffles = []
    train_indices = []
    test_indices = []

    for split in data_splits:
        split_name = f"Documentation_data-openfield_95shuffle{split - 1}.pickle"
        path_metadata = split_folder / split_name
        with open(path_metadata, "rb") as f:
            _, train_idx, test_idx, _ = pickle.load(f)

        num_train_images = len(train_idx)
        for frac in frac_train_data_used:
            if super_animal == "superanimal_topviewmouse":
                shuffle_idx = 50_000 + (1000 * split) + int(100 * frac)
            elif super_animal is None:
                shuffle_idx = 90_000 + (1000 * split) + int(100 * frac)
            else:
                raise ValueError(f"Failed to generate shuffles for super_animal={super_animal}")

            num_samples = int(frac * num_train_images)  # as done by Shaokai
            shuffles.append(shuffle_idx)
            train_indices.append(train_idx[:num_samples])
            test_indices.append(test_idx)

    deeplabcut.create_training_dataset(
        str(config_path),
        net_type=net_type,
        Shuffles=shuffles,
        trainIndices=train_indices,
        testIndices=test_indices,
        userfeedback=False,
        weight_init=weight_init,
        engine=Engine.PYTORCH,
    )


def update_cfg(
    config_path: Path,
    shuffle: int,
    train_augmentations: dict,
    optimizer: dict,
    scheduler: dict,
) -> None:
    loader = DLCLoader(
        config=config_path,
        shuffle=shuffle,
        trainset_index=0,
        modelprefix="",
    )
    loader.model_cfg["data"]["train"] = None
    loader.model_cfg["runner"]["optimizer"] = None
    loader.model_cfg["runner"]["scheduler"] = None
    loader.update_model_cfg({
        "data": {"train": train_augmentations},
        "runner": {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
    })


def data_preparation(
    config_path: Path,
    super_animal: str,
    run_build_conversion_table: bool,
    run_create_shuffles: bool,
) -> None:
    if run_build_conversion_table:
        _ = create_conversion_table(
            config=config_path,
            super_animal=super_animal,
            project_to_super_animal={
                "snout": "nose",
                "leftear": "left_ear",
                "rightear": "right_ear",
                "tailbase": "tail_base",
            },
        )

    if run_create_shuffles:
        create_shuffles(config_path, super_animal)


def main(
    config_path: Path,
    shuffle_index: int,
    epochs: int,
    train_augmentations: dict,
    optimizer: dict,
    scheduler: dict,
    device: str | None = None,
    batch_size: int = 32,
    save_epochs: int = 20,
    eval_interval: int = 5,
):
    metadata = TrainingDatasetMetadata.load(config_path, load_splits=True)
    shuffles = [s for s in metadata.shuffles if s.index == shuffle_index]
    if len(shuffles) != 1:
        raise ValueError(
            "Found multiple shuffles with different train indices but the same index "
            f"({shuffles}). To run this benchmark, there should only be one such "
            "shuffle."
        )

    shuffle = shuffles[0]
    print(f"Training shuffle: {shuffle.name}")
    print(f"  index: {shuffle.index}")
    print(f"  train fraction: {shuffle.train_fraction}")
    print(f"  train indices: {shuffle.split.train_indices}")
    print(f"  test indices: {shuffle.split.test_indices}")
    print()

    # edit config to have the desired training fraction
    af.edit_config(str(config_path), {"TrainingFraction": [shuffle.train_fraction]})

    # information about shuffle
    mode = shuffle.index // 10_000
    split = (shuffle.index - (10_000 * mode)) // 1000
    data_used = (shuffle.index - (10_000 * mode)) % 1000

    project = "SuperAnimal-openfield-finetune-v2"
    uid = "sa-finetune"
    if mode == 5:
        uid = "sa-transfer"
    elif mode == 9:
        uid = "in-transfer"

    # update the pose config to have the correct augmentation, optimizer and scheduler
    update_cfg(
        config_path=config_path,
        shuffle=shuffle.index,
        train_augmentations=train_augmentations,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # train the model
    deeplabcut.train_network(
        str(config_path),
        shuffle=shuffle.index,
        trainingsetindex=0,
        device=device,
        # edit the pytorch config
        detector=dict(train_settings=dict(epochs=0)),
        runner=dict(
            eval_interval=eval_interval,
            snapshots=dict(
                max_snapshots=5,
                save_epochs=save_epochs,
            ),
        ),
        train_settings=dict(batch_size=batch_size, epochs=epochs),
        logger=dict(
            type="WandbLogger",
            project_name=project,
            run_name=f"openfield-{uid}-shuffle{shuffle.index}",
            save_code=True,
            tags=(
                f"mode={uid}",
                f"split={split}",
                f"data_used={data_used}",
            )
        ),
    )


if __name__ == "__main__":
    DATA = Path("/home/niels/datasets/superanimal")
    CONFIG_PATH = DATA / "openfield-Pranav-2018-08-20" / "config.yaml"
    SUPER_ANIMAL = "superanimal_topviewmouse"

    FINETUNE_AUG = {
        "affine": {
            "p": 0.5,
            "scaling": [1, 1],
            "rotation": 90,
            "translation": 0,
        },
        "hflip": {
            "p": 0.5,
            "symmetries": [[1, 2]],
        },
        "gaussian_noise": 12.75,
        "normalize_images": True,
    }
    FINETUNE_OPTIM = {
        "type": "AdamW",
        "params": {"lr": 1e-05},
    }
    FINETUNE_SCHEDULER = {
        "type": "LRListScheduler",
        "params": {
            "lr_list": [[1e-06], [1e-07]],
            "milestones": [450, 590],
        },
    }

    PREP_DATA = False
    PREP_TRANSFER_LEARNING_DATA = False
    if PREP_DATA:
        # ONLY RUN ONCE: prepare data (create shuffles, conversion table)
        data_preparation(
            config_path=CONFIG_PATH,
            super_animal=SUPER_ANIMAL,
            run_build_conversion_table=True,
            run_create_shuffles=True,
        )
    elif PREP_TRANSFER_LEARNING_DATA:
        create_transfer_learning_shuffles(
            CONFIG_PATH, "top_down_hrnet_w32", SUPER_ANIMAL
            )
        create_transfer_learning_shuffles(CONFIG_PATH, "top_down_hrnet_w32", None)
    else:
        # train a shuffle
        for idx in [51001, 52001, 53001, 51005, 52005, 53005]:
            main(
                config_path=CONFIG_PATH,
                shuffle_index=idx,
                epochs=600,
                train_augmentations=FINETUNE_AUG,
                optimizer=FINETUNE_OPTIM,
                scheduler=FINETUNE_SCHEDULER,
                batch_size=32,
                device="cuda",
            )
