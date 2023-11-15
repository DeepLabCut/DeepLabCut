"""Creates a base model configuration file to train a model on a COCO dataset

"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.generate_training_dataset import MakeInference_yaml, make_pytorch_config
from deeplabcut.pose_estimation_pytorch import COCOLoader


def get_base_config(
    dlc_path: str,
    model_architecture: str,
    bodyparts: list[str],
    unique_bodyparts: list[str],
    individuals: list[str],
) -> dict:
    pytorch_cfg_template = af.read_plainconfig(
        str(Path(dlc_path) / "pose_estimation_pytorch" / "apis" / "pytorch_config.yaml")
    )
    cfg = {
        "bodyparts": bodyparts,
        "unique_bodyparts": unique_bodyparts,
        "individuals": individuals,
    }
    return make_pytorch_config(
        cfg,
        model_architecture,
        config_template=pytorch_cfg_template,
    )


def make_inference_config(
    dlc_path: str,
    output_path: str,
    bodyparts: list[str],
    num_individuals: int,
):
    default_config_path = Path(dlc_path) / "inference_cfg.yaml"
    items2change = {
        "minimalnumberofconnections": int(len(bodyparts) / 2),
        "topktoretain": num_individuals,
        "withid": False,  # TODO: implement
    }
    MakeInference_yaml(items2change, output_path, default_config_path)


def main(project_root: str, train_file: str, output: str, model_arch: str):
    output_path = Path(output)
    if output_path.exists():
        raise RuntimeError(
            f"The output path must not exist yet, as otherwise we would risk overwriting"
            f" existing configurations ({output_path} exists)"
        )

    train_dict = COCOLoader.load_json(project_root, train_file)
    num_individuals, bodyparts = COCOLoader.get_project_parameters(train_dict)
    dlc_path = af.get_deeplabcut_path()
    pytorch_cfg = get_base_config(
        dlc_path=dlc_path,
        model_architecture=model_arch,
        bodyparts=bodyparts,
        unique_bodyparts=[],
        individuals=[f"individual{i}" for i in range(num_individuals)],
    )
    output_path.mkdir(parents=True)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    af.write_plainconfig(str(train_dir / "pytorch_config.yaml"), pytorch_cfg)
    make_inference_config(
        dlc_path,
        str(test_dir / "inference_cfg.yaml"),
        bodyparts,
        num_individuals,
    )
    print(f"Saved your model configuration in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("output")
    parser.add_argument("model_arch")
    parser.add_argument("--train_file", default="train.json")
    args = parser.parse_args()
    main(args.project_root, args.train_file, args.output, args.model_arch)
