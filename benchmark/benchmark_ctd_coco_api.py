
from __future__ import annotations

import csv
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ruamel.yaml import YAML


@lru_cache(maxsize=None)
def read_image_shape_fast(path: str | Path) -> tuple[int, int, int]:
    """Blazing fast and does not load the image into memory"""
    with Image.open(path) as img:
        width, height = img.size
        return len(img.getbands()), height, width


def bbox_from_keypoints(
    keypoints: np.ndarray,
    image_h: int,
    image_w: int,
    margin: int,
) -> np.ndarray:
    squeeze = False
    if len(keypoints.shape) == 2:
        squeeze = True
        keypoints = np.expand_dims(keypoints, axis=0)

    bboxes = np.full((keypoints.shape[0], 4), np.nan)
    bboxes[:, :2] = np.nanmin(keypoints[..., :2], axis=1) - margin  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(keypoints[..., :2], axis=1) + margin  # X2, Y2
    bboxes = np.clip(
        bboxes,
        a_min=[0, 0, 0, 0],
        a_max=[image_w, image_h, image_w, image_h],
    )
    bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0]  # to width
    bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1]  # to height
    if squeeze:
        return bboxes[0]

    return bboxes


def to_coco(
    df: pd.DataFrame,
    project_dir: Path,
    individuals: list[str],
    bodyparts: list[str],
    unique_bpts: list[str],
    bbox_margin: int = 0,
    image_size: tuple[int, int] | None = None
) -> dict:
    with_individuals = "individuals" in df.columns.names
    categories = [
        {
            "id": 1,
            "name": "animals",
            "supercategory": "animal",
            "keypoints": bodyparts,
            "skeleton": [],
        },
    ]
    individuals = [idv for idv in individuals]
    if len(unique_bpts) > 0:
        individuals += ["single"]
        categories.append(
            {
                "id": 2,
                "name": "unique_bodypart",
                "supercategory": "animal",
                "keypoints": unique_bpts,
            }
        )

    anns, images = [], []
    for idx, row in df.iterrows():
        image_id = len(images) + 1
        rel_path = Path(*idx) if isinstance(idx, tuple) else Path(str(idx))
        path = str(project_dir / rel_path)
        if image_size is None:
            _, height, width = read_image_shape_fast(path)
        else:
            width, height = image_size

        images.append(
            {
                "id": image_id,
                "file_name": path,
                "width": width,
                "height": height,
            }
        )

        for idv_idx, idv in enumerate(individuals):
            category_id = 1
            if with_individuals:
                if idv == "single":
                    category_id = 2
                data = row.xs(idv, level="individuals")
            else:
                data = row

            mask = np.array([(bpt in bodyparts) for bpt in data.index.get_level_values(level="bodyparts")])
            raw_keypoints = data.to_numpy()
            raw_keypoints = raw_keypoints[mask]
            raw_keypoints = raw_keypoints.reshape((-1, 2))

            keypoints = np.zeros((len(raw_keypoints), 3))
            keypoints[:, :2] = raw_keypoints
            is_visible = np.logical_and(
                ~pd.isnull(raw_keypoints).all(axis=1),
                np.logical_and(
                    np.logical_and(
                        0 < keypoints[..., 0],
                        keypoints[..., 0] < width,
                    ),
                    np.logical_and(
                        0 < keypoints[..., 1],
                        keypoints[..., 1] < height,
                    ),
                )
            )
            keypoints[:, 2] = np.where(is_visible, 2, 0)
            num_keypoints = is_visible.sum()
            if num_keypoints > 1:                           # TODO: AT LEAST 2 KEYPOINTS TO COMPUTE mAP
                bbox = bbox_from_keypoints(
                    keypoints=keypoints,
                    image_h=height,
                    image_w=width,
                    margin=bbox_margin,
                )
                area = bbox[2].item() * bbox[3].item()
                anns.append(
                    {
                        "id": len(anns) + 1,
                        "image_id": image_id,
                        "category_id": category_id,
                        "area": area,
                        "bbox": bbox,
                        "keypoints": keypoints.reshape(-1).tolist(),
                        "iscrowd": 0,
                        "num_keypoints": num_keypoints,
                    }
                )

    return {"annotations": anns, "categories": categories, "images": images}


def to_coco_predictions(
    project_dir: Path,
    images: list[dict],
    individuals: list[str],
    bodyparts: list[str],
    df_pred: pd.DataFrame,
) -> list[dict]:
    image_path_to_image = {image["file_name"]: image for image in images}
    coco_predictions = []

    for idx, row in df_pred.iterrows():
        rel_path = Path(*idx) if isinstance(idx, tuple) else Path(str(idx))
        full_path = str(project_dir / rel_path)
        if full_path not in image_path_to_image:
            continue  # train image

        image = image_path_to_image[full_path]
        image_id = image["id"]
        image_h, image_w = image["height"], image["width"]

        image_keypoints = row.to_numpy().reshape((len(individuals), len(bodyparts), 3))
        for keypoints in image_keypoints:
            if np.any(~np.isnan(keypoints)):
                score = np.nanmean(keypoints[:, 2]).item()
                keypoints = keypoints.copy()
                keypoints[:, 2] = 2

                bbox = bbox_from_keypoints(
                    keypoints=keypoints,
                    image_h=image_h,
                    image_w=image_w,
                    margin=0,
                )
                area = bbox[2].item() * bbox[3].item()

                # NaN predictions to infinity
                keypoints[np.isnan(keypoints)] = np.inf

                coco_pred = {
                    "image_id": int(image_id),
                    "category_id": 1,  # TODO: get category ID from prediction?
                    "keypoints": keypoints.reshape(-1).tolist(),
                    "bbox": bbox,
                    "area": area,
                    "score": float(score),
                }
                coco_predictions.append(coco_pred)
            # else:
            #     print(f"REMOVING {keypoints}")
    
    return coco_predictions


def load_split(file: Path) -> tuple[list[int], list[int]]:
    with open(file, "rb") as f:
        data = pickle.load(f)

    train_idx = sorted([int(idx) for idx in data[1]])  # ARE RESULTS INCONSISTENT AS WE DONT SORT INDICES?
    test_idx = sorted([int(idx) for idx in data[2]])
    return list(train_idx), list(test_idx)


def load_prediction_filenames(
    project_prefix: str,
    output_folder: Path,
) -> dict[str, Path]:
    shuffles = [p for p in output_folder.iterdir() if p.is_dir() and p.name.startswith(project_prefix)]
    return {
        p.stem.split("DLC_")[1]: p
        for shuffle in shuffles
        for p in shuffle.iterdir()
        if p.suffix == ".h5"
    }


def evaluate(results_path: Path, paths: dict[str, dict], bbox_margin: int = 0, plot: bool = False):
    results = []

    for project, data_paths in paths.items():
        print((3 * (100 * "-" + "\n"))[:-1])
        print(f"COCO EVALUATION RESULTS FOR {project}")
    
        _, test_idx = load_split(data_paths["split"])
        df_gt = pd.read_hdf(data_paths["gt"])
        df_test = df_gt.iloc[test_idx]

        reader = YAML()
        with open(data_paths["project"] / "config.yaml", "r") as f:
            cfg = reader.load(f)
    
        coco_test_dict = to_coco(
            df_test,
            data_paths["project"],
            individuals=cfg["individuals"],
            bodyparts=cfg["multianimalbodyparts"],
            unique_bpts=cfg["uniquebodyparts"],
            bbox_margin=bbox_margin,
            image_size=(4096, 4096),  # not needed
        )

        for scorer, pred_path in data_paths["predictions"].items():
            print(100 * "-")
            print(pred_path.name)
            print("Scorer", scorer)
            print(100 * "-")
            df_predictions = pd.read_hdf(pred_path)
            predictions = to_coco_predictions(
                data_paths["project"],
                coco_test_dict["images"],
                individuals=cfg["individuals"],
                bodyparts=cfg["multianimalbodyparts"],
                df_pred=df_predictions,
            )

            coco = COCO()
            coco.dataset["annotations"] = coco_test_dict["annotations"]
            coco.dataset["categories"] = coco_test_dict["categories"]
            coco.dataset["images"] = coco_test_dict["images"]
            coco.createIndex()

            coco_det = coco.loadRes(predictions)
            coco_eval = COCOeval(coco, coco_det, iouType="keypoints")
            coco_eval.params.kpt_oks_sigmas = np.array(len(cfg["multianimalbodyparts"]) * [0.1])
            # coco_eval.params.areaRng = [coco_eval.params.areaRng[0]]
            # coco_eval.params.areaRngLbl = [coco_eval.params.areaRngLbl[0]]

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            results.append((scorer, project.upper(), coco_eval.stats[0]))

            # for p in predictions:
            #     print(p)
    
            # if plot:
            #     image_id = 1
            #     image_data = coco.loadImgs(ids=[image_id])[0]
            #     image = Image.open(image_data["file_name"])    
            #     ann_ids = coco.getAnnIds(imgIds=[image_id])
            #     anns = coco.loadAnns(ids=ann_ids)
            #     plt.imshow(image)
            #     coco.showAnns(anns)

        print((3 * (100 * "-" + "\n"))[:-1])
    
    print(f"Saving to {results_path}")
    with open(results_path, "w") as f:
        writer = csv.writer(f, delimiter='\t')
        for row in results:
            writer.writerow(row)


def main():
    dlc_root_dir = Path("/home/lucas/datasets/")
    root_gt = dlc_root_dir / "ground_truth_test"
    root_preds = Path("/home/lucas/datasets/test-images/fish-dlc-2021-05-07/evaluation-results/iteration-30/fishMay7-trainset94shuffle41/benchmark_BU_EfficientNet/")

    predictions = {"BU_model": dlc_root_dir / "benchmark_chkpts/fish/DLC_EfficientNet_B7_s4_30k.h5",
                   "CTD_90": root_preds / "fishMay7-trainset94shuffle41-snapshot-090.h5",
                   "CTD_120": root_preds / "fishMay7-trainset94shuffle41-snapshot-120.h5",
                   "CTD_150": root_preds / "fishMay7-trainset94shuffle41-snapshot-150.h5",
                   "CTD_180": root_preds / "fishMay7-trainset94shuffle41-snapshot-180.h5",
                   "CTD_210": root_preds / "fishMay7-trainset94shuffle41-snapshot-210.h5"}

    paths = {
        # "fish": {
        #     "project": dlc_root_dir / "fish-dlc-2021-05-07",
        #     "gt": root_gt / "CollectedData_Valentina.h5",
        #     "split": root_gt / "Documentation_data-Schooling_70shuffle1.pickle",
        #     "predictions": load_prediction_filenames("fish", root_preds),
        # },
        # "trimou/media1/data/lucas/DLC_projects/ma_dlc/test-images/fish-dlc-2021-05-07/evaluation-results/iteration-30/fishMay7-trainset94shuffle41/benchmark_BU_EfficientNetse": {
        #     "project": dlc_root_dir / "trimice-dlc-2021-06-22",
        #     "gt": root_gt / "CollectedData_Daniel.h5",
        #     "split": root_gt / "Documentation_data-MultiMouse_70shuffle1.pickle",
        #     "predictions": load_prediction_filenames("trimice", root_preds),
        # },
        # "marmoset": {
        #     "project": dlc_root_dir / "marmoset-dlc-2021-05-07",
        #     "gt": root_gt / "CollectedData_Mackenzie.h5",
        #     "split": root_gt / "Documentation_data-Marmoset_70shuffle1.pickle",
        #     "predictions": load_prediction_filenames("marmoset", root_preds),
        # },

        "fish": {
            "project": dlc_root_dir / "fish-dlc-2021-05-07",
            "gt": root_gt / "CollectedData_Valentina.h5",
            "split": root_gt / "Documentation_data-Schooling_70shuffle1.pickle",
            "predictions": predictions,
        },
    }
    evaluate(root_preds / "pycocotools.csv", paths, bbox_margin=0, plot=False)


if __name__ == "__main__":
    main()
