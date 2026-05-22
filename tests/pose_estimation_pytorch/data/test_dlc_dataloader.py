from types import SimpleNamespace

import numpy as np
import pandas as pd

import deeplabcut.pose_estimation_pytorch.data.dlcloader as dlcloader_mod
from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader


def test_to_coco_ignores_likelihood_columns(monkeypatch, tmp_path):
    fake_shape = (3, 480, 640)
    monkeypatch.setattr(
        dlcloader_mod,
        "read_image_shape_fast",
        lambda _: fake_shape,
    )

    scorer = "testscorer"
    bodyparts = ["nose", "tail"]

    index = pd.MultiIndex.from_tuples(
        [("labeled-data", "video1", "img0001.png")],
        names=["set", "video", "image"],
    )

    # Baseline dataframe: x/y only
    columns_xy = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df_xy = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0]],
        index=index,
        columns=columns_xy,
    )

    # Same data, but with likelihood columns added
    columns_xyl = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    df_xyl = pd.DataFrame(
        [[10.0, 20.0, 0.9, 30.0, 40.0, 0.8]],
        index=index,
        columns=columns_xyl,
    )

    # to_coco only needs these attributes from parameters
    params = SimpleNamespace(
        bodyparts=bodyparts,
        unique_bpts=[],
        individuals=["animal"],
    )

    baseline = DLCLoader.to_coco(tmp_path, df_xy, params)
    got = DLCLoader.to_coco(tmp_path, df_xyl, params)

    assert len(got["images"]) == len(baseline["images"]) == 1
    assert len(got["annotations"]) == len(baseline["annotations"]) == 1

    got_ann = got["annotations"][0]
    expected_ann = baseline["annotations"][0]

    assert got_ann["image_id"] == expected_ann["image_id"]
    assert got_ann["category_id"] == expected_ann["category_id"]
    assert got_ann["num_keypoints"] == expected_ann["num_keypoints"] == 2
    assert np.array_equal(got_ann["keypoints"], expected_ann["keypoints"])
    assert np.allclose(got_ann["bbox"], expected_ann["bbox"])
