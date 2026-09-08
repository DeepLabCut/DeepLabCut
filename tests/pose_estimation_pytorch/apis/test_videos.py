from pathlib import Path

import pandas as pd

import deeplabcut.pose_estimation_pytorch.apis.videos as videos

POSE_CFG = {
    "all_joints": [[0], [1], [2]],
    "all_joints_names": ["snout", "leftear", "rightear"],
    "nmsradius": 5,
    "minconfidence": 0.1,
    "sigma": 1,
}


def test_generate_output_data_handles_empty_predictions():
    output = videos._generate_output_data(POSE_CFG, [])

    assert output == {
        "metadata": {
            "nms radius": 5,
            "minimal confidence": 0.1,
            "sigma": 1,
            "PAFgraph": None,
            "PAFinds": [],
            "all_joints": [[0], [1], [2]],
            "all_joints_names": ["snout", "leftear", "rightear"],
            "nframes": 0,
            "key_str_width": 1,
        }
    }


def test_create_df_from_prediction_handles_empty_predictions(monkeypatch, tmp_path):
    writes: list[tuple[Path, str, str, str]] = []

    def _fake_to_hdf(self, path_or_buf, key, format, mode):
        writes.append((Path(path_or_buf), key, format, mode))

    monkeypatch.setattr(pd.DataFrame, "to_hdf", _fake_to_hdf)

    df = videos.create_df_from_prediction(
        predictions=[],
        dlc_scorer="DLC_test",
        multi_animal=False,
        model_cfg={
            "metadata": {
                "bodyparts": ["snout", "leftear", "rightear"],
                "unique_bodyparts": [],
                "individuals": ["animal_0"],
            }
        },
        output_path=tmp_path,
        output_prefix="video",
        save_as_csv=False,
    )

    assert df.empty
    assert list(df.columns.names) == ["scorer", "bodyparts", "coords"]
    assert writes == [(tmp_path / "video.h5", "df_with_missing", "table", "w")]
