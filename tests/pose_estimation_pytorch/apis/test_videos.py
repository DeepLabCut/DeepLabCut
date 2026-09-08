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
