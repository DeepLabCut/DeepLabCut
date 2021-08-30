import numpy as np
from deeplabcut.pose_estimation_tensorflow.lib import crossvalutils


BEST_GRAPH = [14, 15, 16, 11, 22, 31, 61, 7, 59, 62, 64]


def test_get_n_best_paf_graphs(evaluation_data_and_metadata):
    data, metadata = evaluation_data_and_metadata
    params = crossvalutils._set_up_evaluation(data)
    n_graphs = 5
    paf_inds, dict_ = crossvalutils._get_n_best_paf_graphs(
        data, metadata, params["paf_graph"], n_graphs=n_graphs
    )
    assert len(paf_inds) == n_graphs
    assert len(dict_) == len(params["paf_graph"])
    assert len(paf_inds[0]) == 11
    assert paf_inds[0] == BEST_GRAPH
    assert len(paf_inds[-1]) == len(params["paf_graph"])


def test_benchmark_paf_graphs(evaluation_data_and_metadata):
    data, _ = evaluation_data_and_metadata
    cfg = {
        "individuals": ["mickey", "minnie", "bianca"],
        "uniquebodyparts": [],
        "multianimalbodyparts": [
            "snout",
            "leftear",
            "rightear",
            "shoulder",
            "spine1",
            "spine2",
            "spine3",
            "spine4",
            "tailbase",
            "tail1",
            "tail2",
            "tailend",
        ],
    }
    inference_cfg = {"topktoretain": 3, "pcutoff": 0.1, "pafthreshold": 0.1}
    results = crossvalutils._benchmark_paf_graphs(
        cfg, inference_cfg, data, [BEST_GRAPH]
    )
    all_scores = results[0]
    assert len(all_scores) == 1
    assert all_scores[0][1] == BEST_GRAPH
    miss, purity = results[1].xs("mean", level=1).to_numpy().squeeze()
    assert np.isclose(miss, 0.02, atol=1e-2)
    assert np.isclose(purity, 0.98, atol=1e-2)
