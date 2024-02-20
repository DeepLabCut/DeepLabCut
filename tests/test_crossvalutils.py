#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import numpy as np
import pickle
from deeplabcut.core import crossvalutils

BEST_GRAPH = [14, 15, 16, 11, 22, 31, 61, 7, 59, 62, 64]
BEST_GRAPH_MONTBLANC = [1, 0, 2, 5, 4, 3]


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


def test_get_n_best_paf_graphs_montblanc(evaluation_data_and_metadata_montblanc):
    data, metadata = evaluation_data_and_metadata_montblanc
    params = crossvalutils._set_up_evaluation(data)
    paf_inds, dict_ = crossvalutils._get_n_best_paf_graphs(
        data,
        metadata,
        params["paf_graph"],
    )
    assert len(paf_inds) == 4
    assert len(dict_) == len(params["paf_graph"])
    assert [len(inds) for inds in paf_inds] == list(range(3, 7))
    assert paf_inds[-1] == BEST_GRAPH_MONTBLANC
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


def test_benchmark_paf_graphs_montblanc(evaluation_data_and_metadata_montblanc):
    data, metadata = evaluation_data_and_metadata_montblanc
    cfg = {
        "individuals": [f"bird{i}" for i in range(1, 9)],
        "uniquebodyparts": ["center"],
        "multianimalbodyparts": [
            "head",
            "tail",
            "leftwing",
            "rightwing",
        ],
    }
    inference_cfg = {"topktoretain": 8, "pcutoff": 0.1, "pafthreshold": 0.1}
    results = crossvalutils._benchmark_paf_graphs(
        cfg,
        inference_cfg,
        data,
        [BEST_GRAPH_MONTBLANC],
        split_inds=[metadata["data"]["trainIndices"], metadata["data"]["testIndices"]],
    )
    with open("tests/data/montblanc_map.pickle", "rb") as f:
        results_gt = pickle.load(f)
    np.testing.assert_equal(
        results[1].loc["purity"].to_numpy().squeeze(),
        [
            results_gt[0][6][('purity', 'mean')],
            results_gt[0][6][('purity', 'std')],
        ],
    )
    vals = [
        results[2][0][0]["mAP"],
        results[2][0][0]["mAR"],
        results[2][0][1]["mAP"],
        results[2][0][1]["mAR"],
    ]
    np.testing.assert_equal(
        vals,
        [
            results_gt[0][6][('mAP_train', 'mean')],
            results_gt[0][6][('mAR_train', 'mean')],
            results_gt[0][6][('mAP_test', 'mean')],
            results_gt[0][6][('mAR_test', 'mean')],
        ],
    )
