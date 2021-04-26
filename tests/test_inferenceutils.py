import numpy as np
import os
import pickle
import pytest
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils
from scipy.spatial.distance import squareform


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
pickled_data = os.path.join(TEST_DATA_DIR, "trimouse_full.pickle")
pickled_assemblies = os.path.join(TEST_DATA_DIR, "trimouse_assemblies.pickle")


@pytest.fixture(scope="module")
def ground_truth_assemblies():
    with open(pickled_assemblies, "rb") as file:
        temp = pickle.load(file)
    data = np.stack(list(temp.values()))
    return inferenceutils._parse_ground_truth_data(data[..., :3])


def test_calc_bboxes_from_keypoints():
    width = 200
    height = width * 2
    xyp = np.zeros((1, 2, 3))
    xyp[:, 1, :2] = width, height
    xyp[:, 1, 2] = 1
    with pytest.raises(ValueError):
        _ = inferenceutils.calc_bboxes_from_keypoints(xyp[..., :2])

    bboxes = inferenceutils.calc_bboxes_from_keypoints(xyp)
    np.testing.assert_equal(bboxes, [[0, 0, width, height, 0.5]])

    slack = 20
    bboxes = inferenceutils.calc_bboxes_from_keypoints(xyp, slack=slack)
    np.testing.assert_equal(bboxes, [[-slack, -slack, width + slack, height + slack, 0.5]])

    offset = 50
    bboxes = inferenceutils.calc_bboxes_from_keypoints(xyp, offset=offset)
    np.testing.assert_equal(bboxes, [[offset, 0, width + offset, height, 0.5]])


def test_conv_square_to_condensed_indices():
    n = 5
    rows, cols = np.triu_indices(n, k=1)
    mat = np.zeros((n, n), dtype=int)
    mat[rows, cols] = mat[cols, rows] = np.arange(1, len(rows) + 1)
    vec = squareform(mat)
    vals = []
    for i, j in zip(rows, cols):
        ind = inferenceutils._conv_square_to_condensed_indices(i, j, n)
        vals.append(vec[ind])
    np.testing.assert_equal(vec, vals)


def test_calc_object_keypoint_similarity(ground_truth_assemblies):
    sigma = 0.01
    xy1 = ground_truth_assemblies[0][0].xy
    xy2 = ground_truth_assemblies[0][1].xy
    assert inferenceutils.calc_object_keypoint_similarity(xy1, xy1, sigma) == 1
    assert np.isclose(inferenceutils.calc_object_keypoint_similarity(xy1, xy2, sigma), 0)
    xy3 = xy1.copy()
    xy3[:len(xy3) // 2] = np.nan
    assert inferenceutils.calc_object_keypoint_similarity(xy3, xy1, sigma) == 0.5
    xy3[:] = np.nan
    assert inferenceutils.calc_object_keypoint_similarity(xy3, xy1, sigma) == 0
    assert np.isnan(inferenceutils.calc_object_keypoint_similarity(xy1, xy3, sigma))


def test_match_assemblies(ground_truth_assemblies):
    assemblies = ground_truth_assemblies[0]
    matched, unmatched = inferenceutils.match_assemblies(
        assemblies, assemblies[::-1], 0.01
    )
    assert not unmatched
    for ass1, ass2, oks in matched:
        assert ass1 is ass2
        assert oks == 1

    matched, unmatched = inferenceutils.match_assemblies(
        [], assemblies, 0.01
    )
    assert not matched
    assert all(ass1 is ass2 for ass1, ass2 in zip(unmatched, assemblies))


def test_evaluate_assemblies(ground_truth_assemblies):
    assemblies = {i: ground_truth_assemblies[i] for i in range(3)}
    n_thresholds = 5
    thresholds = np.linspace(0.5, 0.95, n_thresholds)
    dict_ = inferenceutils.evaluate_assembly(
        assemblies, assemblies, oks_thresholds=thresholds
    )
    assert dict_["mAP"] == dict_["mAR"] == 1
    assert len(dict_["precisions"]) == len(dict_["recalls"]) == n_thresholds
    assert dict_["precisions"].shape[1] == 101
    np.testing.assert_allclose(dict_["precisions"], 1)


def test_link():
    pos1 = 1, 1
    idx1 = 0
    pos2 = 10, 10
    idx2 = 1
    conf = 0.5
    j1 = inferenceutils.Joint(pos1, conf, idx=idx1)
    j2 = inferenceutils.Joint(pos2, conf, idx=idx2)
    link = inferenceutils.Link(j1, j2)
    assert link.confidence == conf ** 2
    assert link.idx == (idx1, idx2)
    assert link.to_vector() == [*pos1, *pos2]


def test_assembler(ground_truth_assemblies):
    with open(pickled_data, "rb") as file:
        data = pickle.load(file)
    with pytest.warns(UserWarning):
        ass = inferenceutils.Assembler(
            data,
            max_n_individuals=3,
            n_multibodyparts=12,
            identity_only=True,  # Test warning is properly raised
        )
    assert len(ass.metadata['imnames']) == 2330
    assert ass.n_keypoints == 12
    assert len(ass.graph) == len(ass.paf_inds) == 66
    # Assemble based on the smallest graph to speed up testing
    naive_graph = [
        [0, 1],
        [7, 8],
        [6, 7],
        [10, 11],
        [4, 5],
        [5, 6],
        [8, 9],
        [9, 10],
        [0, 3],
        [3, 4],
        [0, 2]
    ]
    paf_inds = [ass.graph.index(edge) for edge in naive_graph]
    ass.graph = naive_graph
    ass.paf_inds = paf_inds
    ass.assemble()
    assert not ass.unique
    assert len(ass.assemblies) == len(ground_truth_assemblies)
    assert (sum(1 for a in ass.assemblies.values() for _ in a)
            == sum(1 for a in ground_truth_assemblies.values() for _ in a))
