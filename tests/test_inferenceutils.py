import numpy as np
import os
import pickle
import pytest
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils
from scipy.spatial.distance import squareform


TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


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


def test_calc_object_keypoint_similarity(real_assemblies):
    sigma = 0.01
    xy1 = real_assemblies[0][0].xy
    xy2 = real_assemblies[0][1].xy
    assert inferenceutils.calc_object_keypoint_similarity(xy1, xy1, sigma) == 1
    assert np.isclose(
        inferenceutils.calc_object_keypoint_similarity(xy1, xy2, sigma), 0
    )
    xy3 = xy1.copy()
    xy3[: len(xy3) // 2] = np.nan
    assert inferenceutils.calc_object_keypoint_similarity(xy3, xy1, sigma) == 0.5
    xy3[:] = np.nan
    assert inferenceutils.calc_object_keypoint_similarity(xy3, xy1, sigma) == 0
    assert np.isnan(inferenceutils.calc_object_keypoint_similarity(xy1, xy3, sigma))


def test_match_assemblies(real_assemblies):
    assemblies = real_assemblies[0]
    matched, unmatched = inferenceutils.match_assemblies(
        assemblies, assemblies[::-1], 0.01
    )
    assert not unmatched
    for ass1, ass2, oks in matched:
        assert ass1 is ass2
        assert oks == 1

    matched, unmatched = inferenceutils.match_assemblies([], assemblies, 0.01)
    assert not matched
    assert all(ass1 is ass2 for ass1, ass2 in zip(unmatched, assemblies))


def test_evaluate_assemblies(real_assemblies):
    assemblies = {i: real_assemblies[i] for i in range(3)}
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


def test_assembly():
    ass = inferenceutils.Assembly(3)
    assert len(ass) == 0

    j1 = inferenceutils.Joint((1, 1), label=0)
    j2 = inferenceutils.Joint((1, 1), label=1)
    assert ass.add_link(inferenceutils.Link(j1, j2), store_dict=True)
    assert len(ass) == 2
    assert ass.data[j2.label, 0] == 1
    assert ass.data[j2.label, -1] == -1
    assert ass.area == 0
    assert ass.intersection_with(ass) == 1.0
    assert np.all(np.isnan(ass._dict["data"]))

    ass.remove_joint(j2)
    assert len(ass) == 1
    assert np.all(np.isnan(ass.data[j2.label]))

    ass2 = inferenceutils.Assembly(2)
    ass2.add_link(inferenceutils.Link(j1, j2))
    with pytest.raises(ValueError):
        _ = ass + ass2
    ass2.remove_joint(j1)
    assert ass2 not in ass
    ass3 = ass + ass2
    assert len(ass3) == 2


def test_assembler(tmpdir_factory, real_assemblies):
    with open(os.path.join(TEST_DATA_DIR, "trimouse_full.pickle"), "rb") as file:
        data = pickle.load(file)
    with pytest.warns(UserWarning):
        ass = inferenceutils.Assembler(
            data,
            max_n_individuals=3,
            n_multibodyparts=12,
            identity_only=True,  # Test whether warning is properly raised
        )
    assert len(ass.metadata["imnames"]) == 50
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
        [0, 2],
    ]
    paf_inds = [ass.graph.index(edge) for edge in naive_graph]
    ass.graph = naive_graph
    ass.paf_inds = paf_inds
    ass.assemble()
    assert not ass.unique
    assert len(ass.assemblies) == len(real_assemblies)
    assert sum(1 for a in ass.assemblies.values() for _ in a) == sum(
        1 for a in real_assemblies.values() for _ in a
    )

    output_name = tmpdir_factory.mktemp("data").join("fake.h5")
    ass.to_h5(output_name)
    ass.to_pickle(str(output_name).replace("h5", "pickle"))


def test_assembler_with_identity(tmpdir_factory, real_assemblies):
    with open(os.path.join(TEST_DATA_DIR, "trimouse_full.pickle"), "rb") as file:
        data = pickle.load(file)

    # Generate fake identity predictions
    for k, v in data.items():
        if k != "metadata":
            conf = v["confidence"]
            ids = [np.random.rand(c.shape[0], 3) for c in conf]
            v["identity"] = ids

    ass = inferenceutils.Assembler(
        data,
        max_n_individuals=3,
        n_multibodyparts=12,
    )
    assert ass._has_identity
    assert len(ass.metadata["imnames"]) == 50
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
        [0, 2],
    ]
    paf_inds = [ass.graph.index(edge) for edge in naive_graph]
    ass.graph = naive_graph
    ass.paf_inds = paf_inds
    ass.assemble()
    assert not ass.unique
    assert len(ass.assemblies) == len(real_assemblies)
    assert sum(1 for a in ass.assemblies.values() for _ in a) == sum(
        1 for a in real_assemblies.values() for _ in a
    )
    assert all(
        np.all(_.data[:, -1] != -1)
        for a in ass.assemblies.values()
        for _ in a
    )

    # Test now with identity only and ensure assemblies
    # contain only parts of a single group ID.
    ass.identity_only = True
    ass.assemble()
    eq = []
    for a in ass.assemblies.values():
        for _ in a:
            ids = _.data[:, -1]
            ids = ids[~np.isnan(ids)]
            eq.append(np.all(ids == ids[0]))
    assert all(eq)

    output_name = tmpdir_factory.mktemp("data").join("fake.h5")
    ass.to_h5(output_name)
    ass.to_pickle(str(output_name).replace("h5", "pickle"))


def test_assembler_calibration(real_assemblies):
    with open(os.path.join(TEST_DATA_DIR, "trimouse_full.pickle"), "rb") as file:
        data = pickle.load(file)
    ass = inferenceutils.Assembler(data, max_n_individuals=3, n_multibodyparts=12)
    ass.calibrate(os.path.join(TEST_DATA_DIR, "trimouse_calib.h5"))
    assert ass._kde is not None
    assert ass.safe_edge

    assembly = real_assemblies[0][0]
    mahal, proba = ass.calc_assembly_mahalanobis_dist(assembly, return_proba=True)
    assert np.isclose(mahal, 19.541, atol=1e-3)
    assert np.isclose(proba, 1, atol=1e-3)

    j1 = inferenceutils.Joint(tuple(assembly.xy[0]), label=0)
    j2 = inferenceutils.Joint(tuple(assembly.xy[1]), label=1)
    link = inferenceutils.Link(j1, j2)
    p = ass.calc_link_probability(link)
    assert np.isclose(p, 0.990, atol=1e-3)


def test_find_outlier_assemblies(real_assemblies):
    assert len(inferenceutils.find_outlier_assemblies(real_assemblies)) == 13
