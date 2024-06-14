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
import os
import pickle
import pytest
from conftest import TEST_DATA_DIR
from copy import deepcopy
from deeplabcut.core import inferenceutils
from scipy.spatial.distance import squareform


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

    # Test flipped keypoints
    xy4 = xy1.copy()
    symmetric_pair = [0, 11]
    xy4[symmetric_pair] = xy4[symmetric_pair[::-1]]
    assert inferenceutils.calc_object_keypoint_similarity(xy1, xy4, sigma) != 1
    assert (
        inferenceutils.calc_object_keypoint_similarity(
            xy1, xy4, sigma, symmetric_kpts=[symmetric_pair]
        )
        == 1
    )


def test_match_assemblies(real_assemblies):
    assemblies = real_assemblies[0]
    num_gt, matches = inferenceutils.match_assemblies(
        assemblies, assemblies[::-1], 0.01
    )
    assert len(assemblies) == len(matches)
    for m in matches:
        assert m.prediction is m.ground_truth
        assert m.oks == 1

    num_gt, matches = inferenceutils.match_assemblies([], assemblies, 0.01)
    assert len(matches) == 0
    assert num_gt == len(assemblies)


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

    dict_ = inferenceutils.evaluate_assembly(
        assemblies,
        assemblies,
        oks_thresholds=thresholds,
        symmetric_kpts=[(0, 5), (1, 4)],
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
    # Original (cached) coordinates must have remained empty
    assert np.all(np.isnan(ass._dict["data"][:, :2]))

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
    ass.paf_inds = [ass.graph.index(edge) for edge in naive_graph]
    ass.assemble()
    assert not ass.unique
    assert len(ass.assemblies) == len(real_assemblies)
    assert sum(1 for a in ass.assemblies.values() for _ in a) == sum(
        1 for a in real_assemblies.values() for _ in a
    )

    output_name = tmpdir_factory.mktemp("data").join("fake.h5")
    ass.to_h5(output_name)
    ass.to_pickle(str(output_name).replace("h5", "pickle"))


def test_assembler_with_single_bodypart(real_assemblies):
    with open(os.path.join(TEST_DATA_DIR, "trimouse_full.pickle"), "rb") as file:
        temp = pickle.load(file)
    data = {"metadata": temp.pop("metadata")}
    for k, dict_ in temp.items():
        data[k] = {
            "coordinates": (dict_["coordinates"][0][:1],),
            "confidence": dict_["confidence"][:1],
        }
    ass = inferenceutils.Assembler(
        data,
        max_n_individuals=3,
        n_multibodyparts=1,
    )
    ass.metadata["joint_names"] = ass.metadata["joint_names"][:1]
    ass.metadata["num_joints"] = 1
    ass.metadata["paf_graph"] = []
    ass.metadata["paf"] = []
    ass.metadata["bpts"] = [0]
    ass.metadata["ibpts"] = [0]
    ass.assemble(chunk_size=0)
    assert not ass.unique
    assert len(ass.assemblies) == len(real_assemblies)
    assert all(len(a) == 3 for a in ass.assemblies.values())


def test_assembler_with_unique_bodypart(real_assemblies_montblanc):
    with open(os.path.join(TEST_DATA_DIR, "montblanc_full.pickle"), "rb") as file:
        data = pickle.load(file)
    ass = inferenceutils.Assembler(
        data,
        max_n_individuals=3,
        n_multibodyparts=4,
        pcutoff=0.1,
        min_affinity=0.1,
    )
    assert len(ass.metadata["imnames"]) == 180
    assert ass.n_keypoints == 5
    assert len(ass.graph) == len(ass.paf_inds) == 6
    ass.assemble(chunk_size=0)
    assert len(ass.assemblies) == len(real_assemblies_montblanc[0])
    assert len(ass.unique) == len(real_assemblies_montblanc[1])
    assemblies = np.concatenate(
        [ass.xy for assemblies in ass.assemblies.values() for ass in assemblies]
    )
    assemblies_gt = np.concatenate(
        [
            ass.xy
            for assemblies in real_assemblies_montblanc[0].values()
            for ass in assemblies
        ]
    )
    np.testing.assert_equal(assemblies, assemblies_gt)


def test_assembler_with_identity(tmpdir_factory, real_assemblies):
    with open(os.path.join(TEST_DATA_DIR, "trimouse_full.pickle"), "rb") as file:
        data = pickle.load(file)

    # Generate fake identity predictions
    for k, v in data.items():
        if k != "metadata":
            conf = v["confidence"]
            ids = [np.random.rand(c.shape[0], 3) for c in conf]
            v["identity"] = ids

    ass = inferenceutils.Assembler(data, max_n_individuals=3, n_multibodyparts=12)
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
    ass.paf_inds = [ass.graph.index(edge) for edge in naive_graph]
    ass.assemble()
    assert not ass.unique
    assert len(ass.assemblies) == len(real_assemblies)
    assert sum(1 for a in ass.assemblies.values() for _ in a) == sum(
        1 for a in real_assemblies.values() for _ in a
    )
    assert all(np.all(_.data[:, -1] != -1) for a in ass.assemblies.values() for _ in a)

    # Test now with identity only and ensure assemblies
    # contain only parts of a single group ID.
    ass.identity_only = True
    ass.assemble()
    assert len(ass.assemblies) == len(real_assemblies)
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
    assert np.isclose(p, 0.993, atol=1e-3)

    # Test empty assembly
    assembly_ = deepcopy(assembly)
    assembly_.data[:, :2] = np.nan
    mahal, proba = ass.calc_assembly_mahalanobis_dist(assembly_, return_proba=True)
    assert np.isinf(mahal)
    assert proba == 0


def test_find_outlier_assemblies(real_assemblies):
    assert len(inferenceutils.find_outlier_assemblies(real_assemblies)) == 13
