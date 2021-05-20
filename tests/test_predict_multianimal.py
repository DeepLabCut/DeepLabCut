import numpy as np
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.nnet import predict_multianimal


RADIUS = 5
THRESHOLD = 0.01
STRIDE = 8


def test_extract_detections(model_outputs, ground_truth_detections):
    scmaps, locrefs, _ = model_outputs
    inds_gt = []
    for i in range(scmaps.shape[3]):
        scmap = scmaps[0, ..., i]
        peaks = predict_multianimal.find_local_maxima(scmap, RADIUS, THRESHOLD)
        inds_gt.append(np.c_[peaks, np.ones(len(peaks)).reshape((-1, 1)) * i])
    inds_gt = np.concatenate(inds_gt).astype(int)
    pos_gt = np.concatenate(ground_truth_detections[0]["coordinates"][0])
    prob_gt = np.concatenate(ground_truth_detections[0]["confidence"])

    inds = predict_multianimal.find_local_peak_indices(scmaps, RADIUS, THRESHOLD)
    pos = predict_multianimal.calc_peak_locations(locrefs, inds, STRIDE)
    prob = tf.gather_nd(scmaps, inds)
    with tf.Session() as sess:
        inds = sess.run(inds)
        pos = sess.run(pos)
        prob = sess.run(prob)
    idx = np.argsort(inds[:, -1], kind="mergesort")
    np.testing.assert_equal(inds[idx, 1:], inds_gt)
    np.testing.assert_almost_equal(pos[idx], pos_gt, decimal=3)
    np.testing.assert_almost_equal(prob[idx], prob_gt.squeeze(), decimal=5)


def test_association_costs(model_outputs, ground_truth_detections):
    costs_gt = ground_truth_detections[0]["costs"]
    aff_ref = costs_gt[0]["m1"]
    lengths_ref = costs_gt[0]["distance"]
    scmaps, _, pafs = model_outputs
    graph = [[i, j] for i in range(12) for j in range(i + 1, 12)]
    inds_all = predict_multianimal.find_local_peak_indices(scmaps, RADIUS, THRESHOLD)
    aff, lengths, *_ = predict_multianimal.compute_edge_costs(pafs, inds_all, graph)
    aff = tf.reshape(aff[:aff_ref.size], aff_ref.shape)
    lengths = tf.reshape(lengths[:lengths_ref.size], lengths_ref.shape)
    with tf.compat.v1.Session() as sess:
        aff = sess.run(aff)
        lengths = sess.run(lengths)
    np.testing.assert_equal(np.argmax(aff, axis=0), np.argmax(aff_ref, axis=0))
    np.testing.assert_allclose(lengths, lengths_ref, atol=1.5)
