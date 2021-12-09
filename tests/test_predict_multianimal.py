import numpy as np
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.core import predict_multianimal


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
    inds = predict_multianimal.find_local_peak_indices_maxpool_nms(
        scmaps, RADIUS, THRESHOLD,
    )
    with tf.compat.v1.Session() as sess:
        inds = sess.run(inds)
    pos = predict_multianimal.calc_peak_locations(locrefs, inds, STRIDE)
    s, r, c, b = inds.T
    prob = scmaps[s, r, c, b].reshape((-1, 1))
    idx = np.argsort(inds[:, -1], kind="mergesort")
    np.testing.assert_equal(inds[idx, 1:], inds_gt)
    np.testing.assert_almost_equal(pos[idx], pos_gt, decimal=3)
    np.testing.assert_almost_equal(prob[idx], prob_gt, decimal=5)


def test_association_costs(model_outputs, ground_truth_detections):
    costs_gt = ground_truth_detections[0]["costs"]
    peak_inds = predict_multianimal.find_local_peak_indices_maxpool_nms(
        model_outputs[0], RADIUS, THRESHOLD,
    )
    with tf.compat.v1.Session() as sess:
        peak_inds = sess.run(peak_inds)
    graph = [[i, j] for i in range(12) for j in range(i + 1, 12)]
    preds = predict_multianimal.compute_peaks_and_costs(
        *model_outputs,
        peak_inds,
        graph=graph,
        paf_inds=np.arange(len(graph)),
        n_id_channels=0,
        stride=STRIDE,
    )[0]
    assert all(k in preds for k in ("coordinates", "confidence", "costs"))
    costs_pred = preds["costs"]
    assert len(costs_pred) == len(costs_gt)
    eq = [
        np.array_equal(np.argmax(v["m1"], axis=0), np.argmax(costs_gt[k]["m1"], axis=0))
        for k, v in costs_pred.items()
    ]
    assert sum(eq) == 60  # 6 arrays are unequal as cost computation was corrected
    assert all(
        np.allclose(v["distance"], costs_gt[k]["distance"], atol=1.5)
        for k, v in costs_pred.items()
    )


def test_compute_peaks_and_costs_no_graph(model_outputs):
    peak_inds = predict_multianimal.find_local_peak_indices_maxpool_nms(
        model_outputs[0], RADIUS, THRESHOLD,
    )
    with tf.compat.v1.Session() as sess:
        peak_inds = sess.run(peak_inds)
    preds = predict_multianimal.compute_peaks_and_costs(
        *model_outputs,
        peak_inds,
        graph=[],
        paf_inds=[],
        n_id_channels=0,
        stride=STRIDE,
    )[0]
    assert "costs" not in preds
