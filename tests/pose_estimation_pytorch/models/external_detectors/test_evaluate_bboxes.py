from __future__ import annotations


def test_evaluate_snapshot_uses_precomputed_detector_runner_per_split(
    eval_mod,
    fake_eval_topdown_loader,
    fake_snapshot,
    patch_evaluate_snapshot_dependencies,
):
    state = patch_evaluate_snapshot_dependencies

    eval_mod.evaluate_snapshot(
        cfg={"pcutoff": 0.6},
        loader=fake_eval_topdown_loader,
        snapshot=fake_snapshot,
        scorer="fake_scorer",
        show_errors=False,
    )

    assert state["built_modes"] == ["train", "test"]

    evaluate_calls = state["evaluate_calls"]
    assert len(evaluate_calls) == 2

    assert evaluate_calls[0]["mode"] == "train"
    assert evaluate_calls[0]["detector_runner"].mode == "train"

    assert evaluate_calls[1]["mode"] == "test"
    assert evaluate_calls[1]["detector_runner"].mode == "test"


def test_evaluate_network_reports_precomputed_bboxes_instead_of_gt_fallback(
    eval_mod,
    patch_evaluate_network_dependencies,
    tmp_path,
    capsys,
):
    eval_mod.evaluate_network(
        config=tmp_path / "config.yaml",
        shuffles=[4],
        trainingsetindex=0,
    )

    captured = capsys.readouterr()

    assert "Using precomputed detector bounding boxes" in captured.out
    assert "Using ground truth bounding boxes" not in captured.out
    assert "you'll need to train a detector" not in captured.out
