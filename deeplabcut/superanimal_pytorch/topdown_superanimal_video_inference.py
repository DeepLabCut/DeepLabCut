import sys
import os
import subprocess
import deeplabcut


def topdown_video_inference(
    video_path,
    det_config,
    det_checkpoint,
    pose_config,
    pose_checkpoint,
    outdir="",
    device="cpu",
    kpt_threshold=0.6,
    video_adapt=True,
    videotype=".mp4",
    apply_median_filter=True,
):

    video_name = video_path.split(os.sep)[-1].split(videotype)[0]
    dlc_path = os.path.dirname(deeplabcut.__file__)
    root = os.path.join(dlc_path, "superanimal_pytorch")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(root, "third_party"), env["PYTHONPATH"]]
    )

    def run_command(command):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        stdout, stderr = process.communicate()
        print(stdout.decode("utf-8"))
        print(stderr.decode("utf-8"))

        if process.returncode != 0:
            # An erroor happened!
            print(f"An error occurred: {stderr.decode()}")
        else:
            print(f"Script output: {stdout.decode()}")

    inference_demo = os.path.join(
        root, "third_party", "demo", "top_down_video_demo_with_mmdet.py"
    )

    video_outdir = os.path.join(outdir, f"video_pred_{video_name}")

    dataset_info_path = os.path.join(root, "third_party", "quadruped_dataset.json")
    inference_only_command = [
        "python",
        inference_demo,
        det_config,
        det_checkpoint,
        pose_config,
        pose_checkpoint,
        "--video-path",
        video_path,
        "--out-video-root",
        video_outdir,
        "--kpt-thr",
        f"{kpt_threshold}",
        "--dataset-info",
        f"{dataset_info_path}",
    ]

    if video_adapt == False:
        if apply_median_filter:
            inference_only_command.append("--kpt-median-filter")
        run_command(inference_only_command)
    else:

        result_file = os.path.join(video_outdir, f"{video_name}.mp4.json")
        convert_code = os.path.join(root, "third_party", "videopseudo2annotation.py")
        pseudo_label_command = [
            "python",
            convert_code,
            "--video_result_path",
            result_file,
            "--video_path",
            video_path,
        ]

        pose_train_code = os.path.join(root, "third_party", "tools", "train.py")
        train_ann_file = os.path.join(
            video_outdir, f"annotation_{video_name}", "annotations", "train.json"
        )
        val_ann_file = os.path.join(
            video_outdir, f"annotation_{video_name}", "annotations", "test.json"
        )
        img_prefix = os.path.join(
            video_outdir, f"annotation_{video_name}", "images", ""
        )
        adaptation_weight_folder = os.path.join(video_outdir, f"{video_name}_adapted")

        adaptation_command = [
            "python",
            pose_train_code,
            pose_config,
            "--cfg-options",
            f"dataset_info={dataset_info_path}",
            f"data.train.ann_file={train_ann_file}",
            f"data.train.img_prefix={img_prefix}",
            f"data.train.dataset_info={dataset_info_path}",
            f"data.val.dataset_info={dataset_info_path}",
            f"data.val.img_prefix={img_prefix}",
            f"data.val.ann_file={val_ann_file}",
            f"dataset_info={dataset_info_path}",
            "total_epochs=4",
            "lr_config.warmup_iters=1",
            "optimizer.lr=5e-5",
            f"load_from={pose_checkpoint}",
            "--work-dir",
            adaptation_weight_folder,
        ]
        adapted_pose_checkpoint = os.path.join(adaptation_weight_folder, "latest.pth")

        inference_adapted_model_command = [
            "python",
            inference_demo,
            det_config,
            det_checkpoint,
            pose_config,
            adapted_pose_checkpoint,
            "--video-path",
            video_path,
            "--out-video-root",
            f"video_pred_{video_name}",
            "--kpt-thr",
            f"{kpt_threshold}",
            "--kpt-median-filter",
            "--dataset-info",
            f"{dataset_info_path}",
        ]

        run_command(inference_only_command)
        run_command(pseudo_label_command)
        run_command(adaptation_command)
        run_command(inference_adapted_model_command)
