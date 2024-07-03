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
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions


# This dictionary maps the model types to the file locations where the models exist.
MODEL_BASE_PATH = Path("pose_estimation_tensorflow") / "models" / "pretrained"
MODELTYPE_FILEPATH_MAP = {
    "resnet_50": MODEL_BASE_PATH / "resnet_v1_50.ckpt",
    "resnet_101": MODEL_BASE_PATH / "resnet_v1_101.ckpt",
    "resnet_152": MODEL_BASE_PATH / "resnet_v1_152.ckpt",
    "mobilenet_v2_1.0": MODEL_BASE_PATH / "mobilenet_v2_1.0_224.ckpt",
    "mobilenet_v2_0.75": MODEL_BASE_PATH / "mobilenet_v2_0.75_224.ckpt",
    "mobilenet_v2_0.5": MODEL_BASE_PATH / "mobilenet_v2_0.5_224.ckpt",
    "mobilenet_v2_0.35": MODEL_BASE_PATH / "mobilenet_v2_0.35_224.ckpt",
    "efficientnet-b0": MODEL_BASE_PATH / "efficientnet-b0" / "model.ckpt",
    "efficientnet-b1": MODEL_BASE_PATH / "efficientnet-b1" / "model.ckpt",
    "efficientnet-b2": MODEL_BASE_PATH / "efficientnet-b2" / "model.ckpt",
    "efficientnet-b3": MODEL_BASE_PATH / "efficientnet-b3" / "model.ckpt",
    "efficientnet-b4": MODEL_BASE_PATH / "efficientnet-b4" / "model.ckpt",
    "efficientnet-b5": MODEL_BASE_PATH / "efficientnet-b5" / "model.ckpt",
    "efficientnet-b6": MODEL_BASE_PATH / "efficientnet-b6" / "model.ckpt",
}


def check_for_weights(modeltype, parent_path):
    """gets local path to network weights and checks if they are present. If not, downloads them from tensorflow.org"""
    if modeltype not in MODELTYPE_FILEPATH_MAP.keys():
        print(
            "Currently ResNet (50, 101, 152), MobilenetV2 (1, 0.75, 0.5 and 0.35) and EfficientNet (b0-b6) are supported, please change 'resnet' entry in config.yaml!"
        )
        # Exit the function early if an unknown modeltype is provided.
        return parent_path

    exists = False
    model_path = parent_path / MODELTYPE_FILEPATH_MAP[modeltype]
    try:
        for file in os.listdir(model_path.parent):
            if model_path.name in file:
                exists = True
                break
    except FileNotFoundError:
        pass

    if not exists:
        if "efficientnet" in modeltype:
            download_weights(modeltype, model_path.parent)
        else:
            download_weights(modeltype, model_path)

    return str(model_path)


def download_weights(modeltype, model_path):
    """
    Downloads the ImageNet pretrained weights for ResNets, MobileNets et al. from TensorFlow...
    """
    import urllib
    import tarfile
    from io import BytesIO

    target_dir = model_path.parents[0]
    neturls = auxiliaryfunctions.read_plainconfig(
        target_dir / "pretrained_model_urls.yaml"
    )
    try:
        if "efficientnet" in modeltype:
            url = neturls["efficientnet"]
            url = url + modeltype.replace("_", "-") + ".tar.gz"
        else:
            url = neturls[modeltype]
        print("Downloading a ImageNet-pretrained model from {}....".format(url))
        response = urllib.request.urlopen(url)
        with tarfile.open(fileobj=BytesIO(response.read()), mode="r:gz") as tar:
            tar.extractall(path=target_dir)
    except KeyError:
        print("Model does not exist: ", modeltype)
        print("Pick one of the following: ", neturls.keys())


def download_model(modelname, target_dir):
    """
    Downloads a DeepLabCut Model Zoo Project
    """
    import urllib.request
    import tarfile
    from tqdm import tqdm

    def show_progress(count, block_size, total_size):
        pbar.update(block_size)

    def tarfilenamecutting(tarf):
        """' auxfun to extract folder path
        ie. /xyz-trainsetxyshufflez/
        """
        for memberid, member in enumerate(tarf.getmembers()):
            if memberid == 0:
                parent = str(member.path)
                l = len(parent) + 1
            if member.path.startswith(parent):
                member.path = member.path[l:]
                yield member

    dlc_path = auxiliaryfunctions.get_deeplabcut_path()
    neturls = auxiliaryfunctions.read_plainconfig(
        os.path.join(
            dlc_path,
            "pose_estimation_tensorflow",
            "models",
            "pretrained",
            "pretrained_model_urls.yaml",
        )
    )
    if modelname in neturls.keys():
        url = neturls[modelname]
        response = urllib.request.urlopen(url)
        print(
            "Downloading the model from the DeepLabCut server @Harvard -> Go Crimson!!! {}....".format(
                url
            )
        )
        total_size = int(response.getheader("Content-Length"))
        pbar = tqdm(unit="B", total=total_size, position=0)
        filename, _ = urllib.request.urlretrieve(url, reporthook=show_progress)
        with tarfile.open(filename, mode="r:gz") as tar:
            tar.extractall(target_dir, members=tarfilenamecutting(tar))
    else:
        models = [
            fn
            for fn in neturls.keys()
            if "resnet_" not in fn
            and "efficientnet" not in fn
            and "mobilenet_" not in fn
        ]
        print("Model does not exist: ", modelname)
        print("Pick one of the following: ", models)


def set_visible_devices(gputouse: int):
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices("GPU")
    n_devices = len(physical_devices)
    if gputouse >= n_devices:
        raise ValueError(
            f"There are {n_devices} available GPUs: {physical_devices}\nPlease choose `gputouse` in {list(range(n_devices))}."
        )
    tf.config.set_visible_devices(physical_devices[gputouse], "GPU")


def smart_restore(restorer, sess, checkpoint_path, net_type):
    "Restore pretrained weights, smartly redownloading them if missing."
    try:
        restorer.restore(sess, checkpoint_path)
    except ValueError as e:  # The path may be wrong, or the weights no longer exist
        dlcparent_path = auxiliaryfunctions.get_deeplabcut_path()
        correct_model_path = os.path.join(
            dlcparent_path, MODELTYPE_FILEPATH_MAP[net_type],
        )
        if checkpoint_path == correct_model_path:
            # The path is right, hence the weights are missing; we'll download them again.
            _ = check_for_weights(net_type, Path(dlcparent_path))
            restorer.restore(sess, checkpoint_path)
        else:
            raise ValueError(e)


# Aliases for backwards-compatibility
Check4Weights = check_for_weights
Downloadweights = download_weights
DownloadModel = download_model
