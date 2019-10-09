"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import os
from deeplabcut.utils import auxiliaryfunctions
from pathlib import Path

def Check4weights(modeltype,parent_path,num_shuffles):
    ''' gets local path to network weights and checks if they are present. If not, downloads them from tensorflow.org '''
    if 'resnet_50' == modeltype:
        model_path = parent_path  / 'pose_estimation_tensorflow/models/pretrained/resnet_v1_50.ckpt'
    elif 'resnet_101' == modeltype:
        model_path = parent_path / 'pose_estimation_tensorflow/models/pretrained/resnet_v1_101.ckpt'
    elif 'resnet_152' == modeltype:
        model_path = parent_path / 'pose_estimation_tensorflow/models/pretrained/resnet_v1_152.ckpt'
    elif 'mobilenet' in modeltype:
        model_path = Path(os.path.join(parent_path , 'pose_estimation_tensorflow/models/pretrained/'+str(modeltype)+'_224.ckpt'))
    else:
        print("Currently ResNet (50, 101, 152) and MobilenetV2 (1, 0.75, 0.5 and 0.35) are supported, please change 'resnet' entry in config.yaml!")
        num_shuffles=-1 #thus the loop below is empty...
        model_path=parent_path

    if num_shuffles>0:
        if not model_path.is_file():
            Downloadweights(modeltype,model_path)

    return str(model_path),num_shuffles

def Downloadweights(modeltype,model_path):
    """
    Downloads the ImageNet pretrained weights for ResNets, MobileNets et al. from TensorFlow...
    """
    import urllib
    import tarfile
    from io import BytesIO

    target_dir = model_path.parents[0]
    neturls=auxiliaryfunctions.read_plainconfig(target_dir / 'pretrained_model_urls.yaml')
    try:
        url = neturls[modeltype]
        print("Downloading a ImageNet-pretrained model from {}....".format(url))
        response = urllib.request.urlopen(url)
        with tarfile.open(fileobj=BytesIO(response.read()), mode='r:gz') as tar:
            tar.extractall(path=target_dir)
    except KeyError:
        print("Model does not exist: ", modeltype)
        print("Pick one of the following: ", neturls.keys())

def download_mpii_weigths(wd):
    ''' Downloads weights pretrained on human data from DeeperCut. '''
    import urllib.request
    from pathlib import Path

    url = ['https://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/mpii-single-resnet-101.data-00000-of-00001','https://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/mpii-single-resnet-101.meta','https://datasets.d2.mpi-inf.mpg.de/deepercut-models-tensorflow/mpii-single-resnet-101.index']
    for i in url:
        file = str(Path(i).name)
        filename = file.replace("mpii-single-resnet-101","snapshot-103000")
        filename = os.path.join(wd,filename)
        if os.path.isfile(filename):
            print("Weights already present!")
            break # not checking all the 3 files.
        else:
            urllib.request.urlretrieve(i, filename)
    return filename
