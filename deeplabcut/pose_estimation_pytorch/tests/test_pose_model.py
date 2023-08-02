import pytest
import deeplabcut.pose_estimation_pytorch.models as dlc_models


def test_backbone():
    #TODO: load various backbones and check dimension of feature map
    pass

def test_head():
    #TODO: simple test for head (is upsampling correct?)
    pass


def test_pose_model():
    #TODO: Make specific model builds and test!
    pass

## Below we build specific models and check integrity
def test_timm_hrnet():
    #TODO: build timm_hrnet and check dimension of output
    pass

def test_msa_hrnet():
    # TODO: build microsoft asia hrnet and check dimension of output
    # TODO: check if hyperparameters are loaded correctly (from the config file)
    pass

def test_msa_tokenpose():
    # TODO: build microsoft asia hrnet and check dimension of output
    # TODO: check if hyperparameters are loaded correctly (from the config file)
    #cf https://github.com/amathislab/BUCTDdev/blob/main/lib/models/transpose_h.py#L1
    pass

def test_msa_hrnetCOAM():
    # TODO: build BUCTD COAM hrnet and check dimension of output
    # TODO: check if hyperparameters are loaded correctly (from the config file)
    pass

#TODO: add other model variants our pipeline can build ;)
