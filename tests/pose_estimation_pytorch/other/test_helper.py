#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import torch


def test_train_valid_call():
    tmp_model = torch.nn.Linear(3, 10)
    to_train_mode = tmp_model.train
    to_train_mode()
    assert tmp_model.training
    to_valid_mode = tmp_model.eval
    to_valid_mode()
    assert not tmp_model.training
