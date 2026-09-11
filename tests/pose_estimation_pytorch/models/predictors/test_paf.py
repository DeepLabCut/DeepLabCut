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
import numpy as np
import torch

from deeplabcut.core.inferenceutils import Assembly


def test_affinity_is_python_float_when_empty():
    assembly = Assembly(size=4)
    assert assembly.n_links == 0
    assert type(assembly.affinity) is float


def test_affinity_is_python_float_with_numpy_accumulation():
    assembly = Assembly(size=4)
    assembly._affinity = np.float32(1.5)
    assembly._links = [object(), object()]
    assert type(assembly.affinity) is float


def test_affinity_assigns_into_float_tensor():
    """Regression: numpy.float32 into a torch tensor raises under NumPy 2."""
    assembly = Assembly(size=4)
    assembly._affinity = np.float32(0.75)
    assembly._links = [object()]
    poses = -torch.ones((1, 1, 4, 5))
    poses[0, 0, :, 4] = float(assembly.affinity)
    assert torch.allclose(poses[0, 0, :, 4], torch.full((4,), 0.75))
