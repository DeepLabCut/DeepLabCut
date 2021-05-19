import numpy as np
import os
from conftest import TEST_DATA_DIR
from deeplabcut.generate_training_dataset import (
    read_image_shape_fast
)
from skimage import io, color


def test_read_image_shape_fast(tmp_path):
    path_rgb_image = os.path.join(TEST_DATA_DIR, "image.png")
    img = io.imread(path_rgb_image)
    shape = img.shape
    assert read_image_shape_fast(path_rgb_image) == (shape[2], shape[0], shape[1])
    path_gray_image = str(tmp_path / "gray.png")
    io.imsave(path_gray_image, color.rgb2gray(img).astype(np.uint8))
    assert read_image_shape_fast(path_gray_image) == (1, shape[0], shape[1])
