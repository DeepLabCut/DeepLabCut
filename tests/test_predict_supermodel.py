import numpy as np
from deeplabcut.pose_estimation_tensorflow import predict_supermodel


def test_get_multi_scale_frames():
    fake_img = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    ar = fake_img.shape[1] / fake_img.shape[0]
    heights = list(range(100, 1000, 100))
    frames, shapes = predict_supermodel.get_multi_scale_frames(
        fake_img, heights,
    )
    assert len(frames) == len(shapes) == len(heights)
    assert all(shape[0] == h for shape, h in zip(shapes, heights))
    assert all(round(shape[0] * ar) == shape[1] for shape in shapes)
