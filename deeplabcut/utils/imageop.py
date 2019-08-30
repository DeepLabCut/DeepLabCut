
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0

Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
"""

from warnings import warn

try:
    from scipy.misc import imread, imresize
except ImportError:
    import imageio
    from PIL import Image
    import numpy as np

    def imread(path, mode=None):
        return imageio.imread(path, pilmode=mode)

    def imresize(img, size=None, interp='nearest', mode=None):
        if interp == 'nearest':
            interp = Image.NEAREST
        elif interp == 'lanczos':
            interp = Image.LANCZOS
        elif interp == 'bilinear':
            interp = Image.BILINEAR
        elif interp in ('bicubic', 'cubic'):
            interp = Image.BICUBIC
        else:
            warn("unknown mode for imresize ({}). 'nearest' is used in place.".format(interp),
                RuntimeWarning, stacklevel=2)
            interp = Image.NEAREST

        width, height = img.shape[:2]
        if isinstance(size, int):
            size = (round(width*size/100), round(height*size/100))
        elif isinstance(size, float):
            size = (round(width*size), round(height*size))
        else:
            pass # just use `size` as-is
        return np.array(Image.fromarray(img, mode=mode).resize(size, resample=interp))
