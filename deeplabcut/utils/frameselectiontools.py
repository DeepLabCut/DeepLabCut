"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import numpy as np
import math
from skimage import io
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
import cv2
from tqdm import tqdm

def run(picker, numframes2pick, start=0, stop=1, algo='kmeans', subindices=None, **kwargs):
    if algo in selection_algorithms.keys():
        return selection_algorithms[algo](picker, numframes2pick, start=start, stop=stop,
                                        subindices=subindices, **kwargs)
    else:
        raise RuntimeError("algorithm not found: {}; please implement the frame selection method yourself and send us a pull request! Otherwise, choose 'uniform' or 'kmeans'.".format(algo))

def __as_indices(picker, start, stop):
    startindex = int(math.floor(picker.nframes*start))
    stopindex  = int(math.ceil(picker.nframes*stop))
    return startindex, stopindex

def uniform_frame_selection(picker, numframes2pick, start=0, stop=1, subindices=None, **kwargs_ignored):
    ''' Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable subindices allows to pass on a subindex for the frames.
    '''
    print("Uniformly extracting of frames from",
            round(start*picker.duration,2),
            " seconds to",
            round(stop*picker.duration,2),
            " seconds.")
    startindex, stopindex = __as_indices(picker, start, stop)

    if subindices is None:
        if startindex==0:
            frames2pick = np.random.choice(stopindex, size=numframes2pick, replace = False)
        else:
            frames2pick = np.random.choice(np.arange(startindex, stopindex), size=numframes2pick, replace = False)
        return frames2pick
    else:
        subindices = np.array(subindices,dtype=np.int)
        subindices = subindices[(subindices>=startindex)*(subindices<stopindex)] #crop to range!
        if subindices.size >= numframes2pick:
            return np.random.permutation(subindices)[:numframes2pick]
        else:
            return subindices

def kmeans_based_frame_selection(picker, numframes2pick, start=0, stop=1,
                                subindices=None, step=1, resizewidth=30,
                                batchsize=100, max_iter=50):
    ''' This code downsamples the video to a width of resizewidth.

    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick.'''

    print("Kmeans-quantization based extracting of frames from", round(start*pick.duration,2)," seconds to", round(stop*pick.duration,2), " seconds.")

    # prepare list of frame indices
    startindex, stopindex = __as_indices(picker, start, stop)
    if subindices is None:
        subindices = np.arange(startindex, stopindex, step)
    else:
        subindices = np.array(subindices)
        subindices = subindices[(subindices>=startindex)*(subindices<stopindex)] #crop to range!

    nframes = subindices.size
    if nframes < batchsize:
        batchsize = int(nframes / 2)

    if nframes < numframes2pick - 1:
        return subindices

    else: # need to perform clustering
        print("Extracting and downsampling", nframes, " frames from the video.")
        picker.set_resize(resizewidth)
        DATA = None
        for counter, index in tqdm(enumerate(subindices)):
            image = picker.pick_single(index, crop=True, resize=True, transform_color=False) #color trafo not necessary; lack thereof improves speed.
            if picker.is_colored:
                image = np.concatenate([image[0], image[1], image[2]], axis=1)
            elif picker.ncolors == 1:
                image = image.mean(2)
            if DATA is None:
                DATA = np.empty((nframes,)+image.shape, dtype=float)
            DATA[counter,:,:] = image

        print("Kmeans clustering ... (this might take a while)")
        data = (DATA - DATA.mean(0)).reshape((nframes,-1)) # (nframes x H x W) -> (nframes x (H x W))
        kmeans=MiniBatchKMeans(n_clusters=numframes2pick,
                                tol=1e-3,
                                batch_size=batchsize,
                                max_iter=max_iter).fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):
            # pick one frame per cluster
            members = subindices[kmeans.labels_ == clusterid]
            if members.size > 0:
                frames2pick.append(np.random.choice(members, size=1))
        return frames2pick

selection_algorithms = {
    'kmeans': kmeans_based_frame_selection,
    'uniform': uniform_frame_selection
}
