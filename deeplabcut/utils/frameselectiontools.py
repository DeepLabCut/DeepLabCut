"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import math

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def UniformFrames(clip, numframes2pick, start, stop, Index=None):
    """ Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    """
    print(
        "Uniformly extracting of frames from",
        round(start * clip.duration, 2),
        " seconds to",
        round(stop * clip.duration, 2),
        " seconds.",
    )
    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(
                math.ceil(clip.duration * clip.fps * stop),
                size=numframes2pick,
                replace=False,
            )
        else:
            frames2pick = np.random.choice(
                range(
                    math.floor(start * clip.duration * clip.fps),
                    math.ceil(clip.duration * clip.fps * stop),
                ),
                size=numframes2pick,
                replace=False,
            )
        return frames2pick
    else:
        startindex = int(np.floor(clip.fps * clip.duration * start))
        stopindex = int(np.ceil(clip.fps * clip.duration * stop))
        Index = np.array(Index, dtype=np.int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


# uses openCV
def UniformFramescv2(cap, numframes2pick, start, stop, Index=None):
    """ Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    """
    nframes = len(cap)
    print(
        "Uniformly extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )

    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(
                math.ceil(nframes * stop), size=numframes2pick, replace=False
            )
        else:
            frames2pick = np.random.choice(
                range(math.floor(nframes * start), math.ceil(nframes * stop)),
                size=numframes2pick,
                replace=False,
            )
        return frames2pick
    else:
        startindex = int(np.floor(nframes * start))
        stopindex = int(np.ceil(nframes * stop))
        Index = np.array(Index, dtype=np.int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


def KmeansbasedFrameselection(
    clip,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """ This code downsamples the video to a width of resizewidth.

    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick."""

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * clip.duration, 2),
        " seconds to",
        round(stop * clip.duration, 2),
        " seconds.",
    )
    startindex = int(np.floor(clip.fps * clip.duration * start))
    stopindex = int(np.ceil(clip.fps * clip.duration * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = int(nframes / 2)

    if len(Index) >= numframes2pick:
        clipresized = clip.resize(width=resizewidth)
        ny, nx = clipresized.size
        frame0 = img_as_ubyte(clip.get_frame(0))
        if np.ndim(frame0) == 3:
            ncolors = np.shape(frame0)[2]
        else:
            ncolors = 1
        print("Extracting and downsampling...", nframes, " frames from the video.")

        if color and ncolors > 1:
            DATA = np.zeros((nframes, nx * 3, ny))
            for counter, index in tqdm(enumerate(Index)):
                image = img_as_ubyte(
                    clipresized.get_frame(index * 1.0 / clipresized.fps)
                )
                DATA[counter, :, :] = np.vstack(
                    [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                )
        else:
            DATA = np.zeros((nframes, nx, ny))
            for counter, index in tqdm(enumerate(Index)):
                if ncolors == 1:
                    DATA[counter, :, :] = img_as_ubyte(
                        clipresized.get_frame(index * 1.0 / clipresized.fps)
                    )
                else:  # attention: averages over color channels to keep size small / perhaps you want to use color information?
                    DATA[counter, :, :] = img_as_ubyte(
                        np.array(
                            np.mean(
                                clipresized.get_frame(index * 1.0 / clipresized.fps), 2
                            ),
                            dtype=np.uint8,
                        )
                    )

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )

        clipresized.close()
        del clipresized
        return list(np.array(frames2pick))
    else:
        return list(Index)


def KmeansbasedFrameselectioncv2(
    cap,
    numframes2pick,
    start,
    stop,
    crop,
    coords,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """ This code downsamples the video to a width of resizewidth.
    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick.

    Attention: the flow of commands was not optimized for readability, but rather speed. This is why it might appear tedious and repetitive."""
    nframes = len(cap)
    nx, ny = cap.dimensions
    ratio = resizewidth * 1.0 / nx
    if ratio > 1:
        raise Exception("Choice of resizewidth actually upsamples!")

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = nframes // 2

    allocated = False
    if len(Index) >= numframes2pick:
        if (
            np.mean(np.diff(Index)) > 1
        ):  # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]

                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                            )
                            allocated = True
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]
                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1])
                            )
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)
        else:
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]

                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1] * 3)
                            )
                            allocated = True
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame()
                    if frame is not None:
                        if crop:
                            frame = frame[
                                int(coords[2]) : int(coords[3]),
                                int(coords[0]) : int(coords[1]),
                                :,
                            ]
                        # image=img_as_ubyte(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),None,fx=ratio,fy=ratio))
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        if (
                            not allocated
                        ):  #'DATA' not in locals(): #allocate memory in first pass
                            DATA = np.empty(
                                (nframes, np.shape(image)[0], np.shape(image)[1])
                            )
                            allocated = True
                        DATA[counter, :, :] = np.mean(image, 2)

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )
        # cap.release() >> still used in frame_extraction!
        return list(np.array(frames2pick))
    else:
        return list(Index)
