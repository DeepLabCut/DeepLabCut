"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

import numpy as np
import math
from skimage import io
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def UniformFrames(clip,numframes2pick,start,stop,Index="all"):
    ''' Temporally uniformly sampling frames in interval (start,stop). Visual information of video is irrelevant for this method. This code is 
    very fast and sufficient (to extract distinct frames) when behavioral videos naturally covers many states. '''
    print("Uniformly extracting of frames from", start*clip.duration," seconds to", stop*clip.duration, " seconds.")
    if Index=="all":
        if start==0:
            frames2pick = np.random.randint(math.ceil(clip.duration * clip.fps * stop), size=numframes2pick)
        else:
            frames2pick = np.random.randint(low=math.floor(start*clip.duration * clip.fps),high=math.ceil(clip.duration * clip.fps * stop), size=numframes2pick)
        return frames2pick
    else:
        startindex=int(np.floor(clip.fps*clip.duration*start))
        stopindex=int(np.ceil(clip.fps*clip.duration*stop))
        Index=np.array(Index)
        Index=Index[(Index>startindex)*(Index<stopindex)] #crop to range!
        if len(Index)>numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)
    
def KmeansbasedFrameselection(clip,numframes2pick,start,stop,Index="all",resizewidth=30,batchsize=100,max_iter=50):
    ''' This code downsamples the video to a width of resizewidth. The video is extracted as a numpy array, which is then 
    clustered with kmeans whereby each frames is treated as a vector. Frames from different clusters are then selected for labeling. This 
    procedure makes sure that the frames "look different", i.e. different postures etc. 
    On large videos this code is slow. Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior. 
    Note: this method can return fewer images than numframes2pick.'''
    
    print("Kmeans-quantization based extracting of frames from", start*clip.duration," seconds to", stop*clip.duration, " seconds.")
    startindex=int(np.floor(clip.fps*clip.duration*start))
    stopindex=int(np.ceil(clip.fps*clip.duration*stop))
    if Index=="all":
        Index=np.arange(stopindex-startindex)+startindex
    else:
        Index=np.array(Index)
        Index=Index[(Index>startindex)*(Index<stopindex)] #crop to range!
    
    nframes=len(Index)
    if batchsize>nframes:
        batchsize=int(nframes/2)
    
    if len(Index)>=numframes2pick:
        clipresized=clip.resize(width=resizewidth)
        ny, nx = clipresized.size
        
        DATA=np.zeros((nframes,nx,ny))
        frame0=img_as_ubyte(clip.get_frame(0))
        if np.ndim(frame0)==3:
            ncolors=np.shape(frame0)[2]
        else:
            ncolors=1
            
        print("Extracting...",nframes, " (this might take a while).")
        for counter,index in tqdm(enumerate(Index)):
            if ncolors==1:
                DATA[counter,:,:] = img_as_ubyte(clipresized.get_frame(start+index * 1. / clipresized.fps))
            else: #attention averages over color channels to keep size small / perhaps you want to use color information?
                DATA[counter,:,:] = img_as_ubyte(np.array(np.mean(clipresized.get_frame(start+index * 1. / clipresized.fps),2),dtype=np.uint8))
            
        print("Kmeans clustering...(this might take a while).")
        data = DATA - DATA.mean(axis=0)
        data=data.reshape(nframes,-1) #stacking
        kmeans=MiniBatchKMeans(n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize,max_iter=max_iter)
        kmeans.fit(data)

        frames2pick=[]
        for clusterid in range(numframes2pick):
            clusterids=np.where(clusterid==kmeans.labels_)[0]
            numimagesofcluster=len(clusterids)
            if numimagesofcluster>0:
                frames2pick.append(Index[clusterids[np.random.randint(numimagesofcluster)]])

        return list(np.array(frames2pick))
    else:
        return list(Index)