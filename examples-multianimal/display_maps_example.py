import os, sys
os.environ['DLClight']='True' #to suppress gui support

import deeplabcut, pickle, os
import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm
import subprocess
from numpy import ma

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2, sys, math

params = {
   'axes.labelsize': 20,
   'axes.linewidth': 2.5,
   'legend.fontsize': 10,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'lines.linewidth': 2,
   'text.usetex': False,
   'figure.figsize': [4, 4],
   'font.size': 20,
   'figure.subplot.right': 0.99,
   'figure.subplot.top': 0.95,
   'figure.subplot.bottom': 0.17,
   'figure.subplot.left': 0.17,
   }

plt.rcParams.update(params)

#REQUIRED FOR INFERENCE:
nmspath = 'deeplabcut/pose_estimation_tensorflow/lib/nms_cython'
sys.path.append(os.path.join('/usr/local/lib/python3.6/dist-packages',nmspath))

configfile='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/sideviews/silversideschooling-valentina-2019-04-19/config.yaml'

outfolder='/media/alex/dropboxdisk/Dropbox/InterestingCode/social_datasets/sideviews/silversideschooling-valentina-2019-04-19/visualization'

deeplabcut.auxiliaryfunctions.attempttomakefolder(outfolder)
cfg=deeplabcut.auxiliaryfunctions.read_config(configfile)

Ind=[0,1,11,31] #indices for which to extract the maps!

filename=os.path.join(outfolder,'predictedmaps.pickle')
if os.path.isfile(filename):
    with open(filename, 'rb') as handle:
        Maps=pickle.load(handle)
else:
    Maps=deeplabcut.visualizemaps(configfile,0,Indices=Ind)
    with open(filename, 'wb') as f:
            pickle.dump(Maps, f,pickle.HIGHEST_PROTOCOL)



plotpaf=True
plotscmap=True #True
plotlocref=True #False
plotzoomlocref=True #True
scale=1 #adjust scale of image

#loading data for training fraction 95% and last snapshot (-1)
Data=Maps[0.95][-1]
print("Data for images:", Data.keys())
for imagenr in Data.keys():
    image,scmap,locref,paf,bptnames,pagraph,imname,trainingframe=Data[imagenr]


    ny,nx=np.shape(image)[:2]

    scmap = cv2.resize(scmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    LocrefX = cv2.resize(locref[:,:,:,0], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    LocrefY = cv2.resize(locref[:,:,:,1], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    plt.figure(0,frameon=False, figsize=(nx * 1. / 100*scale, ny * 1. / 100*scale))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    plt.imshow(image)

    plt.xlim(0, nx)
    plt.ylim(0, ny)
    plt.axis('off')
    plt.subplots_adjust(
        left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().invert_yaxis()
    plt.savefig(outfolder+'/Img'+str(imagenr)+'--'+str(imname.split('/')[-1]))

    if plotscmap:
        for jj,bpt in enumerate(bptnames):
            plt.figure(1,frameon=False, figsize=(nx * 1. / 100*scale, ny * 1. / 100*scale))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            plt.imshow(image)
            plt.imshow(scmap[:,:,jj], alpha=.5) # right elbow

            plt.xlim(0, nx)
            plt.ylim(0, ny)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(outfolder+'/Img'+str(imagenr)+'scmap'+bpt+'train'+str(trainingframe)+'.png')

            plt.close("all")

        plt.close("all")

    if plotlocref:
        for jj,bpt in enumerate(bptnames):
            plt.figure(2,frameon=False, figsize=(nx * 1. / 100*scale, ny * 1. / 100*scale))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            U = LocrefX[:,:,jj] #* -1
            V = LocrefY[:,:,jj]
            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')

            M[scmap[:,:,jj]<.65]=True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)

            plt.imshow(image, alpha = .25)
            plt.imshow(scmap[:,:,jj], alpha=.25)
            s = 5

            Q = plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], color='r',units='x',scale_units='xy',scale=1,angles='xy')
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()

            plt.savefig(outfolder+'/Img'+str(imagenr)+'locref'+bpt+'train'+str(trainingframe)+'.png')
            plt.close("all")

    if plotzoomlocref:
        margin=100
        for jj,bpt in enumerate(bptnames):
            plt.figure(2,frameon=False, figsize=(nx * 1. / 100*scale, ny * 1. / 100*scale))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            U = LocrefX[:,:,jj]
            V = LocrefY[:,:,jj]
            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')

            M[scmap[:,:,jj]<.5]=True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)

            plt.imshow(image, alpha = .25)
            plt.imshow(scmap[:,:,jj], alpha=.25) # right elbow

            s = 3
            Q = plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], color='r',units='x',scale_units='xy',scale=1,angles='xy')

            maxloc = np.unravel_index(np.argmax(scmap[:, :, jj]),scmap[:, :,jj].shape)

            plt.xlim(maxloc[1]-margin,maxloc[1]+margin)
            plt.ylim(maxloc[0]-margin,maxloc[0]+margin)

            #plt.xlim(0, nx)
            #plt.ylim(0, ny)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            ##### ATTENTION:
            # Check out: quiver when flipping axis -- https://github.com/matplotlib/matplotlib/issues/11857 >> #angles='xy'
            plt.gca().invert_yaxis()

            plt.savefig(outfolder+'/Img'+str(imagenr)+'locref'+bpt+'zoomed_train'+str(trainingframe)+'.png')
            plt.close("all")



    if plotpaf:
        for jj, edge in enumerate(pagraph):

            plt.figure(2,frameon=False, figsize=(nx * 1. / 100*scale, ny * 1. / 100*scale))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            U = paf[:,:,2*jj]
            V = paf[:,:,2*jj+1]

            X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
            M = np.zeros(U.shape, dtype='bool')

            M[U**2 + V**2 < 0.5 * 0.5**2] = True
            U = ma.masked_array(U, mask=M)
            V = ma.masked_array(V, mask=M)

            plt.imshow(image, alpha = .5)
            s = 5
            Q = plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s],
                           scale=50, headaxislength=4, alpha=1, width=0.002, color='r',angles='xy')

            plt.xlim(0, nx)
            plt.ylim(0, ny)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

            ##### ATTENTION:
            # Check out: quiver when flipping axis -- https://github.com/matplotlib/matplotlib/issues/11857 >> #angles='xy'
            plt.gca().invert_yaxis()

            plt.savefig(outfolder+'/Img'+str(imagenr)+'paf'+bptnames[edge[0]]+'_'+bptnames[edge[1]]+'train'+str(trainingframe)+'.png')

            plt.close("all")
