"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import os
import numpy as np
import matplotlib as mpl
import platform
from pathlib import Path
if os.environ.get('DLClight', default=False) == 'True':
    mpl.use('AGG') #anti-grain geometry engine #https://matplotlib.org/faq/usage_faq.html
    pass
elif platform.system() == 'Darwin':
    mpl.use('WXAgg')
else:
    mpl.use('TkAgg') #TkAgg
import matplotlib.pyplot as plt
from deeplabcut.utils.auxiliaryfunctions import attempttomakefolder
from matplotlib.collections import LineCollection
from skimage import io
from tqdm import trange


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def MakeLabeledImage(DataCombined,imagenr,pcutoff,imagebasefolder,Scorers,bodyparts,colors,cfg,labels=['+','.','x'],scaling=1):
    '''Creating a labeled image with the original human labels, as well as the DeepLabCut's! '''
    from skimage import io

    alphavalue=cfg['alphavalue'] #.5
    dotsize=cfg['dotsize'] #=15

    plt.axis('off')
    im=io.imread(os.path.join(imagebasefolder,DataCombined.index[imagenr]))
    if np.ndim(im)>2: #color image!
        h,w,numcolors=np.shape(im)
    else:
        h,w=np.shape(im)
    plt.figure(frameon=False,figsize=(w*1./100*scaling,h*1./100*scaling))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.imshow(im,'gray')
    for scorerindex,loopscorer in enumerate(Scorers):
       for bpindex,bp in enumerate(bodyparts):
           if np.isfinite(DataCombined[loopscorer][bp]['y'][imagenr]+DataCombined[loopscorer][bp]['x'][imagenr]):
                y,x=int(DataCombined[loopscorer][bp]['y'][imagenr]), int(DataCombined[loopscorer][bp]['x'][imagenr])
                if cfg["scorer"] not in loopscorer:
                    p=DataCombined[loopscorer][bp]['likelihood'][imagenr]
                    if p>pcutoff:
                        plt.plot(x,y,labels[1],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                    else:
                        plt.plot(x,y,labels[2],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                else: #this is the human labeler
                        plt.plot(x,y,labels[0],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
    plt.xlim(0,w)
    plt.ylim(0,h)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().invert_yaxis()


def PlottingandSaveLabeledFrame(DataCombined,ind,trainIndices,cfg,colors,comparisonbodyparts,DLCscorer,foldername,scaling=1):
        fn=Path(cfg['project_path']+'/'+DataCombined.index[ind])
        imagename=fn.parts[-1] #fn.stem+fn.suffix
        imfoldername=fn.parts[-2] #fn.suffix
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        MakeLabeledImage(DataCombined,ind,cfg["pcutoff"],cfg["project_path"],[cfg["scorer"],DLCscorer],comparisonbodyparts,colors,cfg,scaling=scaling)

        if ind in trainIndices:
            full_path = os.path.join(foldername,'Training-'+imfoldername+'-'+imagename)
        else:
            full_path = os.path.join(foldername,'Test-'+imfoldername+'-'+imagename)

        # windows throws error if file path is > 260 characters, can fix with prefix. see https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file#maximum-path-length-limitation
        if (len(full_path) >= 260) and (os.name == 'nt'):
            full_path = '\\\\?\\'+full_path
        plt.savefig(full_path)

        plt.close("all")


def prepare_figure_axes(width, height, scale=1, dpi=100):
    fig = plt.figure(frameon=False, figsize=(width * scale / dpi, height * scale / dpi))
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    return fig, ax


def make_labeled_images_from_dataframe(df, cfg, destfolder='', scale=1., dpi=100,
                                       keypoint='+', draw_skeleton=True, color_by='bodypart'):
    """
    Write labeled frames to disk from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the labeled data. Typically, the DataFrame is obtained
        through pandas.read_csv() or pandas.read_hdf().

    cfg : dict
        Project configuration.

    destfolder : string, optional
        Destination folder into which images will be stored. By default, same location as the labeled data.
        Note that the folder will be created if it does not exist.

    scale : float, optional
        Up/downscale the output dimensions.
        By default, outputs are of the same dimensions as the original images.

    dpi : int, optional
        Output resolution. 100 dpi by default.

    keypoint : str, optional
        Keypoint appearance. By default, keypoints are marked by a + sign.
        Refer to https://matplotlib.org/3.2.1/api/markers_api.html for a list of all possible options.

    draw_skeleton : bool, optional
        Whether to draw the animal skeleton as defined in *cfg*. True by default.

    color_by : str, optional
        Color scheme of the keypoints. Must be either 'bodypart' or 'individual'.
        By default, keypoints are colored relative to the bodypart they represent.
    """

    bodyparts = df.columns.get_level_values('bodyparts')
    bodypart_names = bodyparts.unique()
    nbodyparts = len(bodypart_names)
    bodyparts = bodyparts[::2]

    if color_by == 'bodypart':
        map_ = bodyparts.map(dict(zip(bodypart_names, range(nbodyparts))))
        cmap = get_cmap(nbodyparts, cfg['colormap'])
        colors = cmap(map_)
    elif color_by == 'individual':
        try:
            individuals = df.columns.get_level_values('individuals')
            individual_names = individuals.unique().to_list()
            nindividuals = len(individual_names)
            individuals = individuals[::2]
            map_ = individuals.map(dict(zip(individual_names, range(nindividuals))))
            cmap = get_cmap(nindividuals, cfg['colormap'])
            colors = cmap(map_)
        except KeyError as e:
            raise Exception('Coloring by individuals is only valid for multi-animal data') from e
    else:
        raise ValueError('`color_by` must be either `bodypart` or `individual`.')

    bones = []
    if draw_skeleton:
        for bp1, bp2 in cfg['skeleton']:
            match1, match2 = [], []
            for j, bp in enumerate(bodyparts):
                if bp == bp1:
                    match1.append(j)
                elif bp == bp2:
                    match2.append(j)
            bones.extend(zip(match1, match2))
    ind_bones = tuple(zip(*bones))

    sep = '/' if '/' in df.index[0] else '\\'
    images = cfg['project_path'] + sep + df.index
    if sep != os.path.sep:
        images = images.str.replace(sep, os.path.sep)
    if not destfolder:
        destfolder = os.path.dirname(images[0])
    tmpfolder = destfolder + '_labeled'
    attempttomakefolder(tmpfolder)
    ic = io.imread_collection(images.to_list())

    h, w = ic[0].shape[:2]
    fig, ax = prepare_figure_axes(w, h, scale, dpi)
    im = ax.imshow(np.zeros((h, w)), 'gray')
    scat = ax.scatter([], [], s=cfg['dotsize'], alpha=cfg['alphavalue'], marker=keypoint)
    scat.set_color(colors)
    xy = df.values.reshape((df.shape[0], -1, 2))
    segs = xy[:, ind_bones].swapaxes(1, 2)
    coll = LineCollection([], colors=cfg['skeleton_color'])
    ax.add_collection(coll)
    for i in trange(len(ic)):
        coords = xy[i]
        im.set_array(ic[i])
        scat.set_offsets(coords)
        if ind_bones:
            coll.set_segments(segs[i])
        imagename = os.path.basename(ic.files[i])
        fig.savefig(os.path.join(tmpfolder, imagename.replace('.png', f'_{color_by}.png')))
    plt.close(fig)
