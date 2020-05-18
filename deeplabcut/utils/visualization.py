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
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal

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


def MakeLabeledPlots(folder,DataCombined,cfg,Labels,Colorscheme,cc,scale,visualizeindividuals,draw_skeleton=True):
    ''' MakeLabeledPlots for check_labels '''
    tmpfolder = str(folder) + '_labeled'
    auxiliaryfunctions.attempttomakefolder(tmpfolder)
    for index, imagename in enumerate(DataCombined.index.values):
        image = io.imread(os.path.join(cfg['project_path'],imagename))
        plt.axis('off')

        if np.ndim(image)==2:
            h, w = np.shape(image)
        else:
            h, w, nc = np.shape(image)

        plt.figure(
            frameon=False, figsize=(w * 1. / 100 * scale, h * 1. / 100 * scale))
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        plt.imshow(image, 'gray')
        if index==0:
            print("They are stored in the following folder: %s." %tmpfolder) #folder)
            if cfg.get('multianimalproject',False):
                print("Plotting labels for multi animal project.")

        #TODO: This code should be sped up by converting the pandas array into a numpy array with faster indexing as in CreateVideoSlow!
        # THIS indexing should do the job: DataCombined[cfg['scorer'],ind,bp,'x'][index]
        if cfg.get('multianimalproject',False):
            individuals,uniquebodyparts,multianimalbodyparts=auxfun_multianimal.extractindividualsandbodyparts(cfg)
            for ci,ind in enumerate(individuals):
                image_points = []
                if ind == 'single':
                    if visualizeindividuals:
                        for c, bp in enumerate(uniquebodyparts):
                            plt.plot(
                            DataCombined[cfg['scorer'],ind,bp,'x'][index],
                            DataCombined[cfg['scorer'],ind,bp,'y'][index],
                            Labels[cc],
                            color=Colorscheme(ci),
                            alpha=cfg['alphavalue'],
                            ms=cfg['dotsize'])
                    else:
                        for c, bp in enumerate(uniquebodyparts):
                            plt.plot(
                            DataCombined[cfg['scorer'],ind,bp,'x'][index],
                            DataCombined[cfg['scorer'],ind,bp,'y'][index],
                            Labels[cc],
                            color=Colorscheme(c),
                            alpha=cfg['alphavalue'],
                            ms=cfg['dotsize'])

                    if draw_skeleton:
                        x=lambda bp: DataCombined[cfg['scorer'],ind,bp,'x'][index]
                        y=lambda bp: DataCombined[cfg['scorer'],ind,bp,'y'][index]
                        for pair in cfg['skeleton']:
                           if pair[0] in uniquebodyparts and pair[1] in uniquebodyparts:
                               plt.plot([x(pair[0]),x(pair[1])],[y(pair[0]),y(pair[1])],color=cfg['skeleton_color'],alpha=cfg['alphavalue'])
                else:
                    if ind is None:
                        ind=np.NaN
                    if visualizeindividuals:
                        for c, bp in enumerate(multianimalbodyparts):
                            plt.plot(
                            DataCombined[cfg['scorer'],ind,bp,'x'][index],
                            DataCombined[cfg['scorer'],ind,bp,'y'][index],
                            Labels[cc],
                            color=Colorscheme(ci),
                            alpha=cfg['alphavalue'],
                            ms=cfg['dotsize'])
                    else:
                        for c, bp in enumerate(multianimalbodyparts):
                            plt.plot(
                            DataCombined[cfg['scorer'],ind,bp,'x'][index],
                            DataCombined[cfg['scorer'],ind,bp,'y'][index],
                            Labels[cc],
                            color=Colorscheme(c),
                            alpha=cfg['alphavalue'],
                            ms=cfg['dotsize'])

                    if draw_skeleton:
                       x=lambda bp: DataCombined[cfg['scorer'],ind,bp,'x'][index]
                       y=lambda bp: DataCombined[cfg['scorer'],ind,bp,'y'][index]
                       for pair in cfg['skeleton']:
                           if pair[0] in multianimalbodyparts and pair[1] in multianimalbodyparts:
                               plt.plot([x(pair[0]),x(pair[1])],[y(pair[0]),y(pair[1])],color=cfg['skeleton_color'],alpha=cfg['alphavalue'])

        else: #single animals
            for c, bp in enumerate(cfg['bodyparts']):
                plt.plot(
                    DataCombined[cfg['scorer'],bp,'x'][index],
                    DataCombined[cfg['scorer'],bp,'y'][index],
                    Labels[cc],
                    color=Colorscheme(c),
                    alpha=cfg['alphavalue'],
                    ms=cfg['dotsize'])

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()

        if cfg.get('multianimalproject',False):
            if visualizeindividuals:
                plt.savefig(os.path.join(tmpfolder,str(Path(imagename).stem)+'_individuals'+str(Path(imagename).suffix)))
            else:
                plt.savefig(os.path.join(tmpfolder,str(Path(imagename).name)))
        else:
            plt.savefig(os.path.join(tmpfolder,str(Path(imagename).name)))
        plt.close("all")


def make_labeled_image(DataCombined, imagenr, pcutoff, imagebasefolder, Scorers, bodyparts, colors, cfg, labels=['+', '.', 'x'], scaling=1):
    '''Creating a labeled image with the original human labels, as well as the DeepLabCut's! '''
    from skimage import io

    alphavalue=cfg['alphavalue'] #.5
    dotsize=cfg['dotsize'] #=15

    im=io.imread(os.path.join(imagebasefolder,DataCombined.index[imagenr]))
    if np.ndim(im)>2: #color image!
        h,w,numcolors=np.shape(im)
    else:
        h,w=np.shape(im)
    fig, ax = prepare_figure_axes(w, h, scaling)
    ax.imshow(im, 'gray')
    for scorerindex,loopscorer in enumerate(Scorers):
       for bpindex,bp in enumerate(bodyparts):
           if np.isfinite(DataCombined[loopscorer][bp]['y'][imagenr]+DataCombined[loopscorer][bp]['x'][imagenr]):
                y,x=int(DataCombined[loopscorer][bp]['y'][imagenr]), int(DataCombined[loopscorer][bp]['x'][imagenr])
                if cfg["scorer"] not in loopscorer:
                    p=DataCombined[loopscorer][bp]['likelihood'][imagenr]
                    if p>pcutoff:
                        ax.plot(x,y,labels[1],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                    else:
                        ax.plot(x,y,labels[2],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                else: #this is the human labeler
                        ax.plot(x,y,labels[0],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    return fig


def make_multianimal_labeled_image(frame, coords_truth, coords_pred, probs_pred, colors,
                                   dotsize=12, alphavalue=0.7, pcutoff=0.6, labels=['+', '.', 'x']):
    h, w, numcolors = np.shape(frame)
    fig, ax = prepare_figure_axes(w, h)
    ax.imshow(frame, 'gray')
    for n, data in enumerate(zip(coords_truth, coords_pred, probs_pred)):
        color = colors(n)
        coord_gt, coord_pred, prob_pred = data

        ax.plot(*coord_gt.T, labels[0], ms=dotsize,
                alpha=alphavalue, color=color)
        if not coord_pred.shape[0]:
            continue

        reliable = np.repeat(prob_pred >= pcutoff, coord_pred.shape[1], axis=1)
        ax.plot(*coord_pred[reliable[:, 0]].T, labels[1], ms=dotsize,
                alpha=alphavalue, color=color)
        if not np.all(reliable):
            ax.plot(*coord_pred[~reliable[:, 0]].T, labels[2], ms=dotsize,
                    alpha=alphavalue, color=color)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    return fig


def plot_and_save_labeled_frame(DataCombined, ind, trainIndices, cfg, colors, comparisonbodyparts, DLCscorer, foldername, scaling=1):
        fn=Path(cfg['project_path']+'/'+DataCombined.index[ind])
        fig = make_labeled_image(DataCombined, ind, cfg["pcutoff"], cfg["project_path"], [cfg["scorer"], DLCscorer], comparisonbodyparts, colors, cfg, scaling=scaling)
        save_labeled_frame(fig, fn, foldername, ind in trainIndices)


def save_labeled_frame(fig, image_path, dest_folder, belongs_to_train):
    path = Path(image_path)
    imagename = path.parts[-1]
    imfoldername = path.parts[-2]
    if belongs_to_train:
        dest = '-'.join(('Training', imfoldername, imagename))
    else:
        dest = '-'.join(('Test', imfoldername, imagename))
    full_path = os.path.join(dest_folder, dest)
    # Windows throws error if file path is > 260 characters, can fix with prefix.
    # See https://docs.microsoft.com/en-us/windows/desktop/fileio/naming-a-file#maximum-path-length-limitation
    if len(full_path) >= 260 and os.name == 'nt':
        full_path = '\\\\?\\' + full_path
    fig.savefig(full_path)
    plt.close(fig)


def prepare_figure_axes(width, height, scale=1., dpi=100):
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
