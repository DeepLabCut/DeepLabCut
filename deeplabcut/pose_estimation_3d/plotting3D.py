"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import os
import glob
import subprocess
import matplotlib.pyplot as plt
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils import auxiliaryfunctions_3d
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from matplotlib import gridspec
from tqdm import tqdm

def create_labeled_video_3d(config,path,videofolder=None,start=0,end=None,trailpoints=0,videotype='avi',view=[-113, -270],xlim=[None,None],ylim=[None,None],zlim=[None,None],draw_skeleton=True):
    """
    Creates a video with views from the two cameras and the 3d reconstruction for a selected number of frames.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    path : list
        A list of strings containing the full paths to triangulated files for analysis or a path to the directory, where all the triangulated files are stored.

    videofolder: string
        Full path of the folder where the videos are stored. Use this if the vidoes are stored in a different location other than where the triangulation files are stored. By default is ``None`` and therefore looks for video files in the directory where the triangulation file is stored.

    start: int
        Integer specifying the start of frame index to select. Default is set to 0.

    end: int
        Integer specifying the end of frame index to select. Default is set to None, where all the frames of the video are used for creating the labeled video.

    trailpoints: int
        Number of revious frames whose body parts are plotted in a frame (for displaying history). Default is set to 0.

    videotype: string, optional
        Checks for the extension of the video in case the input is a directory.\nOnly videos with this extension are analyzed. The default is ``.avi``

    view: list
        A list that sets the elevation angle in z plane and azimuthal angle in x,y plane of 3d view. Useful for rotating the axis for 3d view

    xlim: list
        A list of integers specifying the limits for xaxis of 3d view. By default it is set to [None,None], where the x limit is set by taking the minimum and maximum value of the x coordinates for all the bodyparts.

    ylim: list
        A list of integers specifying the limits for yaxis of 3d view. By default it is set to [None,None], where the y limit is set by taking the minimum and maximum value of the y coordinates for all the bodyparts.

    zlim: list
        A list of integers specifying the limits for zaxis of 3d view. By default it is set to [None,None], where the z limit is set by taking the minimum and maximum value of the z coordinates for all the bodyparts.

    draw_skeleton: bool
        If ``True`` adds a line connecting the body parts making a skeleton on on each frame. The body parts to be connected and the color of these connecting lines are specified in the config file. By default: ``True``

    Example
    -------
    Linux/MacOs
    >>> deeplabcut.create_labeled_video_3d(config,['/data/project1/videos/3d.h5'],start=100, end=500)

    To create labeled videos for all the triangulated files in the folder
    >>> deeplabcut.create_labeled_video_3d(config,['/data/project1/videos'],start=100, end=500)

    To set the xlim, ylim, zlim and rotate the view of the 3d axis
    >>> deeplabcut.create_labeled_video_3d(config,['/data/project1/videos'],start=100, end=500,view=[30,90],xlim=[-12,12],ylim=[15,25],zlim=[20,30])

    """
    start_path=os.getcwd()

    # Read the config file and related variables
    cfg_3d = auxiliaryfunctions.read_config(config)
    cam_names = cfg_3d['camera_names']
    pcutoff = cfg_3d['pcutoff']
    markerSize = cfg_3d['dotsize']
    alphaValue = cfg_3d['alphaValue']
    cmap = cfg_3d['colormap']
    bodyparts2connect = cfg_3d['skeleton']
    skeleton_color = cfg_3d['skeleton_color']
    scorer_3d = cfg_3d['scorername_3d']

    # Flatten the list of bodyparts to connect
    bodyparts2plot = list(np.unique([val for sublist in bodyparts2connect for val in sublist]))
    color = plt.cm.get_cmap(cmap, len(bodyparts2plot))
    file_list = auxiliaryfunctions_3d.Get_list_of_triangulated_and_videoFiles(path,videotype,scorer_3d,cam_names,videofolder)
    print(file_list)
    if file_list == []:
        raise Exception("No corresponding video file(s) found for the specified triangulated file or folder. Did you specify the video file type? If videos are stored in a different location, please use the ``videofolder`` argument to specify their path.")

    for file in file_list:
        path_h5_file = Path(file[0]).parents[0]
        triangulate_file = file[0]
        # triangulated file is a list which is always sorted as [triangulated.h5,camera-1.videotype,camera-2.videotype]
        #name for output video
        file_name = str(Path(triangulate_file).stem)
        if os.path.isfile(os.path.join(path_h5_file,file_name+'.mpg')):
            print("Video already created...")
        else:
            string_to_remove = str(Path(triangulate_file).suffix)
            pickle_file = triangulate_file.replace(string_to_remove,'_includingmetadata.pickle')
            metadata_ = auxiliaryfunctions_3d.LoadMetadata3d(pickle_file)

            base_filename_cam1 = str(Path(file[1]).stem).split(videotype)[0] # required for searching the filtered file
            base_filename_cam2 = str(Path(file[2]).stem).split(videotype)[0] # required for searching the filtered file
            cam1_view_video = file[1]
            cam2_view_video = file[2]
            cam1_scorer = metadata_['scorer_name'][cam_names[0]]
            cam2_scorer = metadata_['scorer_name'][cam_names[1]]
            print("Creating 3D video from %s and %s using %s"%(Path(cam1_view_video).name,Path(cam2_view_video).name,Path(triangulate_file).name))

            # Read the video files and corresponfing h5 files
            vid_cam1 = cv2.VideoCapture(cam1_view_video)
            vid_cam2 = cv2.VideoCapture(cam2_view_video)

            # Look for the filtered predictions file
            try:
                print("Looking for filtered predictions...")
                df_cam1= pd.read_hdf(glob.glob(os.path.join(path_h5_file,str('*'+base_filename_cam1+cam1_scorer+'*filtered.h5')))[0])
                df_cam2 = pd.read_hdf(glob.glob(os.path.join(path_h5_file,str('*'+base_filename_cam2+cam2_scorer+'*filtered.h5')))[0])
                #print("Found filtered predictions, will be use these for triangulation.")
                print("Found the following filtered data: ",os.path.join(path_h5_file,str('*'+base_filename_cam1+cam1_scorer+'*filtered.h5')),os.path.join(path_h5_file,str('*'+base_filename_cam2+cam2_scorer+'*filtered.h5')))
            except FileNotFoundError:
                print("No filtered predictions found, the unfiltered predictions will be used instead.")
                df_cam1= pd.read_hdf(glob.glob(os.path.join(path_h5_file,str(base_filename_cam1+cam1_scorer+'*.h5')))[0])
                df_cam2 = pd.read_hdf(glob.glob(os.path.join(path_h5_file,str(base_filename_cam2+cam2_scorer+'*.h5')))[0])

            df_3d = pd.read_hdf(triangulate_file,'df_with_missing')
            plt.rcParams.update({'figure.max_open_warning': 0})

            if end==None:
                end=len(df_3d) # All the frames
            frames = list(range(start,end,1))

            # Start plotting for every frame
            for k in tqdm(frames):
                output_folder,num_frames = plot2D(cfg_3d,k,bodyparts2plot,vid_cam1,vid_cam2,bodyparts2connect,df_cam1,df_cam2,df_3d,pcutoff,markerSize,alphaValue,color,path_h5_file,file_name,skeleton_color,view,draw_skeleton,trailpoints,xlim,ylim,zlim)

            # Once all the frames are saved, then make a movie using ffmpeg.
            cwd = os.getcwd()
            os.chdir(str(output_folder))
            subprocess.call(['ffmpeg', '-start_number', str(start) ,'-framerate', str(30), '-i', str('img%0'+str(num_frames)+'d.png'), '-r', str(30),'-vb', '20M' ,os.path.join(output_folder,str('../'+file_name+'.mpg'))])
            os.chdir(cwd)

    os.chdir(start_path)

def plot2D(cfg_3d,k,bodyparts2plot,vid_cam1,vid_cam2,bodyparts2connect,df_cam1_view,df_cam2_view,xyz_pts,pcutoff,markerSize,alphaValue,color,path_h5_file,file_name,skeleton_color,view,draw_skeleton,trailpoints,xlim,ylim,zlim):
    """
    Creates 2D gif for a selected number of frames
    """
    # Create the fig and define the axes
    gs = gridspec.GridSpec(1, 3,width_ratios=[1,1,1])
    fig = plt.figure(figsize=(20,8))
    axes1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
    axes2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
    axes3 = fig.add_subplot(gs[0, 2],projection='3d') # row 1, span all columns
    fig.tight_layout()

    # Clear plot and initialize the variables
    plt.cla()
    axes1.cla()
    axes2.cla()
    axes3.cla()

    # Initialize arrays for appending the 3d data for actual plotting the points
    xdata_3d = []
    ydata_3d = []
    zdata_3d = []

    # Initialize arrays for appending the 3d data for drawing the lines
    xlines_3d = []
    ylines_3d = []
    zlines_3d = []

    # Initialize arrays for appending the 2d data from cam1 for actual plotting the points
    xdata_cam1=[]
    ydata_cam1=[]

    # Initialize arrays for appending the 2d data from cam2 for actual plotting the points
    xdata_cam2=[]
    ydata_cam2=[]

    # Initialize arrays for appending the 2d data from cam1 for drawing the lines
    xcam1 = []
    ycam1 = []

    # Initialize arrays for appending the 2d data from cam2 for drawing the lines
    xcam2 = []
    ycam2 = []

    # Get the scorer names from the dataframe
    scorer_cam1 = df_cam1_view.columns.get_level_values(0)[0]
    scorer_cam2 = df_cam2_view.columns.get_level_values(0)[0]
    scorer_3d = xyz_pts.columns.get_level_values(0)[0]

    # Set the x,y, and z limits for the 3d view

    numberFrames = min([int(vid_cam1.get(cv2.CAP_PROP_FRAME_COUNT)),int(vid_cam2.get(cv2.CAP_PROP_FRAME_COUNT))]) #minimum of two cameras / TODO: clean up!
    df_x = np.empty((len(bodyparts2plot),numberFrames))
    df_y = np.empty((len(bodyparts2plot),numberFrames))
    df_z = np.empty((len(bodyparts2plot),numberFrames))
    for bpindex, bp in enumerate(bodyparts2plot):
        df_x[bpindex,:]=xyz_pts[scorer_3d][bp]['x'].values
        df_y[bpindex,:]=xyz_pts[scorer_3d][bp]['y'].values
        df_z[bpindex,:]=xyz_pts[scorer_3d][bp]['z'].values
    if xlim==[None,None]:
        axes3.set_xlim3d([np.nanmin(df_x),np.nanmax(df_x)])
    else:
        axes3.set_xlim3d(xlim)
    if ylim==[None,None]:
        axes3.set_ylim3d([np.nanmin(df_y),np.nanmax(df_y)])
    else:
        axes3.set_ylim3d(ylim)
    if zlim==[None,None]:
        axes3.set_zlim3d([np.nanmin(df_z),np.nanmax(df_z)])
    else:
        axes3.set_zlim3d(zlim)

    axes3.set_xticklabels([])
    axes3.set_yticklabels([])
    axes3.set_zticklabels([])
    axes3.xaxis.grid(False)
    axes3.view_init(view[0], view[1])
    axes3.set_xlabel('X', fontsize=10)
    axes3.set_ylabel('Y', fontsize=10)
    axes3.set_zlabel('Z', fontsize=10)


    # Set the frame number to read#max(0,index-trailpoints):index
    vid_cam1.set(1,k)
    vid_cam2.set(1,k)
    ret_cam1, frame_cam1 = vid_cam1.read() #TODO: use ret_camj
    ret_cam2,frame_cam2 = vid_cam2.read()

    # Plot the labels for each body part
    for bpindex,bp in enumerate(bodyparts2plot):
        axes1.imshow(cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2RGB))
        axes2.imshow(cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2RGB))
        if (df_cam1_view[scorer_cam1][bp]['likelihood'].values[k]) > pcutoff and (df_cam2_view[scorer_cam2][bp]['likelihood'].values[k]) > pcutoff:
            if trailpoints > 0:
                xdata_cam1.append(df_cam1_view.iloc[max(0,k-trailpoints):k][scorer_cam1][bp]['x'])
                ydata_cam1.append(df_cam1_view.iloc[max(0,k-trailpoints):k][scorer_cam1][bp]['y'])
                xdata_cam2.append(df_cam2_view.iloc[max(0,k-trailpoints):k][scorer_cam2][bp]['x'])
                ydata_cam2.append(df_cam2_view.iloc[max(0,k-trailpoints):k][scorer_cam2][bp]['y'])
                xdata_3d.append(xyz_pts.iloc[max(0,k-trailpoints):k][scorer_3d][bp]['x'])
                ydata_3d.append(xyz_pts.iloc[max(0,k-trailpoints):k][scorer_3d][bp]['y'])
                zdata_3d.append(xyz_pts.iloc[max(0,k-trailpoints):k][scorer_3d][bp]['z'])
            else:
                xdata_cam1.append(df_cam1_view.iloc[k][scorer_cam1][bp]['x'])
                ydata_cam1.append(df_cam1_view.iloc[k][scorer_cam1][bp]['y'])
                xdata_cam2.append(df_cam2_view.iloc[k][scorer_cam2][bp]['x'])
                ydata_cam2.append(df_cam2_view.iloc[k][scorer_cam2][bp]['y'])
                xdata_3d.append(xyz_pts.iloc[k][scorer_3d][bp]['x'])
                ydata_3d.append(xyz_pts.iloc[k][scorer_3d][bp]['y'])
                zdata_3d.append(xyz_pts.iloc[k][scorer_3d][bp]['z'])
        else:
            xdata_cam1.append(np.nan)
            ydata_cam1.append(np.nan)
            xdata_cam2.append(np.nan)
            ydata_cam2.append(np.nan)
            xdata_3d.append(np.nan)
            ydata_3d.append(np.nan)
            zdata_3d.append(np.nan)
        p =axes1.scatter(xdata_cam1[bpindex],ydata_cam1[bpindex],s=markerSize,c=color(bodyparts2plot.index(bp)))
        p =axes2.scatter(xdata_cam2[bpindex],ydata_cam2[bpindex],s=markerSize,c=color(bodyparts2plot.index(bp)))
        p =axes3.scatter(xdata_3d[bpindex],ydata_3d[bpindex],zdata_3d[bpindex],c=color(bodyparts2plot.index(bp)))

    # Connecting the bodyparts specified in the config file.3d file is created based on the likelihoods of cam1 and cam2. Using 3d file and check if the body part is nan then dont plot skeleton
    if draw_skeleton:
        for i in range(len(bodyparts2connect)):
            bool_above_pcutoff = [np.isnan(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][0]]['x']) or (np.isnan(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][1]]['x']))]
            if not bool_above_pcutoff[0]:
                xcam1.append(df_cam1_view.iloc[k][scorer_cam1][bodyparts2connect[i][0]]['x'])
                ycam1.append(df_cam1_view.iloc[k][scorer_cam1][bodyparts2connect[i][0]]['y'])
                xcam1.append(df_cam1_view.iloc[k][scorer_cam1][bodyparts2connect[i][1]]['x'])
                ycam1.append(df_cam1_view.iloc[k][scorer_cam1][bodyparts2connect[i][1]]['y'])

                xcam2.append(df_cam2_view.iloc[k][scorer_cam2][bodyparts2connect[i][0]]['x'])
                ycam2.append(df_cam2_view.iloc[k][scorer_cam2][bodyparts2connect[i][0]]['y'])
                xcam2.append(df_cam2_view.iloc[k][scorer_cam2][bodyparts2connect[i][1]]['x'])
                ycam2.append(df_cam2_view.iloc[k][scorer_cam2][bodyparts2connect[i][1]]['y'])

                xlines_3d.append(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][0]]['x'])
                ylines_3d.append(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][0]]['y'])
                zlines_3d.append(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][0]]['z'])
                xlines_3d.append(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][1]]['x'])
                ylines_3d.append(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][1]]['y'])
                zlines_3d.append(xyz_pts.iloc[k][scorer_3d][bodyparts2connect[i][1]]['z'])

                axes1.plot(xcam1,ycam1, color=skeleton_color,alpha=alphaValue)
                axes2.plot(xcam2,ycam2, color=skeleton_color,alpha=alphaValue)
                axes3.plot(xlines_3d,ylines_3d,zlines_3d, color=skeleton_color,alpha=alphaValue)

                xcam1 = []
                ycam1 = []
                xcam2 = []
                ycam2 = []
                xlines_3d = []
                ylines_3d = []
                zlines_3d = []
    # Saving the frames
    output_folder = Path(os.path.join(path_h5_file,'temp_'+file_name))
    output_folder.mkdir(parents = True, exist_ok = True)
    num_frames = int(np.ceil(np.log10(numberFrames)))
    img_name = str(output_folder) +'/img'+str(k).zfill(num_frames) + '.png'
    plt.savefig(img_name)
    plt.close('all')
    return(output_folder,num_frames)
