import deeplabcut
import os

# from deeplabcut.pose_tracking_pytorch import  train_tracking_transformer
# from deeplabcut.pose_estimation_tensorflow import create_tracking_dataset
from .eval_tracker import reconstruct_all_bboxes, print_all_metrics,compute_mot_metrics_bboxes, calc_proximity_and_visibility_indices
import glob


def transformer_reID(
    path_config_file,
    dlcscorer,
    videos,
    n_tracks=None,
    train_frac = 0.8, 
    modelprefix="",
    track_method="ellipse",
    train_epochs=100,
    n_triplets=1000,
    shuffle=1,
):

    """

    Performs tracking with transformer.

    Substeps include:

    Mines triplets from videos and these triplets are later used to tran a transformer that's
    able to perform reID. The transformer is then used as a stitching loss when tracklets are
    stitched during tracking.

    Outputs: The tracklet file is saved as _track_trans.h5 in the same folder where the non transformer tracklet file is stored.

    Parameters
    ----------
    path_config_file: string
        Full path of the config.yaml file as a string.

    dlcscorer: string
        dlcscorer that kepps track of the model it uses

    videos: list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    n_tracks: int

        number of tracks to be formed in the videos. (TODO) handling videos with different number of tracks

    train_epochs: (optional), int

        number of epochs to train the transformer

    n_triplets: (optional) int

        number of triplets to be mined from the videos


    """

    # calling create_tracking_dataset, train_tracking_transformer, stitch_tracklets

    # should take number of triplets to mine


    
    deeplabcut.pose_estimation_tensorflow.create_tracking_dataset(
        path_config_file, videos, track_method, modelprefix=modelprefix, n_triplets=n_triplets
    )

    (
        trainposeconfigfile,
        testposeconfigfile,
        snapshotfolder,
    ) = deeplabcut.return_train_network_path(
        path_config_file, shuffle=shuffle, modelprefix=modelprefix, trainingsetindex=0
    )

    # modelprefix impacts where the model is loaded
    deeplabcut.pose_tracking_pytorch.train_tracking_transformer(
        path_config_file,
        dlcscorer,
        videos,
        train_frac = train_frac, 
        modelprefix=modelprefix,
        train_epochs=train_epochs,
        ckpt_folder=snapshotfolder,
    )

    transformer_checkpoint = os.path.join(
        snapshotfolder, f"dlc_transreid_{train_epochs}.pth"
    )

    if not os.path.exists(transformer_checkpoint):
        raise FileNotFoundError(f"checkpoint {transformer_checkpoint} not found")

    deeplabcut.stitch_tracklets(
        path_config_file,
        videos,
        track_method=track_method,
        modelprefix=modelprefix,
        n_tracks=n_tracks,
        transformer_checkpoint=transformer_checkpoint,
    )


def eval_tracking(
    path_config_file,
    dlcscorer,
    path_to_ground_truth,
    video,
    n_tracks,
    track_method,        
    modelprefix="",
    top_frames=5,
    shuffle=1,
    use_trans=False,
):

    """

    Evaluating tracking result based on ground truth for a video

    Outputs: standard tracking metrics such as mota

    Parameters
    ----------
    path_config_file: string
        Full path of the config.yaml file as a string.

    path_to_ground_truth

        tracking ground truth file to the video

    video: string
        Path to the video we want to evaluate

    top_frames: (optional), float, in the range (0,100) as percentage

        Option to only evaluate top_frames % of these frames with highest scene density

    n_tracks: (optinal), int

        number of tracks to be formed in the videos. (TODO) handling videos with different number of tracks

    use_trans: (optional) boolean

        if true, evaluate transformer based tracking. if false, evaluate non transformer based tracking


    """

    if use_trans:
        eval_transformer(
            path_config_file,
            dlcscorer,
            path_to_ground_truth,
            video,
            n_tracks,
            track_method,
            modelprefix=modelprefix,
            top_frames=top_frames,
            shuffle=shuffle,
        )
    else:
        eval_non_transformer(
            path_config_file,
            dlcscorer,
            path_to_ground_truth,
            video,
            n_tracks,
            track_method,
            modelprefix=modelprefix,
            top_frames=top_frames,
            shuffle=shuffle,
        )


def eval_non_transformer(
    path_config_file,
    dlcscorer,
    path_to_ground_truth,
    video,
    n_tracks,
    track_method,
    modelprefix="",
    top_frames=5,
    shuffle=1,
):

    if track_method == 'ellipse':
        method = 'el'
    elif track_method == 'box':
        method = 'bx'
    elif track_method == 'skeleton':
        method = 'sk'
    else:
        raise ValueError (f'{track_method} is not supported here')
    
    vname = Path(video).stem
    videofolder = str(Path(video).parents[0])
    track_fnames = os.path.join(videofolder, vname + dlcscorer + f"_{method}.pickle")
    
    path_to_track_pickle = track_fnames[-1]
    prox, viz = calc_proximity_and_visibility_indices(path_to_ground_truth)
    thres = np.percentile(prox, 100 - top_frames)
    crossing_indices = prox > thres

    ground_truth = pd.read_hdf(path_to_ground_truth)
    ground_truth_data = reconstruct_all_bboxes(ground_truth, 0, to_xywh=True)
    stitcher = TrackletStitcher.from_pickle(path_to_track_pickle, n_tracks)
    stitcher.build_graph()
    stitcher.stitch()
    df = stitcher.format_df().reindex(range(ground_truth_data.shape[1]))
    bboxes_with_stitcher = reconstruct_all_bboxes(df, 0, to_xywh=True)
    try:
        temp = compute_mot_metrics_bboxes(
            bboxes_with_stitcher[:, crossing_indices, :],
            ground_truth_data[:, crossing_indices, :],
        )
    except:
        temp = compute_mot_metrics_bboxes(bboxes_with_stitcher, ground_truth_data)
    print_all_metrics([temp])

def eval_transformer(
    path_config_file,
    dlcscorer,
    path_to_ground_truth,
    video,
    n_tracks,
    track_method,    
    modelprefix="",
    top_frames=5,
    shuffle=1,
):

    # get path_to_track_pickle from video path
    # get path_to_features from video path

    if track_method == 'ellipse':
        method = 'el'
    elif track_method == 'box':
        method = 'bx'
    elif track_method =='skeleton':
        method = 'sk'
    else:
        raise ValueError (f'{track_method} is not supported here')
    


    (
        trainposeconfigfile,
        testposeconfigfile,
        snapshotfolder,
    ) = deeplabcut.return_train_network_path(
        path_config_file, shuffle=shuffle, modelprefix=modelprefix, trainingsetindex=0
    )

    ckpts = glob.glob(os.path.join(snapshotfolder, "dlc_transreid_*.pth"))

    if len(ckpts) == 0:
        raise FileNotFoundError("transformer checkpoint not found")

    # use the very last one
    transformer_checkpoint = os.path.join(snapshotfolder, ckpts[-1])

    vname = Path(video).stem
    videofolder = str(Path(video).parents[0])
    feature_fnames = os.path.join(videofolder, vname + dlcscorer +  "_bpt_features.mmdpickle")

    track_fnames = os.path.join(videofolder, vname + dlcscroer + f"_{method}.pickle")

    nframe = len(VideoWriter(video))
    zfill_width = int(np.ceil(np.log10(nframe)))

    path_to_features = feature_fnames[-1]
    feature_dict = mmapdict(path_to_features, True)
    path_to_track_pickle = track_fnames[-1]    
    prox, viz = calc_proximity_and_visibility_indices(path_to_ground_truth)
    thres = np.percentile(prox, 100 - top_frames)
    crossing_indices = prox > thres
    print("path_to_ground_truth", path_to_ground_truth)
    print("path_to_track_pickle", path_to_track_pickle)
    print("path_to_features", path_to_features)
    ground_truth = pd.read_hdf(path_to_ground_truth)
    ground_truth_data = reconstruct_all_bboxes(ground_truth, 0, to_xywh=True)
    dlctrans = inference.DLCTrans(transformer_checkpoint)
    def trans_weight_func(tracklet1, tracklet2, nframe, feature_dict):

        if tracklet1 < tracklet2:
            ind_img1 = tracklet1.inds[-1]
            coord1 = tracklet1.data[-1][:, :2]
            ind_img2 = tracklet2.inds[0]
            coord2 = tracklet2.data[0][:, :2]
        else:
            ind_img2 = tracklet2.inds[-1]
            ind_img1 = tracklet1.inds[0]
            coord2 = tracklet2.data[-1][:, :2]
            coord1 = tracklet1.data[0][:, :2]

        t1 = (coord1, ind_img1)
        t2 = (coord2, ind_img2)

        dist = dlctrans(t1, t2, zfill_width, feature_dict, return_features=False)

        dist = (dist + 1) / 2
        
        # original cost 
        #w = 0.01 if tracklet1.identity == tracklet2.identity else 1
        #cost = w * stitcher.calculate_edge_weight(tracklet1, tracklet2)

        return -dist


    stitcher = TrackletStitcher.from_pickle(path_to_track_pickle, n_tracks)

    stitcher.build_graph(
        weight_func=partial(trans_weight_func, nframe=nframe, feature_dict=feature_dict)
    )
    stitcher.stitch()
    df = stitcher.format_df().reindex(range(ground_truth_data.shape[1]))
    bboxes_with_stitcher = reconstruct_all_bboxes(df, 0, to_xywh=True)


    try:
        temp = compute_mot_metrics_bboxes(
            bboxes_with_stitcher[:, crossing_indices, :],
            ground_truth_data[:, crossing_indices, :],
        )
    except:
        temp = compute_mot_metrics_bboxes(bboxes_with_stitcher, ground_truth_data)

    print_all_metrics([temp])
