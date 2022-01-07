import deeplabcut
import os

#from deeplabcut.pose_tracking_pytorch import  train_tracking_transformer
#from deeplabcut.pose_estimation_tensorflow import create_tracking_dataset
from .eval_tracker import *
import glob

def transformer_reID(path_config_file, videos, modelprefix = '', n_tracks=3, track_method ='ellipse', train_epochs = 100, n_triplets = 1000, shuffle = 1):
    # calling create_tracking_dataset, train_tracking_transformer, stitch_tracklets

    # should take number of triplets to mine
    deeplabcut.pose_estimation_tensorflow.create_tracking_dataset(path_config_file, videos, modelprefix = modelprefix, n_triplets = n_triplets)

    trainposeconfigfile, testposeconfigfile, snapshotfolder = deeplabcut.return_train_network_path(
        path_config_file,
        shuffle=shuffle,
        modelprefix=modelprefix,
        trainingsetindex=0
    )
    
    
    # modelprefix impacts where the model is loaded
    deeplabcut.pose_tracking_pytorch.train_tracking_transformer(path_config_file, videos, modelprefix = modelprefix, train_epochs = train_epochs, ckpt_folder = snapshotfolder)


    transformer_checkpoint = os.path.join(snapshotfolder, f'dlc_transreid_{train_epochs}.pth')


    if not os.path.exists(transformer_checkpoint):
        raise FileNotFoundError(f'checkpoint {transformer_checkpoint} not found')

    deeplabcut.stitch_tracklets(path_config_file, videos, track_method=track_method, modelprefix = modelprefix, n_tracks = n_tracks , transformer_checkpoint = transformer_checkpoint)
    

def eval_transformer(path_config_file, path_to_ground_truth, video, modelprefix = '',  n_tracks = 3, top_frames = 5, shuffle = 1):

    # get path_to_el_pickle from video path
    # get path_to_features from video path

    trainposeconfigfile, testposeconfigfile, snapshotfolder = deeplabcut.return_train_network_path(
        path_config_file,
        shuffle=shuffle,
        modelprefix=modelprefix,
        trainingsetindex=0
    )

    
    ckpts = glob.glob(os.path.join(snapshotfolder,'dlc_transreid_*.pth'))

    if len(ckpts)==0:
        raise FileNotFoundError('transformer checkpoint not found')

    # use the very last one
    transformer_checkpoint = os.path.join(snapshotfolder, ckpts[-1])
        
    
    vname = Path(video).stem
    videofolder = str(Path(video).parents[0])
    feature_fnames = glob.glob(os.path.join(videofolder,vname +'*_bpt_features.mmdpickle'))    
    el_fnames = glob.glob(os.path.join(videofolder, vname +'*_el.pickle'))

    nframe = len(VideoWriter(video))
    zfill_width = int(np.ceil(np.log10(nframe)))
    
    assert len(feature_fnames) == 1 and len(el_fnames) == 1
    path_to_features = feature_fnames[0]

    feature_dict = mmapdict(path_to_features, True)            
    
    path_to_el_pickle = el_fnames[0]                                      
    
    prox, viz = calc_proximity_and_visibility_indices(path_to_ground_truth)

    thres = np.percentile(prox, 100-top_frames)

    crossing_indices =  prox>thres

    crossing_frames = np.where(crossing_indices == True)



    print ('path_to_ground_truth', path_to_ground_truth)
    print ('path_to_el_pickle', path_to_el_pickle)
    print ('path_to_features', path_to_features)
        
    ground_truth = pd.read_hdf(path_to_ground_truth)
    
    ground_truth_data = reconstruct_all_bboxes(
        ground_truth, 0, to_xywh=True
    )


    
    dlctrans = inference.DLCTrans(transformer_checkpoint)
    
    def trans_weight_func(tracklet1,tracklet2, nframe, feature_dict):

        if tracklet1 < tracklet2:
            ind_img1 = tracklet1.inds[-1]            
            coord1 = tracklet1.data[-1][:,:2]
            ind_img2 = tracklet2.inds[0]            
            coord2 = tracklet2.data[0][:,:2]
        else:
            ind_img2 = tracklet2.inds[-1]
            ind_img1 = tracklet1.inds[0]
            coord2 = tracklet2.data[-1][:,:2]
            coord1 = tracklet1.data[0][:,:2]


        t1 = (coord1, ind_img1)
        t2 = (coord2, ind_img2)

        dist = dlctrans(t1,t2, zfill_width, feature_dict,return_features = False)

        
        dist = (dist+1)/2
        w = 0.01 if tracklet1.identity == tracklet2.identity else 1
        cost = w * stitcher.calculate_edge_weight(tracklet1, tracklet2)
 
        return -dist



    def original_weight(tracklet1, tracklet2):
        w = 0.01 if tracklet1.identity == tracklet2.identity else 1
        cost = w * stitcher.calculate_edge_weight(tracklet1, tracklet2)            
        return cost

        
    
    stitcher = TrackletStitcher.from_pickle(path_to_el_pickle, n_tracks)

    
    stitcher.build_graph(weight_func = partial(trans_weight_func, nframe = nframe, feature_dict = feature_dict))
    stitcher.stitch()
    df = stitcher.format_df().reindex(range(ground_truth_data.shape[1]))
    bboxes_with_stitcher = reconstruct_all_bboxes(df, 0, to_xywh=True)

    #print ('crossing num')
    #print (np.sum(crossing_indices))
        

    try:
        temp = compute_mot_metrics_bboxes(bboxes_with_stitcher[:,crossing_indices,:], ground_truth_data[:,crossing_indices,:])
    except:
        temp = compute_mot_metrics_bboxes(bboxes_with_stitcher, ground_truth_data)

    print_all_metrics([temp])

