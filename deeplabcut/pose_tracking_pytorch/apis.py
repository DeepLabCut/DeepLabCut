import deeplabcut
import os
from deeplabcut.utils import auxiliaryfunctions


def transformer_reID(
    path_config_file,
    videos,
    n_tracks=None,
    train_frac = 0.8,
    trainingsetindex=0, # this is only used to get DLC scorer, might need further tweaking to work
    modelprefix="",
    track_method="ellipse",
    train_epochs=100,
    n_triplets=1000,
    videotype="mp4",
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

    videotype: (optional) str
        extension for the video file

    """

    # calling create_tracking_dataset, train_tracking_transformer, stitch_tracklets

    # should take number of triplets to mine

    cfg = auxiliaryfunctions.read_config(path_config_file)

    DLCscorer, _ = deeplabcut.utils.auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle,
        cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    deeplabcut.pose_estimation_tensorflow.create_tracking_dataset(
        path_config_file, videos, track_method, modelprefix=modelprefix, n_triplets=n_triplets, videotype = videotype
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
        DLCscorer,
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
        videotype = videotype,
        transformer_checkpoint=transformer_checkpoint,
    )
