#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#


def transformer_reID(
    config: str,
    videos: list[str],
    videotype: str = "",
    shuffle: int = 1,
    trainingsetindex: int = 0,
    track_method: str = "ellipse",
    n_tracks: int | None = None,
    n_triplets: int = 1000,
    train_epochs: int = 100,
    train_frac: float = 0.8,
    modelprefix: str = "",
    destfolder: str = None,
):
    """
    Enables tracking with transformer.

    Substeps include:
        - Mines triplets from tracklets in videos (from another tracker)
        - These triplets are later used to tran a transformer with triplet loss
        - The transformer derived appearance similarity is then used as a stitching loss
            when tracklets are stitched during tracking.

    Outputs: The tracklet file is saved in the same folder where the non-transformer
    tracklet file is stored.

    Parameters
    ----------
    config: string
        Full path of the config.yaml file as a string.

    videos: list
        A list of strings containing the full paths to videos for analysis or a path to
        the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a
        directory. Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg',
        'mkv') are kept.

    shuffle : int, optional
        which shuffle to use

    trainingsetindex : int. optional
        which training fraction to use, identified by its index

    track_method: str, optional
        track method from which tracklets are sampled

    n_tracks: int
        number of tracks to be formed in the videos.
        TODO: handling videos with different number of tracks

    n_triplets: (optional) int
        number of triplets to be mined from the videos

    train_epochs: (optional), int
        number of epochs to train the transformer

    train_frac: (optional), fraction
        fraction of triplets used for training/testing of the transformer

    Examples
    --------

    Training model for one video based on ellipse-tracker derived tracklets
    >>> config = "/home/users/.../dlc-project-2025-01-01/config.yaml"
    >>> videos = ['/home/alex/video.mp4']
    >>> deeplabcut.transformer_reID(config, videos, shuffle=1, track_method="ellipse")
    >>> deeplabcut.create_labeled_video(
    >>>     config,
    >>>     videos,
    >>>     shuffle=1,
    >>>     track_method="transformer",
    >>> )
    --------

    """
    import deeplabcut
    import os
    from deeplabcut.utils import auxiliaryfunctions

    # calling create_tracking_dataset, train_tracking_transformer, stitch_tracklets

    cfg = auxiliaryfunctions.read_config(config)

    DLCscorer, _ = deeplabcut.utils.auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle=shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    deeplabcut.compat.create_tracking_dataset(
        config,
        videos,
        track_method,
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        n_triplets=n_triplets,
        destfolder=destfolder,
    )

    (
        trainposeconfigfile,
        testposeconfigfile,
        snapshotfolder,
    ) = deeplabcut.return_train_network_path(
        config,
        shuffle=shuffle,
        modelprefix=modelprefix,
        trainingsetindex=trainingsetindex,
    )

    deeplabcut.pose_tracking_pytorch.train_tracking_transformer(
        config,
        DLCscorer,
        videos,
        videotype=videotype,
        train_frac=train_frac,
        modelprefix=modelprefix,
        train_epochs=train_epochs,
        ckpt_folder=snapshotfolder,
        destfolder=destfolder,
    )

    transformer_checkpoint = os.path.join(
        snapshotfolder, f"dlc_transreid_{train_epochs}.pth"
    )

    if not os.path.exists(transformer_checkpoint):
        raise FileNotFoundError(f"checkpoint {transformer_checkpoint} not found")

    deeplabcut.stitch_tracklets(
        config,
        videos,
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        track_method=track_method,
        modelprefix=modelprefix,
        n_tracks=n_tracks,
        transformer_checkpoint=transformer_checkpoint,
        destfolder=destfolder,
    )
