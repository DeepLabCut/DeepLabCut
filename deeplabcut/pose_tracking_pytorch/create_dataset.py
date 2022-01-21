import numpy as np
import os
from deeplabcut.refine_training_dataset.stitch import (
    TrackletStitcher,
)
from pathlib import Path
from mmappickle import mmapdict
from .tracking_utils.preprocessing import query_feature_by_coord_in_img_space

np.random.seed(0)


def generate_train_triplets_from_pickle(path_to_track, n_triplets=1000):
    ts = TrackletStitcher.from_pickle(path_to_track, 3)
    triplets = ts.mine(n_triplets)
    assert len(triplets) == n_triplets
    return triplets


def save_train_triplets(feature_fname, triplets, out_name):
    ret_vecs = []

    feature_dict = mmapdict(feature_fname, True)

    nframes = len(feature_dict.keys())

    zfill_width = int(np.ceil(np.log10(nframes)))

    for triplet in triplets:

        anchor, pos, neg = triplet[0], triplet[1], triplet[2]

        anchor_coord, anchor_frame = anchor
        pos_coord, pos_frame = pos
        neg_coord, neg_frame = neg

        # if animal == 'pup' and anchor_coord.shape[0]!=5:
        #    continue

        anchor_frame = "frame" + str(anchor_frame).zfill(zfill_width)
        pos_frame = "frame" + str(pos_frame).zfill(zfill_width)
        neg_frame = "frame" + str(neg_frame).zfill(zfill_width)

        anchor_vec = query_feature_by_coord_in_img_space(
            feature_dict, anchor_frame, anchor_coord
        )
        pos_vec = query_feature_by_coord_in_img_space(
            feature_dict, pos_frame, pos_coord
        )
        neg_vec = query_feature_by_coord_in_img_space(
            feature_dict, neg_frame, neg_coord
        )

        ret_vecs.append([anchor_vec, pos_vec, neg_vec])

    ret_vecs = np.array(ret_vecs)

    with open(out_name, "wb") as f:
        np.save(f, ret_vecs)


def create_train_using_pickle(feature_fname, path_to_pickle, out_name, n_triplets=1000):
    triplets = generate_train_triplets_from_pickle(
        path_to_pickle, n_triplets=n_triplets
    )
    save_train_triplets(feature_fname, triplets, out_name)


def create_triplets_dataset(videos, dlcscorer, track_method, n_triplets=1000):

    # 1) reference to video folder and get the proper bpt_feature file for feature table
    # 2) get either the path to gt or the path to track pickle

    for video in videos:
        vname = Path(video).stem
        videofolder = str(Path(video).parents[0])
        feature_fname = os.path.join(
            videofolder, vname + dlcscorer + "_bpt_features.mmdpickle"
        )
        track_file = os.path.join(videofolder, vname + dlcscorer + f"_{track_method}.pickle")
        out_fname = os.path.join(videofolder, vname + dlcscorer + "_triplet_vector.npy")
        create_train_using_pickle(
            feature_fname, track_file, out_fname, n_triplets=n_triplets
        )
