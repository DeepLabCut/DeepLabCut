import numpy as np


def load_features_from_coord(feature, coords, valid_mask_for_fish=False):
    # locate the right spot from the features
    if valid_mask_for_fish:

        mask = np.array([1, 2, 6])
        coords = coords[mask, :]

    feat_vec = np.zeros((coords.shape[0], coords.shape[1], 2048))

    for animal_idx in range(coords.shape[0]):
        for kpt_idx in range(coords.shape[1]):
            coord = coords[animal_idx][kpt_idx]
            x, y = coord

            vec = feature[y, x, :]
            if np.sum(coord) != 0:
                feat_vec[animal_idx][kpt_idx] = vec

    return feat_vec


def convert_coord_from_img_space_to_feature_space(arr, stride):

    stride = stride * 2 # because there is only one deconv. This will change if there are different numbers of deconv layers    

    arr = np.nan_to_num(arr).astype(np.int64)

    # take care of difference between feature map space and original image space
    
    arr = (arr - (stride//2) ) // stride

    return arr.astype(np.int64)


def query_feature_by_coord_in_img_space(feature_dict, frame_id, ref_coord):

    features = feature_dict[frame_id]["features"]
    coordinates = feature_dict[frame_id]["coordinates"]

    diff = coordinates - ref_coord
    diff[np.where(np.logical_or(diff > 9000, diff < 0))] = np.nan
    match_id = np.argmin(np.nanmean(diff, axis=(1, 2)))

    return features[match_id]
