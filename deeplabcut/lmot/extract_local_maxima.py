# Copyright 2019 by
# Tabet Ehsainieh, ehsainit@informatik.uni-freiburg.de
# All rights reserved.

import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from deeplabcut.lmot.detection import Detection


def extract_locmaxima(scmap, locref, neighborhood_size=5, threshold=0.99999):
    # credit: https://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    data_max = filters.maximum_filter(scmap, neighborhood_size)
    maxima = (scmap == data_max)
    data_min = filters.minimum_filter(scmap, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    possible_objects = ndimage.find_objects(labeled)
    detections = []
    for dy, dx in possible_objects:
        height, width, center = compute_bbox(dy, dx, locref)
        measurement = Detection(center, height, width)
        detections.append(measurement)
    return detections


def compute_bbox(dy, dx, locref):
    startloc = extract_point_from_nn((dy.start, dx.start), locref)
    width = extract_point_from_nn((dy.start, dx.stop), locref) - extract_point_from_nn((dy.start, dx.start), locref)
    height = extract_point_from_nn((dy.stop, dx.start), locref) - extract_point_from_nn((dy.start, dx.start), locref)
    return abs(height[1]), abs(width[0]), startloc


def extract_point_from_nn(loc, locref):
    offset = np.array(locref[loc])[::-1]
    pos_f8 = (np.array(loc).astype('float') * 8.0 + 0.5 * 8.0 + offset)
    pose = np.array([pos_f8[1], pos_f8[0], 0.0, 0.0])
    return pose
