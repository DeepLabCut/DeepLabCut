# Copyright 2019 by
# Tabet Ehsainieh, ehsainit@informatik.uni-freiburg.de
# All rights reserved.

import random

import cv2


def get_random_color(n):
    ''' generate rgb using a list comprehension '''
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
    return r, g, b


def bbox(frame, track):
    x = track.prediction[0]
    y = track.prediction[1]
    return cv2.rectangle(frame, (int(x + track.width + 1), int(y + track.height + 1)),
                         (int(x - track.width + 1), int(y - track.height + 1)), track.color, 2)
