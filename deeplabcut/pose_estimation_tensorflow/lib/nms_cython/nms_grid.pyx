# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

np.import_array()

cdef extern from "nms_scoremap.hxx":
  vector[int] nms_grid_cpp_test(const float* img, int H, int W)
  vector[int] nms_grid_cpp(const float *unProbValues, size_t height, size_t width,
                           unsigned char *gridData, int grid_height, int grid_width,
                           double nms_prob_thresh)

@cython.boundscheck(False)
@cython.wraparound(False)
def nms_grid(np.ndarray[np.float32_t, ndim=2, mode="c"] scoremap, np.ndarray[np.uint8_t, ndim=2, mode="c"] grid, prob_thresh):
  cdef int W, H
  W, H = scoremap.shape[1], scoremap.shape[0]

  cdef int grid_W, grid_H
  grid_W, grid_H = grid.shape[1], grid.shape[0]

  cdef double prob_thresh_c = prob_thresh
  
  cdef vector[int] v = nms_grid_cpp(&scoremap[0, 0], H, W, &grid[0, 0], grid_H, grid_W, prob_thresh_c)

  return v
