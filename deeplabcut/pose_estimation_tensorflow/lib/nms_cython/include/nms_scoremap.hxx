#include "andres/marray.hxx"

#include <algorithm>

// #include <numpy/arrayobject.h>

#include <vector>

#include <iostream>

// namespace bp = boost::python;

using namespace std;

/**
   MA: unProbValues, scoregrid in "row major" layout
 */

std::vector<int> nms_grid_cpp(const float *unProbValues, size_t height, size_t width,
                              unsigned char *gridData, int grid_height, int grid_width,
                              double nms_prob_thresh)
{

  // cout << "height: " << height << endl;
  // cout << "width: " << width << endl;
    const size_t num_locs = height * width;

    // MA: is this row or column major? 
    std::array<size_t, 2> shape;
    shape[0] = grid_height;
    shape[1] = grid_width;
    andres::View<unsigned char> grid(shape.begin(), shape.end(), gridData); 

    std::vector<int> indices(height*width);
    for(int i = 0; i < num_locs; ++i)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&unProbValues](int a, int b) {
        return unProbValues[a] > unProbValues[b];
    });

    // std::cout << unProbValues[indices[0]] << " " << unProbValues[indices[1]] << " " << unProbValues[indices[2]] << std::endl;
    const int c_x = grid_width/2;
    const int c_y = grid_height/2;

    // std::cout << "width " << grid_width << " c_x " << c_x << std::endl;

    std::vector<int> wasted(num_locs);
    std::vector<int> locations;

    for(int k = 0; k < num_locs; ++k)
    {
        int idx = indices[k];
        if (wasted[idx] == 1)
            continue;

        if (unProbValues[idx] < nms_prob_thresh)
            break;

        wasted[idx] = 1;
	      locations.push_back(idx);

	// MA: numpy arrays are in row major format
        const int ii = idx % width;
        const int jj = idx / width;

        const int start_i = ii - grid_width + c_x + 1;
        const int end_i = start_i + grid_width;
        const int start_j = jj - grid_height + c_y + 1;
        const int end_j = start_j + grid_height;

	//cout << "idx: " << idx << " jj: " << jj << " ii: " << ii << endl ;
        //std::cout << " jj " << jj << " start_j " << start_j << "  end_j " << end_j << std::endl;

        for(int i = start_i; i < end_i; ++i)
        {
            if(i < 0 || i >= width)
                continue;
            int grid_i = i - start_i;
            for(int j = start_j; j < end_j; ++j)
            {
                if(j < 0 || j >= height)
                    continue;
                int grid_j = j - start_j;
                if(grid(grid_j, grid_i) == 1)
                {
		  // MA: numpy arrays are row-major
		  wasted[j*width + i] = 1;
                }
            }
        }
    }

    return locations;
}


std::vector<int> nms_grid_cpp_test(float *unProbValues, size_t height, size_t width)
{
  std::array<size_t, 2> shape;
  shape[0] = height;
  shape[1] = width;
  andres::View<float> grid(shape.begin(), shape.end(), unProbValues); 

  std::cout << "Hello fuck! " << height << " " << width << std::endl;
  std::vector<int> result;
  result.push_back(5);
  result.push_back(3);
  result.push_back(1);
  return result;
}