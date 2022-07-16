#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   visualize.py
@Time    :   2022/7/14 02:28
@Author  :   Songnan Lin, Ye Ma
@Contact :   songnan.lin@ntu.edu.sg, my17@tsinghua.org.cn
@Note    :   https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
'''


import os.path
import cv2
import numpy as np

def events_to_voxel_grid(events, num_bins, width, height):
    """
    https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array (np.int32) containing one event per row in the form:
        [timestamp(us), x, y, polarity(0 or 1)]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    ts = events[:, 0].astype(np.float32)
    ts = (num_bins - 1) * (ts - first_stamp) / deltaT
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3].astype(np.float32)
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def visual_voxel_grid(voxel_grid, output_folder, filename_key):
    for i in range(voxel_grid.shape[0]):
        path = os.path.join(output_folder, '%s_%02d.png' % (filename_key, i))
        normalize_im = (voxel_grid[i] - np.min(voxel_grid[i])) / (np.max(voxel_grid[i]) - np.min(voxel_grid[i]))
        cv2.imwrite(path, (normalize_im * 255).astype(np.uint8))