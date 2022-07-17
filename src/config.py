#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/7/12 18:17
@Author  :   Songnan Lin, Ye Ma
@Contact :   songnan.lin@ntu.edu.sg, my17@tsinghua.org.cn
@Note    :   
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
'''

from easydict import EasyDict as edict

__C     = edict()
cfg     = __C

# SENSOR
__C.SENSOR                              = edict()
__C.SENSOR.CAMERA_TYPE = 'DVS346'

__C.SENSOR.K = None
if cfg.SENSOR.CAMERA_TYPE == 'DVS346':
    __C.SENSOR.K = [0.00018 * 29250, 20, 0.0001, 1e-7, 5e-9, 0.00001]
elif cfg.SENSOR.CAMERA_TYPE == 'DVS240':
    __C.SENSOR.K = [0.000094 * 47065, 23, 0.0002, 1e-7, 5e-8, 0.00001]


# Directories
__C.DIR                                 = edict()
__C.DIR.IN_PATH = 'data_samples/interp/'
__C.DIR.OUT_PATH = 'data_samples/output/'


# Visualize
__C.Visual                              = edict()
__C.Visual.FRAME_STEP = 5
