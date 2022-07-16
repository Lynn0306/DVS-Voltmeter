#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/7/12 17:30
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

import argparse
import os
import numpy as np
import cv2
import tqdm
from simulation_scripts.config import cfg
from simulation_scripts.simulator import EventSim
from simulation_scripts.visualize import events_to_voxel_grid, visual_voxel_grid


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--camera_type', type=str, help='Camera type, such as DVS346', default='DVS346')
    parser.add_argument('--model_para', type=float, nargs='+', help='Set parameters for a specific camera type', default=None)
    parser.add_argument('--input_dir', type=str, help='Set dataset root_path', default=None)
    parser.add_argument('--output_dir', type=str, help='Set output path', default=None)
    args = parser.parse_args()
    return args

def is_valid_dir(dirs):
    return os.path.exists(os.path.join(dirs, 'info.txt'))

def process_dir(cfg, file_info, video_name):
    indir = os.path.join(cfg.DIR.IN_PATH, video_name)
    outdir = os.path.join(cfg.DIR.OUT_PATH, video_name)
    print(f"Processing folder {indir}... Generating events in file {outdir}")

    # file info
    file_timestamps_us = [int(info_i.split()[0]) for info_i in file_info]
    file_paths = [info_i.split()[1] for info_i in file_info]

    # set simulator
    sim = EventSim(cfg=cfg, output_folder=cfg.DIR.OUT_PATH, video_name=video_name)

    # process
    pbar = tqdm.tqdm(total=len(file_paths))
    num_events, num_on_events, num_off_events = 0, 0, 0
    events = []
    for i in range(0, len(file_paths)):
        timestamp_us = file_timestamps_us[i]
        image = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)

        # event generation!!!
        event = sim.generate_events(image, timestamp_us)

        if event is not None:
            events.append(event)
            num_events += event.shape[0]
            num_on_events += np.sum(event[:, -1] == 1)
            num_off_events += np.sum(event[:, -1] == 0)

        # visualization: save stacked event images
        if (i + 1) % cfg.Visual.FRAME_STEP == 0:
            events = np.concatenate(events, axis=0)
            if events.shape[0] > 0:
                event_tensor = events_to_voxel_grid(np.ascontiguousarray(events, dtype=np.float32),
                                                    num_bins=cfg.Visual.FRAME_STEP,
                                                    width=image.shape[1], height=image.shape[0])
                visual_voxel_grid(event_tensor,
                                  output_folder=os.path.join(cfg.DIR.OUT_PATH, video_name),
                                  filename_key='voxel_%d' % (i + 1))
            events = []

        pbar.set_description(f"Events generated: {num_events}")
        pbar.update(1)
    sim.reset()


if __name__ == "__main__":
    args = get_args_from_command_line()
    cfg.SENSOR.CAMERA_TYPE = args.camera_type if args.camera_type is not None else cfg.SENSOR.CAMERA_TYPE
    cfg.SENSOR.K = args.model_para if args.model_para is not None else cfg.SENSOR.K
    cfg.DIR.IN_PATH = args.input_dir if args.input_dir is not None else cfg.DIR.IN_PATH
    cfg.DIR.OUT_PATH = args.output_dir if args.output_dir is not None else cfg.DIR.OUT_PATH
    if cfg.SENSOR.K is None or len(cfg.SENSOR.K) != 6:
        raise Exception('No model parameters given for sensor type %s' % cfg.SENSOR.CAMERA_TYPE)
    print(cfg)

    video_list = sorted(os.listdir(cfg.DIR.IN_PATH))
    for video_i in video_list:
        video_i_path = os.path.join(cfg.DIR.IN_PATH, video_i)
        os.makedirs(os.path.join(cfg.DIR.OUT_PATH, video_i), exist_ok=True)

        if is_valid_dir(video_i_path):
            # video info
            with open(os.path.join(cfg.DIR.IN_PATH, video_i, 'info.txt'), 'r') as f:
                video_info = f.readlines()
            # simulation
            process_dir(cfg=cfg, file_info=video_info, video_name=video_i)