import argparse
from pathlib import Path


def parse_opts_prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_or_frames_path', type=Path,
                        help='Input data folder. Could contain video or frames. '
                             'N.B: the structure of the folder should not be the same')
    parser.add_argument('--save_npy_rgb_path', type=Path, help='Output dir for RGB npy arrays')
    parser.add_argument('--save_npy_flow_path', type=Path, help='Output dir for OPTICAL FLOW npy arrays')
    parser.add_argument('--type_prediction', default='joint', type=str,
                        help='The type of prediction. Can be one of: (joint, rgb, flow)')
    parser.add_argument('--use_frames', action='store_true',
                        help='If used, frames will be read instead of videos.')

    args = parser.parse_args()

    return args
