import cv2
from tqdm import tqdm
import os
import sys

#My own library with utils functions
from utility.utility import *

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--det', required=True,
                        metavar="/path/to/balloon/dataset/",
                        help='Path to detections file')
    parser.add_argument('--video', required=True,
                        metavar="path or URL to video",
                        help='Video to apply the tracking on')
    parser.add_argument('--output', required=True,
                        metavar="Name of output video",
                        help='Video to apply the tracking on')
    args = parser.parse_args()

    #Convert file detection to dictionary
    gt_dict = get_dict(args.det)
    save_to_video(gt_dict, args.video, "output/{}".format(args.output))