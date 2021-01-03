import os
import sys
import json
import datetime
import numpy as np
import random
import cv2
import colorsys
import skimage.io
from time import sleep
from google.colab.patches import cv2_imshow
from imutils.video import FPS
from tqdm import tqdm


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

def get_mask(filename):
	mask = cv2.imread(filename,0)
	mask = mask / 255.0
	return mask

# Compute background subtraction
def bg_subtraction(video_path):
    bg = cv2.imread('bg.jpg')
    video = cv2.VideoCapture(video_path)
    length_input = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    #Output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('difference.avi',fourcc, 30.0, (int(video.get(3)),int(video.get(4))))
    
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    frame_id = 0
    ret = True

    mask = get_mask('roi_mask.jpg')
    mask = np.expand_dims(mask,2)
    mask = np.repeat(mask,3,2)

    _, bg = video.read()

    with tqdm(total=length_input, file=sys.stdout) as pbar:
        while ret:
            ret, frame = video.read()

            if not ret:
                continue

            #Apply the basket field mask
            frame = frame * mask
            frame = frame.astype(np.uint8)

            #Compute the bg subtraction
            difference = cv2.absdiff(bg, frame)

            out.write(difference)

            pbar.update(1)
            sleep(0.01)

    out.release()

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--video', required=True,
                        metavar="path or URL to video",
                        help='Video to apply the tracking on')
    args = parser.parse_args()

    print("Video: ", args.video)

    bg_subtraction(args.video)
