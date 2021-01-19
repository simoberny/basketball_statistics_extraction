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
#from google.colab.patches import cv2_imshow
from imutils.video import FPS
from tqdm import tqdm
import math

from utility.utility import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

def interpolation(detection_path):
    stat = open("stats/stat.txt", "a")

    #Convert file detection to dictionary
    gt_dict = get_dict(detection_path)

    inter_frame = 0
    i = 0
    nt = 0

    while (i in gt_dict): 
        track, _, _, _ = get_gt(i, gt_dict)

        if track == []:
            nt+=1

        #Check if there is hole of max 50 frames
        if track != [] and nt > 0 and nt < 50:
            print("NT: {}, index ora: {}".format(nt, i))

            before_index = i-(nt+1)

            if before_index > 0: 
                before = gt_dict[before_index][0]['coords']
                after = gt_dict[i][0]['coords']
                
                space_x = (after[0] - before[0])/(nt+1)
                space_y = (after[1] - before[1])/(nt+1)

                eucl = math.sqrt((after[0] - before[0]) ** 2 + (after[1] - before[1]) ** 2)

                if eucl > 800: 
                    nt = 0
                    continue

                print("Space x: {}, before: {}, after: {}".format(space_x, before[0], after[0]))

                for k in range(nt):
                    new_pos_x = before[0] + space_x * (k + 1)
                    new_pos_y = before[1] + space_y * (k + 1)

                    gt_dict[before_index + (k + 1)].append({'coords': [new_pos_x, new_pos_y, before[2], before[3]],'conf':1, 'ids':-1})
                    inter_frame+=1

            nt = 0

        i+=1

    jso = json.dumps(gt_dict)
    #dic_add.write(jso)

    stat.write("\n Contributo interpolazione: {}".format(inter_frame))
    stat.close()

    save_mot(dic=gt_dict, txt="det/det_interpolation.txt")

    return gt_dict
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--video', required=True,
                        metavar="path or URL to video",
                        help='Video to apply the tracking on')
    parser.add_argument('--det', required=False,
                        default="det/det_track_maskrcnn.txt",
                        metavar="/path/to/balloon/dataset/",
                        help='Path to detections file')
    args = parser.parse_args()

    print("Detections: ", args.det)

    dict_finale = interpolation(args.det)
    save_to_video(dict_finale, args.video, 'output/det_track_inter.mp4')
