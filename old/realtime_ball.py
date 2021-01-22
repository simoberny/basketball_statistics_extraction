import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random
import itertools
import colorsys
import cv2
from time import sleep
from tqdm import tqdm
import math
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
prev_det = [0,0]

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class BasketConfig(Config):
    # Give the configuration a recognizable name
    NAME = "basket"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + basketball

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 150

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.90
    BACKBONE = 'resnet50'

# define random colors
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors
 
#apply mask to image
def apply_mask(image, mask, color, alpha=0.7):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask == 1, image[:, :, n] * (1-alpha) + alpha * c, image[:, :, n])
    
    return image

#take the image and apply the mask, box, and Label
def ball_instances(count, image, boxes, masks, ids, names, scores, resize):
    f = open("det/det_maskrcnn.txt", "a")

    #Finetuning of the ball detection to avoid outsiders
    min_ball_size = 30
    max_ball_size = 1500

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    best_index = -1
    best_score = 0

    dict_result = []

    if not n_instances:
        return image, []
        #print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
        
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None

        width = x2 - x1
        height = y2 - y1

        area = width * height

        if score > 0.85: 
            label = names[ids[i]]
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]
            image = apply_mask(image, mask, (0,255,0))
            #image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            #image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)

        if score > 0.92 and min_ball_size < area < max_ball_size:
            if score > best_score: 
                best_score = score
                best_index = i

    if best_index >= 0:
        y1, x1, y2, x2 = boxes[best_index]
        label = names[ids[best_index]]
        caption = '{} {:.2f}'.format(label, score) if best_score else label
        mask = masks[:, :, best_index]
        image = apply_mask(image, mask, (255,0,0))
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0,0), 10)
        #image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)

        dict_result = [x1*resize, y1*resize, (x2 - x1)*resize, (y2 - y1)*resize, best_score]

        f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(count, x1*resize, y1*resize, (x2 - x1)*resize, (y2 - y1)*resize, best_score))

    f.close()

    return image, dict_result

def ball_tracking(ball_model, video_path, txt_path="det/det_track_maskrcnn.txt", resize=2, display = False):
    # Different classes for different train
    ball_class = ['BG', 'basketball']

    f = open(txt_path, "w")

    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    length_input = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Totale frame: {}".format(length_input))

    # Define codec and create video writer
    file_name = "output/full_detection_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (int(width/resize), int(height/resize)))
    
    count = 0
    success = True

    tracker = cv2.TrackerCSRT_create()

    initBB = None
    bbox_offset = 10
    prev_box = [0, 0]

    frame_diff = 0

    start = time.time()

    with tqdm(total=length_input, file=sys.stdout) as pbar:
        while success:
            success, image = vcapture.read()
            if success:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                c_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                #Reduce computing impact
                image = cv2.resize(image, (int(width/resize), int(height/resize)))

                ball_ret = ball_model.detect([image], verbose=0)[0]

                # Process ball
                frame, detection = ball_instances(count, image, ball_ret["rois"], ball_ret["masks"], ball_ret["class_ids"], ball_class, ball_ret["scores"], resize)

                #If no bbox initialized
                if detection != [] and (initBB is None or count % 1 == 0):
                    min_distance = 99999
                    
                    coor = np.array(detection[:4], dtype=np.int32)
                    initBB = (coor[0] - bbox_offset, coor[1] - bbox_offset, coor[2] + 2*bbox_offset, coor[3] + 2*bbox_offset)
                    eucl = math.sqrt((coor[0] - prev_box[0]) ** 2 + (coor[1] - prev_box[1]) ** 2)  

                    if (prev_box[0] == 0 and prev_box[1] == 0) or initBB is None:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(c_image, initBB)
                    elif (frame_diff > 8 or eucl < 150): 
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(c_image, initBB)

                        frame_diff = 0        
                    else:
                        frame_diff += 1
                        cv2.putText(frame, "Frame without valid det: {}".format(frame_diff), (16, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                
                # If there is a new bbox, update the tracker
                if initBB is not None: 
                    (success, tracked_box) = tracker.update(c_image)
                    #(success, tracked_boxes) = trackers.update(frame)

                    if success:
                        #Save tracking boxes (include also det)
                        f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(count, tracked_box[0], tracked_box[1], tracked_box[2], tracked_box[3], 1))

                        p1 = (int(tracked_box[0]/resize), int(tracked_box[1]/resize))
                        p2 = (int(tracked_box[0]/resize + tracked_box[2]/resize), int(tracked_box[1]/resize + tracked_box[3]/resize))
                        cv2.rectangle(frame, p1, p2, (255,0,100), 6, 3)

                        prev_box = [tracked_box[0], tracked_box[1]]
                    else: 
                        initBB = None

                # RGB -> BGR to save image to video
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)              

                end = time.time()
                frame_time = end - start

                fps = round(count / frame_time, 2)

                frame = cv2.rectangle(frame, (50,50), (200, 100), (100,100,100), -1)
                frame = cv2.putText(frame, "{} FPS".format(fps), (80, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (230,230,230), 2)

                if display:
                    #toshow = cv2.resize(frame, (int(width/3), int(height/3)))

                    cv2.imshow('MaskRCNN Ball Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Add image to video writer
                vwriter.write(frame)
                count += 1

            #Fancy print
            pbar.update(1)
            sleep(0.1)
            #print("FPS: {}".format(length_input/(end-start)))

    f.close()    
    vwriter.release()

    print("Saved to ", file_name)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--video', required=True,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('-d', '--display', required=False, action='store_true')
    args = parser.parse_args()

    class InferenceConfig(BasketConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    # Create model for ball
    ball_model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Assign the two weights (coco for the player)
    ball_weight = args.weights

    # Assign the weights to the model
    ball_model.load_weights(ball_weight, by_name=True)

    # TODO detection player and ball
    ball_tracking(ball_model, video_path=args.video, display=args.display)