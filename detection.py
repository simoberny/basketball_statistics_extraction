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

    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # Background + basketball

    STEPS_PER_EPOCH = 175
    DETECTION_MIN_CONFIDENCE = 0.90
    BACKBONE = 'resnet50'
    DETECTION_NMS_THRESHOLD = 0.2
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    WEIGHT_DECAY = 0.005

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

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
def display_instances(count, image, boxes, masks, ids, names, scores, resize):
    f = open("det/det_maskrcnn.txt", "a")

    #Finetuning of the ball detection to avoid outsiders
    min_ball_size = 10
    max_ball_size = 1750

    det_ok = 0

    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    best_index = -1
    best_score = 0

    if not n_instances:
        return image, 0
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
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            #image = cv2.putText(image, caption, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if score > 0.92 and min_ball_size < area < max_ball_size:
            if label == 'basketball' or label == 'sports ball':
                det_ok += 1

            if score > best_score: 
                best_score = score
                best_index = i
    
    if best_index >= 0:
        y1, x1, y2, x2 = boxes[best_index]
        label = names[ids[best_index]]
        caption = '{} {:.2f}'.format(label, score) if best_score else label
        mask = masks[:, :, best_index]
        image = apply_mask(image, mask, (255,0,0))
        image = cv2.rectangle(image, (x1 - 6, y1 -6), (x2 + 12, y2 +12), (255, 0,0), 8)

        (t_width, t_height), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)

        image = cv2.putText(image, caption, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 0, 0), 2)
        f.write('{},-1,{},{},{},{},{},-1,-1,-1\n'.format(count, x1*resize, y1*resize, (x2 - x1)*resize, (y2 - y1)*resize, best_score))

    f.close()

    return image, int(det_ok > 0)

#take the image and apply the mask, box, and Label
def display_instances_image(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        return image
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

        if score > 0.75: 
            label = names[ids[i]]
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]
            image = apply_mask(image, mask, (0,255,0))
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 10)
            image = cv2.putText(image, caption, (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 2)

    return image

def image_segmentation(model, class_names, image_path):
    image = skimage.io.imread(image_path)
    r = model.detect([image], verbose=0)[0]

    frame = display_instances_image(image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"])

    file_name = "detection_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, frame)

    print("Saved to ", file_name)

def video_segmentation(model, class_names, video_path, txt_path="det/det_maskrcnn.txt", resize=1, display=False):
    start = time.time()

    stat = open("stats/stat.txt", "a")
    f = open(txt_path, "w").close()
    
    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    length_input = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Totale frame: {}".format(length_input))

    # Define codec and create video writer
    file_name = "output/detection_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (int(width/resize), int(height/resize)))
    
    count = 0
    success = True
    total_det = 0

    start = time.time()
    with tqdm(total=length_input, file=sys.stdout) as pbar:
        while success:
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                #Reduce computing impact
                image = cv2.resize(image, (int(width/resize), int(height/resize)))

                # Detect objects
                r = model.detect([image], verbose=0)[0]

                frame, st = display_instances(count, image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"], resize)

                # RGB -> BGR to save image to video
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if display:
                    toshow = cv2.resize(frame, (int(width/3), int(height/3)))

                    cv2.imshow('YOLO Object Detection', toshow)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Add image to video writer
                vwriter.write(frame)
                count += 1
                total_det += st

                # FPS calculation
                end = time.time()
                frame_time = end - start

                d_fps = round(count / frame_time, 2)

                print("FPS: {}".format(d_fps))
                
            #Fancy print
            pbar.update(1)
            sleep(0.1)

    vwriter.release()

    stat.write("\n---- Statistiche Detection ---- \n")
    stat.write("Numero tatale frame: {}\n".format(count))
    stat.write("Numero tatale frame con posizione individuata: {}\n".format(total_det))

    stat.close()

    end = time.time()
    print("Saved to ", file_name)

    print("Detections time: ", end-start)
    print("FPS: {}".format(length_input/(end-start)))
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('-d', '--display', required=False, action='store_true')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    class InferenceConfig(BasketConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights


    class_names = ['BG', 'basketball']

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        #Coco labels
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        #class_names = ['BG', 'basketball', 'person']
        class_names = ['BG', 'basketball']
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "detect":
        if args.video:
            video_segmentation(model, class_names, video_path=args.video, display=args.display, resize=1)
        else: 
            image_segmentation(model, class_names, args.image)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))