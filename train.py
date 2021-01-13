import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage
import os

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '../../'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

############################################################
#  Configurations
############################################################

class CustomConfig(Config):
    NAME = "objects"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + person + ball

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.8
    BACKBONE = 'resnet50'

class CustomDataset(utils.Dataset):
  def load_custom(self, dataset_dir, subset):
    #Class we want to train
    self.add_class("objects", 1, "basketball")

    # Train or validation dataset?
    assert subset in ["train", "val"]
    dataset_dir = os.path.join(dataset_dir, subset)

    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Add images
    for a in annotations:
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. These are stores in the
        # shape_attributes (see json format above)
        # The if condition is needed to support VIA versions 1.x and 2.x.
        if type(a['regions']) is dict:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in a['regions']] 

        image_path = os.path.join(dataset_dir, a['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        self.add_image(
            "objects",
            image_id=a['filename'],  # use file name as a unique image id
            path=image_path,
            width=width, height=height,
            polygons=polygons)

  def load_mask(self, image_id):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a balloon dataset image, delegate to parent class.
    image_info = self.image_info[image_id]
    if image_info["source"] != "objects":
        return super(self.__class__, self).load_mask(image_id)

    #[(x,y) of center, radius]
    info = self.image_info[image_id]
    mask = np.zeros([ info["height"], info["width"], len(info["polygons"])],
                    dtype=np.uint8)

    for i, p in enumerate(info["polygons"]):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc, i] = 1

    return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

  def image_reference(self, image_id):
      """Return the path of the image."""
      info = self.image_info[image_id]
      if info["source"] == "objects":
          return info["path"]
      else:
          super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################

def train(model, dataset, epoch):
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    start_train = time.time()
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epoch,
                layers='heads')
    end_train = time.time()

    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--weight', required=True,
                        metavar="name of the weight",
                        help='Name of the weights')
    parser.add_argument('--dataset', required=True,
                        metavar="path to dataset url",
                        help='Dataset should contain train and val')
    parser.add_argument('--epoch', type=int, required=False,
                        metavar="number of epoch",
                        default=100,
                        help='Epoch should be higher than 50')
    args = parser.parse_args()

    #Load configuration
    config = CustomConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)

    dataset = args.dataset

    #Weight to start with
    init_with = args.weight  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    #Start training
    train(model, dataset, int(args.epoch))