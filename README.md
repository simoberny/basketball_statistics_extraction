# Basketball game statistics extraction
Extraction of high level statistics by tracking Basketball game using MaskRCNN, CSRT and interpolation.

YOLO Version of this project: [Matteo's Repository](https://github.com/MatteoDalponte/Basketball_statistics)

### Game tracking
![gif tracking](https://github.com/simoberny/basket_tracking/blob/master/data/game_track.gif)

[Check out full video!](https://youtu.be/R6vTXeZziyA)

## Results
#### Ball Accuracy
| Running | Accuracy | Precision  | Recall | mAP (%) |
| ------------- |:-------------:| :--------: | :-----:| :-----:|
| MaskRCNN  | 0.83  | 0.99    | 0.77   | 87% |
| MaskRCNN + Track  | 0.90  | 0.89    | 0.99  | - |
| YOLOv3   | 0.73    | 0.99      | 0.65       | - |
| YOLOv3 + Track  | 0.89  | 0.91    | 0.94   | - |

#### Times
| Running | Detection | Det + Track  | Real-time |
| ------------- |:-------------:| :--------: | :-----:|
| MaskRCNN (CPU) | 0.8 FPS   | 0.5 FPS      | 0.2 FPS   |
| MaskRCNN (GPU**) | 4 FPS     | 3 FPS        | 0.7+ FPS  |
| YOLOv3 (GPU)   | 12 FPS    | @10 FPS      | -         |

###### ** GPU = RTX 2070

## Instructions
1. This files need to be placed in a "project folder" inside samples folder of [MaskRCNN Matterport implementation](https://github.com/matterport/Mask_RCNN).
   -  Something like: *MASKRCNN FOLDER/samples/{project folter}/files*
2. Follow instructions of original Maskrcnn repository to install dependencies and maskrcnn itself
3. Run the script as indicated below!

#### Detection's save file format
All the script read and save in MOT challenge format
```
frame_id, bbox_id, x_pos, y_pos, width, height, score, x, y, z
```

## Files explanation
<img src="https://i.imgur.com/FuHGDAZ.png" width="550">

##### Training
Dataset folder should contain train and val folder. Follow MaskRCNN rules. 
```
python train.py --weight=[coco|last|imagenet] --dataset=/path/to/dataset
```
___

### Ball Analysis
##### Detection
Return a txt with all the detection in mot format (det folder) and a video with the bboxes (output folder)
```
python detection.py detect --weight=[coco or path to your new .h5 weights] --video=/path/to/video
```

##### Tracking and Interpolation
Given the previous detection output, it tries to track them using CSRT.
```
python tracking.py --video=/path/to/video --det=/path/to/detections.txt
```

Try to fill all the frame without a detection, interpolating tracking and detection infos
```
python interpolation.py --video=/path/to/video --det=/path/to/tracking.txt
```

___

### Player Analysis
##### Player Detection
Extract dominant color from player bboxes masks to indentify the team
```
python detection.py detect --weight=coco --video=/path/to/video
```

___

### Statistics Extraction
##### Extract Statistics
Extract some base statistic, ball possession and ball position in the two half of the pitch
```
python stats.py --video=/path/to/video --det_ball=/path/to/ball_tracking.txt --det_player=/path/to/det.txt
```

##### Online Computation (Ball + Player detection and stat extration)
Chained and merged ball detection phase, player extraction and statistics
```
python realtime.py -d --video=/path/to/video --weight=[path to your new .h5 weights]
```
___

##### Network evaluation
Calculate mAP and FPS 
```
python evaluate.py --dataset=/path/to/dataset/ --weights=/path/to/new_weights
```


## Ball Network training fine-tunes
- Data Agugmentation using imgaug ([Github repo](https://github.com/aleju/imgaug))
- Reduced **RPN_ANCHOR_SCALES** for better recognition of small objects
- Increased **WEIGHTS_DECAY**
- Custom training steps: 

```
   # Training network Heads
   model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=int(epochs/2),
               augmentation=augmentation,
               layers='heads')

   # Training network 4+ layers
   model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=epochs,
               layers='4+')
    
   # Training all network layers
   model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE/10,
               epochs=int(epochs*1.2),
               layers='all')
```
