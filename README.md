# Basketball game tracking
Extraction of high level statistics by tracking Basketball game using MaskRCNN, CSRT and interpolation

### Game tracking
![gif tracking](https://github.com/simoberny/basket_tracking/blob/master/data/game_track.gif)

[Check out full video!](https://youtu.be/R6vTXeZziyA)

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

## Instructions
1. This files need to be placed in a "project folder" inside samples folder of MaskRCNN.
   -  Something like: *MASKRCNN FOLDER/samples/{custom}/files*
2. Follow instructions on Maskrcnn repository to install dependencies and maskrcnn itself
3. Run the script as indicated below!

### Detection's save file format
All the script read and save in MOT challenge format
```
frame_id, bbox_id, x_pos, y_pos, width, height, score, x, y, z
```

## Files explanation
#### Training
Dataset folder should contain train and val folder. Follow MaskRCNN rules. 
```
python train.py --weight=[coco|last|imagenet] --dataset=/path/to/dataset
```

#### Detection
Return a txt with all the detection in mot format (det folder) and a video with the bboxes (output folder)
```
python detection.py detect --weight=[coco or path to your new .h5 weights] --video=/path/to/video
```

#### Player Detection
Extract dominant color from player bboxes masks to indentify the team
```
python detection.py detect --weight=coco --video=/path/to/video
```

#### Tracking and Interpolation
Given the previous detection output, it tries to track them using CSRT.
```
python tracking.py --video=/path/to/video --det=/path/to/detections.txt
```

Try to fill all the frame without a detection, interpolating tracking and detection infos
```
python interpolation.py --video=/path/to/video --det=/path/to/tracking.txt
```

#### Extract Statistics
Extract some base statistic, ball possession and ball position in the two half of the pitch
```
python stats.py --video=/path/to/video --det_ball=/path/to/ball_tracking.txt --det_player=/path/to/det.txt
```

#### Ball Detection + Tracking and Player Extration all-in-one
Chained and merged ball detection phase, player extraction and statistics
```
python realtime.py -d --video=/path/to/video --weight=[path to your new .h5 weights]
```

#### Network evaluation
Calculate mAP and FPS 
```
python evaluate.py --dataset=/path/to/dataset/ --weights=/path/to/new_weights
```

#### Utility
Includes some useful functions, like a converter from mot to dict. 





