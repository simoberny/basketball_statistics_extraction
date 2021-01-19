# Basketball game tracking
Extraction of high level statistics by tracking Basketball game using MaskRCNN, SORT, CSRT and interpolation

### Ball Tracking
![gif tracking](https://github.com/simoberny/basket_tracking/blob/master/data/ball_track.gif)

## TODO
1. All phase merged with realtime statistics

## Instructions
1. This files need to be placed in a "project folder" inside samples folder of MaskRCNN.
   -  Something like: *MASKRCNN FOLDER/samples/{custom}/files*
2. Follow instructions on Maskrcnn repository to install dependencies and maskrcnn itself
3. Run the script as indicated below!

## Files explanation
All the script read and save in MOT challenge format

```
frame_id, bbox_id, x_pos, y_pos, width, height, score, other_param (default -1), other_param (default -1), other_param (default -1)
```

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

#### Ball Detection + Tracking
Chained and merged detection phase plus tracking
```
python realtime_ball.py -d --video=/path/to/video --weight=[path to your new .h5 weights]
```

#### Ball Detection + Tracking and Player Extration all-in-one (no stat for now)
Chained and merged ball detection phase and player extraction
```
python realtime.py -d --video=/path/to/video --weight=[path to your new .h5 weights]
```

#### Utility
Includes some useful functions, like a converter from mot to dict. 





