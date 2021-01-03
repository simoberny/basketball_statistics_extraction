# Basketball game tracking
Extraction of high level statistics by tracking Basketball game using MaskRCNN, SORT, CSRT and interpolation
This files should be inserted in the Maskrcnn samples folder 

### Players Tracking
![gif tracking](https://github.com/simoberny/basket_tracking/blob/master/data/sort.gif)

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

#### Tracking
Given the previous detection output, it tries to track them using CSRT.
```
python tracking.py --video=/path/to/video --det=/path/to/detections.txt
```

#### Interpolation
Try to fill all the frame without a detection, interpolating tracking and detection infos
```
python interpolation.py --video=/path/to/video --det=/path/to/tracking.txt
```

#### Utility
Includes some useful functions, like a converter from mot to dict. 





