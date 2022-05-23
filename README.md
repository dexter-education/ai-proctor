# dexter-ai-proctor

## Introduction

Library to generate flags when running proctoring tests. The current version supports:

1. Face counting and face center detection tests.
2. Person, mobile and laptop counting tests.
3. Mouth open or mouth hidden tests.
4. Head pose estimation with yaw, pitch, and roll.


## Installation

The library can be installed via pip. Look for the latest version in the `dist` folder and install the wheel file as:

```
pip install dexter-ai-proctor-0.2.0-py3-none-any.whl
```
    
Currently, install requires were not added to the setup.py so the requirements will need to be installed from the requirements.txt file. This will be sorted out in a later release after removing unneeded dependencies.

## Usage
 
Below is an example of how to use the library with a video.

```
import cv2
from dexter_ai_proctor import main_runner

obj = main_runner.runner(mtcnn='mtcnn',
    yolov5_model=<yololov5_model_path>, 
    face_seg_model=<face_model_path>, 
    head_pose_model=<head_pose_model_path>)

# yolov5_model_path = 'yolov5s.pt'
# face_model_path = 'face.pth'
# head_pose_model_path = 'shuff_epoch_120.pkl'

cap = cv2.VideoCapture('video.mp4')
while True:
    ret, img = cap.read()
    if ret:
        obj.run_mtcnn(img)
        obj.run_yolov5(img)
        obj.run_face_seg(img)
        obj.run_head_pose(img)
        obj.check_counts()
    else:
        break
```

Be careful when passing all models together when creating object, as all will be loaded into the memory causing the program to crash. If a path to a model is not passed it is not initialized and subsequently won't be run. The head pose model by default will also initialize the mtcnn model as it requires face coordinates.

The flags will be printed on the terminal.
