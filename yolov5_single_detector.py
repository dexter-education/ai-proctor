# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

from pathlib import Path

import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device

torch.cuda.empty_cache()

class infer_single:

    def __init__(self,
                 weights,
                 imgsz=640,  # inference size (pixels)
                 half=False  # use FP16 half-precision inference
                 ):
        # Load model
        self.half = half
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Run inference
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))  # warmup

    @torch.no_grad()
    def detect(self, img, conf=0.25):

        img = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, conf, 0.45, None, False, max_det=100)
        labels_dict = {0: 0, 76: 0, 72: 0} # person, movile phone and laptop
        for det in pred:  # per image
            if len(det):
                # Print results
                for c in det[:, -1].unique():
                    if c in labels_dict.keys():
                        n = (det[:, -1] == c).sum()  # detections per class
                        labels_dict[c] = n
        
        return list(labels_dict.values())