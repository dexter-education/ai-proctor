import cv2
import numpy as np
from PIL import Image
from math import cos, sin
import torch
import torch.nn.functional as F
from mtcnn.mtcnn import MTCNN
from torchvision.transforms import transforms

from . import config
from .models.common import DetectMultiBackend
from .utils.augmentations import letterbox
from .utils.general import check_img_size, non_max_suppression
from .utils.model_celebmask import BiSeNet
from .utils import stable_hopenetlite
from .utils.torch_utils import select_device

torch.cuda.empty_cache()
config_dict = config.get_config()

class yolov5_infer_single:

    def __init__(self,
                 weights, # path to weights
                 imgsz=640,  # inference size (pixels)
                 half=False  # use FP16 half-precision inference
                 ):
        # Load model
        self.half = half
        self.device = select_device('')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.confidence = config_dict['person']['confidence']

        # Run inference
        self.model.warmup(imgsz=(1, 3, self.imgsz, self.imgsz))  # warmup

    @torch.no_grad()
    def detect(self, img):

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
        pred = non_max_suppression(pred, self.confidence, 0.45, None, False, max_det=100)
        labels_dict = {0: 0, 67: 0, 63: 0} # person, mobile phone and laptop
        for det in pred:  # per image
            if len(det):
                det = det.cpu().numpy()
                # Print results
                for c in np.unique(det[:, -1]):
                    c = int(c)
                    if c in labels_dict.keys():
                        n = (det[:, -1] == c).sum()  # detections per class
                        labels_dict[c] = n
        
        return dict(zip(['person', 'mobile phone', 'laptop'], list(labels_dict.values()))) # list returning counts of person, mobile phone and laptop

class mtcnn_face:
    """Class to count faces in the image and if face is centered or not
    """
    def __init__(self):
        self.model = MTCNN()
        self.confidence = config_dict['face']['confidence']
        self.center = config_dict['face center']['center_ratio']

    def detect(self, img):

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.model.detect_faces(img_rgb)
        face_count = sum([1 if face['confidence'] > self.confidence else 0 for face in res])
        if face_count != 0:
            x0, y0, w1, h1 = res[0]['box']
            x1, y1 = x0 + w1, y0 + h1
            cx = ((x0 + x1)/2) / w
            cy = (y0 + y1)/2 / h
            if self.center < cx < (1 - self.center) and self.center < cy < (1 - self.center):
                face_center = 0
            else:
                face_center = 1
            print(cx, cy)
        
        else:
            x0, y0, x1, y1 = 0, 0, 0, 0
            face_center = 0
        return {'face': face_count, 'face center': face_center}, [x0, y0, x1, y1] # dictionary returning count of faces and if face centered; along with face bounding box
        
class face_segmentation:
    """Class to run face segmentation model
    """
    def __init__(self, weights):
        self.model = BiSeNet(n_classes=19)
        self.confidence = config_dict['mouth open']['confidence']
        self.device = select_device('')
        if torch.cuda.is_available():
            self.model.cuda()
            self.model.load_state_dict(torch.load(weights))
        else:
            self.model.load_state_dict(torch.load(weights, map_location='cpu'))
        self.model.eval()

    def detect(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = transform(img_rgb)
        img_rgb = img_rgb.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(img_rgb)[0]
        mask = out.squeeze(0).cpu().numpy().argmax(0)
        mask = mask.astype(np.uint8)
        mask_mouth = mask.copy()
        mask[mask != 11] = 0
        mouth_pixels = int(np.sum(mask) / 11)
        mouth_open = mouth_pixels > config_dict['mouth open']['pixels_required']
        
        mask_mouth[(mask_mouth < 11) | (mask_mouth > 13)] = 0
        mouth_hidden = np.sum(mask_mouth)/12 < config_dict['mouth hidden']['pixels_required']

        return  {'mouth open': int(mouth_open), 'mouth hidden': int(mouth_hidden)}

class head_pose:
    """Class to run head pose estimation model
    """
    def __init__(self, weights):
        self.model = stable_hopenetlite.shufflenet_v2_x1_0()
        self.device = select_device('')
        if torch.cuda.is_available():
            self.model.cuda()
            self.model.load_state_dict(torch.load(weights), strict=False)
        else:
            self.model.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
        self.model.eval()
        self.confidence = config_dict['yaw']['confidence']
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def detect(self, img, face):
        transformations = transforms.Compose([transforms.Scale(224),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        x_min, y_min, x_max, y_max = face
        x_min -= 50
        x_max += 50
        y_min -= 50
        y_max += 30
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)
        # Crop face loosely
        cv2_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2_frame[y_min:y_max,x_min:x_max]
        img = Image.fromarray(img)

        # Transform
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = torch.autograd.Variable(img).to(self.device)

        yaw, pitch, roll = self.model(img)

        yaw_predicted = F.softmax(yaw, -1)
        pitch_predicted = F.softmax(pitch, -1)
        roll_predicted = F.softmax(roll, -1)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99
        return {'yaw': int(abs(yaw_predicted.item()) > config_dict['yaw']['angle']),
                'pitch': int(abs(pitch_predicted.item()) > config_dict['pitch']['angle']),
                'roll': int(abs(roll_predicted.item()) > config_dict['roll']['angle'])} # dictionary returning if head is tilted in any direction