import cv2

import config
from . import detector
from . import flagger
from . import load_models


def get_yolov5(model_path, imgsz=640, half=False):
    model, device, stride, names, pt, imgsz, half = load_models.load_yolov5(model_path, imgsz, half)
    return model, device, stride, names, pt, imgsz, half

def get_face_segmentation(model_path):
    model, device = load_models.load_face_segmentation(model_path)
    return model, device

def get_head_pose(model_path):
    model, device, idx_tensor = load_models.load_head_pose(model_path)
    return model, device, idx_tensor

class runner:

    def __init__(self, frame_num=0):
        
        self.frame_num = frame_num
        self.config_dict = config.get_config()
        self.count_obj = flagger.flagger(self.config_dict)
        self.face = None
    
    def init_yolov5_face(self, model, device, stride, names, pt, imgsz, half):
        self.yolov5_face_obj = detector.yolov5_face(model, device, stride, names, pt, imgsz, half)
    
    def init_yolov5(self, model, device, stride, names, pt, imgsz, half):
        self.yolov5_obj = detector.yolov5_infer_single(model, device, stride, names, pt, imgsz, half)

    def init_face_segmentation(self, model, device):
        self.seg_obj = detector.face_segmentation(model, device)
    
    def init_head_pose(self, model, device, idx_tensor):
        self.head_obj = detector.head_pose(model, device, idx_tensor)

    def run_yolov5_face(self, img):
        Dict, self.face = self.yolov5_face_obj.detect(img)
        self.count_obj.add_count(Dict)

    def run_yolov5(self, img):
        self.count_obj.add_count(self.yolov5_obj.detect(img))
    
    def run_face_seg(self, img):
        if self.face != None and self.face != [0, 0, 0, 0]:
            self.count_obj.add_count(self.seg_obj.detect(img))
    
    def run_head_pose(self, img):
        if self.face != [0, 0, 0, 0]:
            self.count_obj.add_count(self.head_obj.detect(img, self.face))
    
    def check_counts(self):
        flag = self.count_obj.check_counts(self.frame_num)
        self.frame_num += 1
        return flag


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    runner_obj = runner(yolov5_model='yolov5s.pt')
    
    while True:
        ret, img = cap.read()
        if ret:
            runner_obj.run_yolov5(img)
            flag = runner_obj.check_counts()
            print(flag)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
