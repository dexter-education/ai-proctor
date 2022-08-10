import cv2

import config
from . import detector
from . import flagger

def model_objects(yolov5_face=None, yolov5_model=None, face_seg_model=None, head_pose_model=None):
    yolov5_face_obj = detector.yolov5_face(yolov5_face) if yolov5_face is not None or head_pose_model is not None else None
    yolov5_obj = detector.yolov5_infer_single(yolov5_model) if yolov5_model is not None else None
    seg_obj = detector.face_segmentation(weights=face_seg_model) if face_seg_model is not None else None
    head_obj = detector.head_pose(head_pose_model) if head_pose_model is not None else None
    return yolov5_face_obj, yolov5_obj, seg_obj, head_obj

class runner:

    def __init__(self, model_objects, frame_num=0):
        self.frame_num = frame_num
        self.config_dict = config.get_config()
        self.count_obj = flagger.flagger(self.config_dict)
        self.face = None
        self.yolov5_face_obj, self.yolov5_obj, self.seg_obj, self.head_obj = model_objects  # unpacking model objects

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
    model_objs = main_runner.model_objects(yolov5_model='yolov5s.pt')
    runner_obj = main_runner.runner(model_objs)

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