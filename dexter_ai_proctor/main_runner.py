import cv2

from . import config
from . import detector
from . import flagger

class runner:

    def __init__(self, mtcnn=None, yolov5_model=None, face_seg_model=None, head_pose_model=None, frame_num=0):
        
        self.frame_num = frame_num
        self.config_dict = config.get_config()
        if mtcnn is not None and head_pose_model is not None:
            self.mtcnn_obj = detector.mtcnn_face()
        if yolov5_model is not None:
            self.yolov5_obj = detector.yolov5_infer_single(yolov5_model)
        if face_seg_model is not None:
            self.seg_obj = detector.face_segmentation(weights=face_seg_model)
        if head_pose_model is not None:
            self.head_obj = detector.head_pose(head_pose_model)
        self.count_obj = flagger.flagger(self.config_dict)
        self.face = None

    def run_mtcnn(self, img):
        Dict, self.face = self.mtcnn_obj.detect(img)
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
        self.count_obj.check_counts(self.frame_num)
        self.frame_num += 1


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    runner_obj = runner(yolov5_model='yolov5s.pt')
    
    while True:
        ret, img = cap.read()
        if ret:
            runner_obj.run_yolov5(img)
            runner_obj.check_counts()
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()