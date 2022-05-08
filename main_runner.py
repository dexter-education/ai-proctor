import cv2

import config
import detector
import flagger

class runner:

    def __init__(self, yolov5_model='yolov5s.pt', frame_num=0):
        
        self.frame_num = frame_num
        self.config_dict = config.get_config()
        self.yolov5_obj = detector.yolov5_infer_single(yolov5_model)
        self.mtcnn_obj = detector.mtcnn_face()
        self.count_obj = flagger.flagger(self.config_dict)

    def run(self, img):

        self.count_obj.add_count(self.mtcnn_obj.detect(img))
        self.count_obj.add_count(self.yolov5_obj.detect(img))
        self.count_obj.check_counts(self.frame_num)
        self.frame_num += 1


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    runner_obj = runner()
    
    while True:
        ret, img = cap.read()
        if ret:
            runner_obj.run(img)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
