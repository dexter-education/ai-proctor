import cv2

import config
import detector
import flagger

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    frame_num = 0
    config_dict = config.get_config()
    yolov5_obj = detector.yolov5_infer_single('yolov5s.pt', half=False)
    mtcnn_obj = detector.mtcnn_face()
    count_obj = flagger.flagger(config_dict)
    
    while True:
        ret, img = cap.read()
        if ret:
            count_obj.add_count(mtcnn_obj.detect(img))
            count_obj.add_count(yolov5_obj.detect(img))
            count_obj.check_counts(frame_num)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        frame_num += 1
    cap.release()
    cv2.destroyAllWindows()
