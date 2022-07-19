import cv2
from dexter_ai_proctor import main_runner

yolov5_face_model, device, stride, names, pt, imgsz, half = main_runner.get_yolov5('face.pt')
yolov5_model, device, stride, names, pt, imgsz, half = main_runner.get_yolov5('dexter_ai_proctor/yolov5s.pt')
obj = main_runner.runner()
obj.init_yolov5_face(yolov5_face_model, device, stride, names, pt, imgsz, half)
obj2 = main_runner.runner()
obj2.init_yolov5(yolov5_model, device, stride, names, pt, imgsz, half)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        obj.run_yolov5_face(frame)
        obj2.run_yolov5(frame)
        flag = obj.check_counts()
        if len(flag['message']):
            print(flag)
        flag2 = obj2.check_counts()
        if len(flag2['message']):
            print('flag2:', flag2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break