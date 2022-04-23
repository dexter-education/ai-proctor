import cv2
from mtcnn.mtcnn import MTCNN

def flag(message):
    print(message)

if __name__ == "__main__":

    MIN_CONTINOUS_FRAMES = 10
    cap = cv2.VideoCapture(0)
    model = MTCNN()
    frame_counter = 0
    
    while True:
        ret, img = cap.read()
        if ret:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = model.detect_faces(img_rgb)
            face_count = sum([1 if face['confidence'] > 0.95 else 0 for face in res])
            if face_count != 1:
                frame_counter += 1
            else:
                frame_counter = 0
            if frame_counter > MIN_CONTINOUS_FRAMES:
                flag("No or more than one face detected")
            # cv2.imshow('img', img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
