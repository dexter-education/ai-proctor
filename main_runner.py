import cv2
from mtcnn.mtcnn import MTCNN
from yolov5_single_detector import infer_single

class count:
    """Class to count and raise flags
    """

    def __init__(self):

        self.MIN_CONTINOUS_FRAMES = [10, 10, 10, 10]
        self.frame_counter = [0, 0, 0, 0]
        self.required_counts = [1, 1, 0, 0]
        self.level = ['red', 'red', 'red', 'red']
        self.message = ['face count violation', 'person count violation', 'mobile phone violation', 'laptop violation']

    def flag(self, level, frame_num, message):
        print(level, frame_num, message)

    def check_counts(self, counts, frame_num):

        for i in range(len(counts)):
            
            if self.frame_counter[i] > self.MIN_CONTINOUS_FRAMES[i]:
                self.flag(self.level[i], frame_num - self.MIN_CONTINOUS_FRAMES[i], self.message[i])
                self.frame_counter[i] = 0

            if counts[i] != self.required_counts[i]:
                self.frame_counter[i] += 1
            else:
                self.frame_counter[i] = 0

def face_count(img):
    """Function to count faces in the image
    """
    model = MTCNN()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = model.detect_faces(img_rgb)
    face_count = sum([1 if face['confidence'] > 0.95 else 0 for face in res])

    return face_count

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    frame_num = 0
    yolov5_obj = infer_single('yolov5s.pt', half=False)
    count_obj = count()
    
    while True:
        ret, img = cap.read()
        if ret:
            counts = [face_count(img)]
            counts.extend(yolov5_obj.detect(img))
            count_obj.check_counts(counts, frame_num)
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        frame_num += 1
    cap.release()
    cv2.destroyAllWindows()
