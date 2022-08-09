import cv2
import os
import json
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm

model = MTCNN()
images = os.listdir('images')
# loop over images and detect faces in them. Generate a coco json file from that
faces = []
images_list = []
id_annotations = 1
os.makedirs('coco', exist_ok=True)

for i, image in enumerate(tqdm(images)):
    img = cv2.imread(f'images/{image}')
    height, width = img.shape[:2]
    res = model.detect_faces(img)
    for face in res:
        confidence = face['confidence']
        if confidence > 0.9:
            x0, y0, w, h = face['box']
            faces.append({"id": id_annotations, "image_id": i + 1, "bbox": [x0, y0, w, h], "category_id": 4, "segmentation": [], "area": w * h, "iscrowd": 0, "attributes": {"occluded": False}})

    images_list.append({'id': i + 1, 'file_name': image, 'height': height, 'width': width})

    # draw faces on image
    for face in res:
        confidence = face['confidence']
        if confidence > 0.9:
            x0, y0, w, h = face['box']
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)

    # save image with faces drawn on it
    cv2.imwrite(f'coco/{image}', img)

coco = {'images': images_list, 'categories': [{'supercategory': 'none', 'id': 4, 'name': 'face'}], 'annotations': faces}
with open('face_annotations.json', 'w') as f:
    json.dump(coco, f)
