part_bash """
wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg
"""
part_python """
import cv2
import mediapipe as mp
import numpy as np
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_FILE = 'image.jpg'
img = cv2.imread(IMAGE_FILE)

base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

image = mp.Image.create_from_file(IMAGE_FILE)
detection_result = detector.detect(image)

image_copy = np.copy(image.numpy_view())

object_data = []

for i, detection in enumerate(detection_result.detections):
    bbox = detection.bounding_box

    x_min = int(bbox.origin_x)
    y_min = int(bbox.origin_y)
    x_max = int(bbox.origin_x + bbox.width)
    y_max = int(bbox.origin_y + bbox.height)

    coordinates = {
        \"id\": i,
        \"category\": detection.categories[0].category_name,
        \"score\": detection.categories[0].score,
        \"bounding_box\": {
            \"x_min\": x_min,
            \"y_min\": y_min,
            \"x_max\": x_max,
            \"y_max\": y_max
        },
        \"centroid\": {
            \"x\": (x_min + x_max) // 2,
            \"y\": (y_min + y_max) // 2
        }
    }

    object_data.append(coordinates)

    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    label = f'{detection.categories[0].category_name}: {detection.categories[0].score:.2f}'
    label_position = (x_min, y_min - 10)
    cv2.putText(image_copy, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
cv2.imwrite('output_image_with_boxes.jpg', image_copy)

with open('detected_objects.json', 'w') as f:
    json.dump(object_data, f, indent=4)

print(\"Object data saved as detected_objects.json\")
"""