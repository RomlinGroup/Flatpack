part_bash """
wget -nc -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
"""
part_bash """
if [ -f ../cat_and_dog.png ]; then
    cp -f ../cat_and_dog.png cat_and_dog.png
fi
"""
part_python """
import cv2
import json
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_FILE = 'cat_and_dog.png'
img = cv2.imread(IMAGE_FILE)

height, width = img.shape[:2]
new_width = 800
new_height = int(height * (new_width / width))
img_resized = cv2.resize(img, (new_width, new_height))

img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
detection_result = detector.detect(image)

image_copy = np.copy(image.numpy_view())
object_data = []

for i, detection in enumerate(detection_result.detections):
    bbox = detection.bounding_box
    x_min, y_min = int(bbox.origin_x), int(bbox.origin_y)
    x_max, y_max = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)

    coordinates = {
        \"id\": i,
        \"category\": detection.categories[0].category_name,
        \"score\": detection.categories[0].score,
        \"bounding_box\": {\"x_min\": x_min, \"y_min\": y_min, \"x_max\": x_max, \"y_max\": y_max},
        \"centroid\": {\"x\": (x_min + x_max) // 2, \"y\": (y_min + y_max) // 2}
    }
    object_data.append(coordinates)

    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
    label = f'{detection.categories[0].category_name}: {detection.categories[0].score:.2f}'
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = x_min + (x_max - x_min) // 2 - label_width // 2
    label_y = y_min + (y_max - y_min) // 2 + label_height // 2
    cv2.rectangle(image_copy, (label_x - 2, label_y - label_height - 2), (label_x + label_width + 2, label_y + 2), color=(255, 0, 0), thickness=-1)
    cv2.putText(image_copy, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

image_copy_bgr = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
cv2.imwrite('output_image_with_boxes.jpg', image_copy_bgr)

with open('detected_objects.json', 'w') as f:
    json.dump(object_data, f, indent=4)

print(\"Object data saved as detected_objects.json\")
"""
