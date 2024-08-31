part_bash """
wget -q -O efficientdet.tflite -q https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite

wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg
"""
part_python """
import cv2
import mediapipe as mp
import numpy as np

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

for detection in detection_result.detections:
    bbox = detection.bounding_box
    start_point = (int(bbox.origin_x), int(bbox.origin_y))
    end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
    cv2.rectangle(image_copy, start_point, end_point, color=(0, 255, 0), thickness=2)

    category = detection.categories[0]
    label = f'{category.category_name}: {category.score:.2f}'
    label_position = (start_point[0], start_point[1] - 10)
    cv2.putText(image_copy, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)

output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, image_copy)

print(f\"Annotated image saved as {output_image_path}\")
"""