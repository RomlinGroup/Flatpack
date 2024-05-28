import cv2
from transformers import pipeline

# Load the local image file using OpenCV
image_path = "thunderstorm.png"
image = cv2.imread(image_path)

# Convert the image from BGR (OpenCV default) to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a pipeline for image classification using the specified model
pipe = pipeline("image-classification", model="dima806/weather_types_image_detection")

# Perform classification on the image
results = pipe(image)

# Display the results
print(results)
