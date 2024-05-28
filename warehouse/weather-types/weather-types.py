import cv2
from PIL import Image
from transformers import pipeline

# Load the local image file using OpenCV
image_path = "thunderstorm.png"  # Replace with the correct path to your image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise ValueError("Image not found or unable to load")

# Convert the image from BGR (OpenCV default) to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the OpenCV image (numpy array) to a PIL image
image_pil = Image.fromarray(image_rgb)

# Create a pipeline for image classification using the specified model
pipe = pipeline("image-classification", model="dima806/weather_types_image_detection")

# Perform classification on the image
results = pipe(image_pil)

# Display the results
print(results)
