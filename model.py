# Install dependencies
#pip install torch torchvision
#pip install -U ultralytics

# Clone YOLOv5 Repository
# git clone https://github.com/ultralytics/yolov5.git


# Install YOLOv5 Package
#pip install -e .

# Clone BCCD Dataset
#git clone https://github.com/Shenggan/BCCD_Dataset.git


# Verify Dataset Structure
import os

if not os.path.exists('train/images') or not os.path.exists('val/images'):
    raise FileNotFoundError('Dataset structure is incorrect. Make sure train and val directories contain images.')

# Create data.yaml File
data_yaml_content = """
train: /content/BCCD_Dataset/BCCD/train/images
val: /content/BCCD_Dataset/BCCD/val/images

# Number of classes
nc: 3

# Class names
names: ['RBC', 'WBC', 'Platelets']
"""

with open('data.yaml', 'w') as f:
    f.write(data_yaml_content)


# Import necessary libraries
import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load YOLOv5 model
model = YOLO('yolov5s.yaml')

# Fine-tune the model on the BCCD dataset
model.train(data='/content/BCCD_Dataset/BCCD/data.yaml', epochs=50)

# Save the fine-tuned model
model.save('yolov5_bccd.pt')

# Image Preprocessing

def preprocess_image(image, augmentations):
    """
    Apply data augmentations to the image.
    """
    for aug in augmentations:
        image = aug(image)
    return image

# Example augmentations
def rotate_image(image, angle):
    """
    Rotate the image by a given angle.
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def crop_image(image, x, y, w, h):
    """
    Crop the image to the specified rectangle.
    """
    return image[y:y+h, x:x+w]

# Model Inference

def perform_inference(model, image):
    """
    Perform inference using the fine-tuned YOLO model.
    """
    results = model(image)
    return results

# Example usage
# Load an image
image = cv2.imread('example.jpg')

# Preprocess the image
augmentations = [lambda x: rotate_image(x, 30), lambda x: crop_image(x, 50, 50, 200, 200)]
preprocessed_image = preprocess_image(image, augmentations)

# Perform inference
results = perform_inference(model, preprocessed_image)

# Display results
plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
plt.show()
print(results)