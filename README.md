---
title: "BCCD Object Detection"
emoji: "📷"
colorFrom: "blue"
colorTo: "purple"
sdk: "gradio"
sdk_version: "5.21.0"  
app_file: "app.py"
pinned: false
---

# Object Detection App with YOLOS  

This is an object detection web application powered by **YOLOS** (You Only Look One-level Set). The app allows users to upload an image, detect objects, and visualize results with bounding boxes. It is built using **Gradio, Transformers, and PyTorch**.  

## 🚀 Features  
- Upload or drag and drop an image  
- Detect objects using a **pre-trained YOLOS model**  
- Adjust probability threshold for better accuracy  
- Select specific object classes to display (optional)  
- Download the output image with detections 

## 🛠 Installation  

1️⃣ **Clone the repository**  

git clone https://huggingface.co/spaces/Shailly29/BCCD_Object_Detection
cd BCCD_Object_Detection


2️⃣ **Install dependencies**  

pip install -r requirements.txt


3️⃣ **Run the application**  

python app.py

