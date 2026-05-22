# 🚦 Smart City Automated Traffic Enforcement

## An enterprise-grade Computer Vision pipeline that monitors live traffic feeds, detects red-light runners, estimates vehicle speeds, classifies vehicle types, and automatically crops offenders to extract license plates via OCR.

## 🔴 **Live Demo:** [Click here to test the app](https://huggingface.co/spaces/NivedhReddy/Smart-City-Traffic-Enforcement)

## *(Note: The live demo is deployed on a free Hugging Face CPU tier. To prevent hardware timeouts and "thread-thrashing" on a 2-core CPU, the cloud deployment is aggressively optimized to process a maximum of 3 seconds of footage and limits Deep Learning OCR scans. The full architecture was originally developed and run on T4 GPUs via Google Colab).*

![App Screenshot](https://github.com/NivedhReddy2048/smart-city-traffic-enforcement/blob/main/Screenshots/Traffic_1.png?raw=true)

![App Screenshot](https://github.com/NivedhReddy2048/smart-city-traffic-enforcement/blob/main/Screenshots/Traffic_2.png?raw=true)

![App Screenshot](https://github.com/NivedhReddy2048/smart-city-traffic-enforcement/blob/main/Screenshots/Traffic_3.png?raw=true)

![App Screenshot](https://github.com/NivedhReddy2048/smart-city-traffic-enforcement/blob/main/Screenshots/Traffic_4.png?raw=true)

## 🧠 The Problem & Solution

### Modern traffic enforcement relies heavily on manual monitoring or expensive proprietary ALPR (Automatic License Plate Recognition) hardware. 

### This project simulates a software-first smart city dashboard using open-source AI. By utilizing state-of-the-art object detection, mathematical object tracking, spatial logic (virtual tripwires), and optical character recognition, the system automatically flags violations, tracks vehicle velocity, and extracts license plates in real-time.

## ⚙️ Tech Stack

### **Computer Vision:** `OpenCV` (for dynamic frame extraction, cropping, and thresholding).
### **Object Detection:** `YOLOv8` (Ultralytics) for high-speed vehicle identification.
### **Multi-Object Tracking:** `ByteTrack` algorithm to maintain unique IDs and prevent tracking amnesia.
### **Spatial Analytics:** `Supervision` (Roboflow) for coordinate mapping and virtual line-zone logic.
### **Optical Character Recognition:** `EasyOCR` (Deep Learning-based text extraction).
### **Frontend/UI:** `Gradio` (Blocks API) for a responsive, security-style admin dashboard.

## ✨ Key Engineering Features

### 1. **Real-Time Inference & Tracking:** Optimized to run complex YOLO inference and ByteTrack logic smoothly across crowded traffic frames.
### 2. **Spatial Tripwires & Speed Math:** Implemented mathematical line-zones to flag vehicles crossing restricted boundaries. Integrated a pixel-distance-over-time formula to estimate vehicle speeds dynamically.
### 3. **Dynamic Tensor Cropping & Caching:** The engine extracts the exact pixel bounding box of an offending vehicle. It utilizes a caching system (`ticketed_cars` set) to ensure a single vehicle is only processed once, reducing compute load by 90%.
### 4. **Cloud MLOps Optimization:** Implemented strict OS-level thread locking (`os.environ["OMP_NUM_THREADS"] = "1"`) to prevent PyTorch and OpenCV from thread-thrashing when deployed to limited-core Docker containers.

## 🛑 Known Limitations & Real-World Scaling 

### As a Proof of Concept (PoC) built on open-source weights, this architecture encounters several expected bottlenecks that would be resolved in a production deployment:

### 1. **COCO Dataset Bias (Vehicle Misclassification):** The base `yolov8n.pt` model is pre-trained on the Microsoft COCO dataset, which lacks regional context. For example, it does not recognize Indian Auto-Rickshaws, mathematically defaulting them to "Heavy Vehicles." *Production Fix: Fine-tune the YOLO model on a localized, custom dataset.*
### 2. **Resolution & Compression Loss (OCR Failures):** The pipeline frequently returns "UNREADABLE" text for license plates due to internet video compression permanently destroying the pixel density of the plates. *Production Fix: Utilize physical 4K optical-zoom cameras paired with a specialized License Plate Detection YOLO model.*
### 3. **Perspective Distortion (Inaccurate Speeds):** Because 2D video distorts distance, vehicles appear to accelerate mathematically as they approach the camera. *Production Fix: Implement Homography Projection to warp the 2D camera feed into a top-down 3D grid for accurate spatial math.*

## 💻 Run It Locally

### Want to test the full-scale GPU enforcement engine on your own machine?

### 1. Clone the repository:
   ```bash
   git clone [https://github.com/NivedhReddy2048/smart-city-traffic-enforcement.git](https://github.com/NivedhReddy2048/smart-city-traffic-enforcement.git)
```
### 2. Install the required dependencies:
```bash
pip install ultralytics supervision opencv-python easyocr gradio
```
### 3. Run the Jupyter Notebook to launch the Gradio web interface.

![App Screenshot](https://github.com/NivedhReddy2048/smart-city-traffic-enforcement/blob/main/Screenshots/Traffic_5.png?raw=true)
