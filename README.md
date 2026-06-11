# 🔐 Security AI System — Computer Vision & Security AI Engineer

## 📌 Project Overview
An AI-powered security system built using Python, OpenCV, Face Recognition, and YOLOv8.
This system can detect faces, recognize known persons, detect motion, and send automated email alerts.

## 🚀 Modules

### Module 1 — Face Recognition
- Recognizes known faces in real-time
- Alerts when unknown person is detected
- Green box for known, red box for unknown

### Module 2 — Motion Detection
- Detects any motion in camera frame
- Logs motion events with timestamp
- Smart background updating

### Module 3 — YOLO Human & Vehicle Detection
- Detects humans and vehicles using YOLOv8
- Shows confidence percentage
- Real-time detection at 54+ FPS

### Module 4 — Security Automation
- Captures photo of unknown person
- Sends email alert with photo automatically
- Logs all security events

### Module 5 — AI Model Optimization
- Compares YOLOv8 Nano vs Small models
- Benchmarks FPS and accuracy
- Recommends best model for device

## 🛠️ Technologies Used
- Python 3.11
- OpenCV
- face_recognition
- YOLOv8 (Ultralytics)
- PyTorch
- dlib

## 📋 Requirements
- Python 3.11
- Webcam

## ⚙️ Installation
```bash
pip3 install opencv-python face-recognition ultralytics torch numpy
```

## ▶️ How to Run

### Run all modules together:
```bash
python3.11 main.py
```

### Run individual modules:
```bash
# Face Recognition
python3.11 face_recognition_module/recognize.py

# Motion Detection
python3.11 motion_detection/detect.py

# YOLO Detection
python3.11 smart_camera/camera_stream.py

# Model Optimization
python3.11 model_optimization/optimize.py
```

## 📁 Project Structure
security-ai-project/
├── dataset/              # Known faces
├── face_recognition_module/
├── motion_detection/
├── smart_camera/
├── security_automation/
├── model_optimization/
├── reports/              # Saved alerts and reports
└── main.py               # Run all modules together

## 📊 Results
- Face Recognition: ✅ Working
- Motion Detection: ✅ Working  
- YOLO Detection: ✅ 54.5 FPS
- Email Alerts: ✅ Working
- Best Model: YOLOv8 Nano (54.5 FPS)

## 👩‍💻 Developer
Jahnavi Bandaru — Computer Vision & Security AI 