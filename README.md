# 🦺 PPE Detection & Construction Site Safety Monitoring

An AI-powered Computer Vision system for monitoring construction site safety by detecting workers and verifying compliance with Personal Protective Equipment (PPE) using YOLOv8.

---

## 📌 Overview

This project automatically detects construction workers and checks whether they are wearing the required safety equipment.

The system identifies PPE violations, calculates safety statistics, estimates the overall site risk level, and generates an interactive dashboard for safety monitoring.

---

## 🚀 Features

- 👷 Worker Detection
- 🪖 Hardhat Detection
- 🦺 Safety Vest Detection
- ❌ PPE Violation Detection
- 📊 Safety Compliance Dashboard
- 📈 Interactive Charts
- 🚨 Risk Level Assessment
- 🤖 AI Safety Recommendations
- 📄 Downloadable Safety Report
- 🎥 Image & Video Inference

---

## 🧠 Model

- YOLOv8
- Custom Fine-Tuned Model
- 5 Detection Classes

| Class |
|--------|
| Person |
| Hardhat |
| Safety Vest |
| NO-Hardhat |
| NO-Safety Vest |

---

## 📊 Business Logic

After object detection, the system:

- Counts total workers
- Matches PPE with each worker
- Detects safety violations
- Calculates compliance rate
- Determines risk level
- Generates safety recommendations
- Produces an inspection report

---

## 🏗️ Streamlit Application

The application consists of three main sections:

### 🏗️ Project Overview

- About Project
- Business Problem
- KPIs
- Workflow
- Features

### 🤖 AI Analytics

- Dataset Information
- YOLO11 Model
- Performance Metrics
- Results
- Confusion Matrix
- Sample Detection

### 📊 Safety Monitoring Dashboard

- Upload Image / Video
- Original Media
- Detection Results
- Safety Dashboard
- Compliance Charts
- Risk Level
- AI Recommendation
- Download Report

---

## 📈 Evaluation Metrics

| Metric | Value |
|---------|-------|
| Precision | 79.6% |
| Recall | 63.1% |
| mAP@50 | 70.2% |
| mAP@50-95 | 44.3% |

---

## 🛠️ Technologies

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Streamlit
- Plotly
- Matplotlib

---

## 📂 Project Structure

```
PPE-Detection/
│
├── app.py
├── model_engine.py
├── best.pt
├── requirements.txt
├── Final_Project.ipynb
│
├── outputs/
│
├── dataset/
│
├── filtered_dataset/
│
└── README.md
```

---

## 🎯 Future Improvements

- Live CCTV Monitoring
- Multi-Camera Support
- Fire & Smoke Detection
- Fall Detection
- Email/SMS Alerts
- Cloud Deployment
- Real-Time Notifications

---

## 👥 Team

· Nada Ahmed Ahmed    . Nancy Nabil Mohamed   · Eman Mohamed Mousa   
