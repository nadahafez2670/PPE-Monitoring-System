# 🏗️ SiteGuard AI – PPE Detection & Construction Site Safety Monitoring

An AI-powered Computer Vision system for monitoring construction site safety by detecting workers and verifying compliance with Personal Protective Equipment (PPE) using a fine-tuned **YOLOv8s** model.

---

## 📸 Application Preview

<p align="center">
<img src="assets/Screenshot 2026-07-25 005815.png" width="100%">
</p>

---

# 📌 Overview

Construction sites are among the highest-risk working environments, where failure to wear Personal Protective Equipment (PPE) can lead to serious accidents.

This project provides an AI-powered safety monitoring system that automatically detects workers, identifies PPE violations, calculates compliance statistics, estimates the overall site risk level, and generates an interactive dashboard for safety inspection.

---

# ✨ Features

- 👷 Worker Detection
- 🪖 Hardhat Detection
- 🦺 Safety Vest Detection
- ❌ PPE Violation Detection
- 📊 Interactive Safety Dashboard
- 📈 Compliance Analytics
- 🚨 Risk Level Assessment
- 🤖 AI Safety Recommendations
- 📄 Downloadable Safety Report
- 🎥 Image & Video Processing

---

# 🧠 Model

**Model:** YOLOv8s (Fine-Tuned)

### Detection Classes

| Class |
|--------|
| Person |
| Hardhat |
| Safety Vest |
| NO-Hardhat |
| NO-Safety Vest |

---

# 📊 Model Performance

| Metric | Score |
|---------|------:|
| Precision | **79.6%** |
| Recall | **63.1%** |
| mAP@50 | **70.2%** |
| mAP@50-95 | **44.3%** |

---

# ⚙️ Business Logic

After object detection, the system performs additional business analysis by:

- Counting total workers
- Matching PPE items to each detected worker
- Detecting safety violations
- Calculating PPE compliance rate
- Estimating construction site risk level
- Generating AI safety recommendations
- Producing a downloadable inspection report

---

# 🖥️ Streamlit Application

The application consists of three professional pages.

## 🏗️ Project Overview

- About Project
- Business Problem
- KPIs
- Workflow
- Project Features

---

## 🤖 AI Analytics

- Dataset Information
- YOLOv8s Architecture
- Evaluation Metrics
- Training Results
- Confusion Matrix
- Sample Detection

---

## 📊 Safety Monitoring Dashboard

- Upload Image or Video
- Original Media
- Detection Results
- Worker Statistics
- Compliance Dashboard
- Interactive Charts
- Risk Level Indicator
- AI Recommendation
- Download Inspection Report

---

# 📸 Screenshots


## Safety Monitoring Dashboard

<p align="center">
<img src="assets/Screenshot 2026-07-23 104353.png" width="90%">
</p>

---

## Sample Detection

<p align="center">
<img src="assets/Screenshot 2026-07-23 104244.png" width="90%">
</p>

---


# 🛠️ Tech Stack

- Python
- YOLOv8s (Ultralytics)
- OpenCV
- NumPy
- Streamlit
- Plotly
- Matplotlib

---

# 📂 Project Structure

```
PPE-Monitoring-System
│
├── app.py
├── model_engine.py
├── best.pt
├── requirements.txt
├── README.md
│
├── assets/
│   ├── banner.png
│   ├── overview.png
│   ├── analytics.png
│   ├── dashboard.png
│   ├── detection.png
│   ├── results.png
│   └── confusion_matrix.png
│
├── notebooks/
│   ├── Training.ipynb
│   └── Final_Project.ipynb
│
├── outputs/
├── dataset/
└── filtered_dataset/
```

---

# 🚀 Future Improvements

- Live CCTV Monitoring
- Multi-Camera Support
- Worker Tracking
- Fire & Smoke Detection
- Fall Detection
- Email & SMS Alerts
- Cloud Deployment
- Real-Time Notifications

---

# 👥 Team

- **Nada Ahmed Ahmed**
- **Nancy Nabil Mohamed**
- **Eman Mohamed Mousa**

---

## ⭐ If you found this project useful, consider giving it a Star!
