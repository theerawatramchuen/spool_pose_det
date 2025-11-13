# Spool Pose Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green)
![YOLO](https://img.shields.io/badge/YOLO-Pose-orange)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900)

An automated computer vision system for detecting specific human poses ("spool poses") in video files using YOLO-based pose estimation. The system processes videos in batch, identifies frames meeting predefined pose criteria, and saves both individual frames and video clips when spool poses are detected.

## 🚀 Features

- **Real-time Pose Detection**: Uses YOLO11s-pose model for accurate human pose estimation
- **Batch Processing**: Automatically processes all MP4 files in a designated folder
- **Smart Validation**: Configurable pose validation based on wrist positions and shoulder distances
- **GPU Acceleration**: Automatically utilizes CUDA for faster processing
- **Visual Feedback**: Real-time visualization with pose overlays and validation status
- **Evidence Capture**: Saves both individual frames and video clips around detections
- **File Management**: Organized output structure with automatic file categorization

## 📋 Requirements

### Python Dependencies
- Python 3.8+
- ultralytics
- opencv-python
- numpy
- torch
- pathlib
- shutil

### Hardware
- CUDA-capable GPU (recommended) or CPU
- Sufficient storage for video processing

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/spool-pose-detection.git
   cd spool-pose-detection
