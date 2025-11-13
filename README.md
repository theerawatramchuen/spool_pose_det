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

2. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python numpy torch

3. **Download YOLO model**
   - The system automatically downloads yolo11s-pose.pt on first run
   - Or manually download from Ultralytics and place in project directory





## Spool Pose Detection System - Code Documentation
### Overview
This system is designed to automatically detect "spool poses" in video files using pose estimation with YOLO. It processes videos in a specified folder, identifies frames where subjects meet specific pose criteria (wrist positions and shoulder distances), and saves both individual frames and video clips when spool poses are detected.

### System Architecture
### Core Components
- Pose Detection Engine - YOLO-based pose estimation
- Pose Validation System - Rules-based validation of spool poses
- Video Processing Pipeline - Frame-by-frame analysis
- File Management System - Automated file organization
- Visualization Tools - Real-time display and overlay generation

Key Functions Documentation
1. validate_wrist_position(person_kpts, min_percent, max_percent, wrist_type="both", max_shoulder_percent=30)
Purpose: Validates if detected wrists meet spool pose criteria

Parameters:

person_kpts: Array of person keypoints (17 COCO format keypoints)

min_percent: Minimum vertical percentage threshold

max_percent: Maximum vertical percentage threshold

wrist_type: Which wrists to validate ("left", "right", "both")

max_shoulder_percent: Maximum shoulder width percentage

Validation Logic:

Calculates reference range from shoulders (0%) to hips (100%)

Validates wrist vertical positions within specified percentage range

Validates shoulder horizontal distance doesn't exceed maximum percentage

Returns validation results for left wrist, right wrist, and shoulders

Key Features:

Robust to missing keypoints

Percentage-based normalization for scale invariance

Configurable validation parameters

2. process_video(source, output_path, model, confidence_threshold=0.5, ...)
Purpose: Main video processing pipeline

Processing Flow:

Opens video source and initializes processing

Processes central 50% region of each frame for efficiency

Runs YOLO pose estimation on central region

Adjusts keypoint coordinates to full frame

Validates poses using validate_wrist_position

Saves frames and clips when spool poses detected

Provides real-time visualization and controls

Optimization Features:

Central region processing reduces computation by 50%

GPU acceleration support

Frame skipping after clip detection to avoid redundant processing

3. process_specific_sources()
Purpose: Automated batch processing of video files

Workflow:

Scans input folder for .mp4 files

Moves each file to processed_files folder before processing

Processes files sequentially from processed_files folder

Uses GPU if available for accelerated processing

Maintains organized file structure

File Management:

Input folder: C:\RecordDownload

Processed files: C:\RecordDownload\processed_files

Output files: Saved in input folder with processed_ prefix

Spool pose images: Saved in spool_pose folder

Video clips: Saved in spool_pose_clips folder

4. draw_pose_keypoints(image, keypoints, validation_results=None)
Purpose: Visualizes pose estimation results and validation status

Visual Elements:

Skeleton connections between keypoints

Keypoint circles

Validation status text (PASS/FAIL)

Percentage measurements

Color-coded results (green=valid, red=invalid)

5. save_spool_pose_frame() & save_spool_pose_clip()
Purpose: Captures and saves spool pose evidence

Frame Capture:

Saves individual frames with pose overlays

Includes timestamp and frame information

Stores in spool_pose folder

Clip Capture:

Creates 600-frame video clips (300 frames before/after detection)

Saves in spool_pose_clips folder

Provides context around detected poses

Configuration Parameters
Pose Validation Settings
python
MIN_VERTICAL_PERCENT = -20    # Minimum wrist vertical position (%)
MAX_VERTICAL_PERCENT = 20     # Maximum wrist vertical position (%)
WRIST_TYPE = "both"           # Which wrists to validate
MAX_SHOULDER_PERCENT = 10     # Maximum shoulder width (%)
Processing Settings
python
confidence_threshold = 0.5    # YOLO detection confidence
central_region_width = 0.5    # Process only central 50% of frame
clip_duration = 600           # Frames in saved clips (300 before/after)
Keypoint Mapping (COCO 17 Format)
The system uses COCO 17 keypoint format:

0: nose

1-2: eyes

3-4: ears

5-6: shoulders

7-8: elbows

9-10: wrists

11-12: hips

13-16: knees and ankles

Dependencies
Core Libraries
ultralytics - YOLO pose estimation

opencv-python - Video I/O and image processing

numpy - Numerical computations

torch - GPU acceleration

shutil, os, pathlib - File management

Model Requirements
Primary model: yolo11s-pose.pt

GPU support via CUDA (falls back to CPU)

Usage Examples
Basic Execution
python
python spool4vid_folder_gpu.py
Custom Processing
python
process_video(
    "input_video.mp4", 
    "output_video.mp4", 
    model,
    min_vertical_percent=-15,
    max_vertical_percent=25,
    wrist_type="left"
)
Output Structure
text
C:\RecordDownload\
├── processed_*.mp4              # Processed video outputs
├── processed_files\             # Original files after processing
│   └── *.mp4
├── spool_pose\                  # Detected pose frames
│   └── spool_pose_*.jpg
└── spool_pose_clips\            # Video clips around detections
    └── spool_pose_*.mp4
Performance Considerations
GPU Acceleration
Automatically detects and uses CUDA-capable GPUs

Falls back to CPU if GPU unavailable

Significant speed improvement with GPU

Processing Optimizations
Central region processing reduces computation

Frame skipping after detection events

Efficient keypoint coordinate adjustment

Memory Management
Video streams processed sequentially

Frames released after processing

Model loaded once for batch processing

Error Handling
Robust Keypoint Processing
Handles missing or invalid keypoints

Validates reference ranges before calculations

Continues processing despite individual frame errors

File Management
Creates necessary directories automatically

Handles file move operations with error checking

Continues processing next file on errors

Extension Points
Adding New Pose Validations
python
def validate_custom_pose(person_kpts, parameters):
    # Add custom validation logic
    return validation_results
Supporting Additional Models
python
model = YOLO("custom-pose-model.pt")
Custom Output Formats
Modify save_spool_pose_frame() and save_spool_pose_clip() for different output formats.

Troubleshooting
Common Issues
No GPU detection: Check CUDA installation and torch CUDA support

Missing model: Ensure yolo11s-pose.pt is available

File permission errors: Verify write permissions in target directories

Video read errors: Check video file integrity and codec support

Debug Features
Real-time visualization with pose overlays

Pause/resume with 'p' key

Manual frame save with 's' key

Detailed console logging

Future Development Opportunities
Potential Enhancements
Multi-person tracking - Track individuals across frames

Temporal smoothing - Apply filters to reduce jitter

Additional pose types - Support for multiple pose validations

Web interface - Browser-based control and monitoring

Database integration - Store detection results and metadata

Real-time streaming - Process live video feeds

Advanced analytics - Pose statistics and trend analysis

This documentation provides a comprehensive overview of the spool pose detection system for future development and maintenance.
