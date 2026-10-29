# AI-Powered Real-Time Object Detection & Multi-Object Tracking

A comprehensive, real-time Computer Vision pipeline that integrates **YOLO** (Ultralytics) for object detection and multiple tracking algorithms (custom **SORT** implementation and native **ByteTrack** / **BoT-SORT**) to perform multi-object tracking.

It features:
1. A **Modular Computer Vision Engine** written in Python.
2. A **High-Performance CLI App** using an interactive OpenCV GUI.
3. A **Premium Web Dashboard** built with Streamlit for video uploaded testing, config adjustment, live telemetry, and tracking history exports (CSV).

---

## Architecture Overview

### 1. Object Detection (YOLO)
The detector utilizes YOLO models (e.g. YOLOv8/v11) to perform frame-by-frame object detection. Detections are filtered by class (such as person, car, etc.) and confidence threshold.

### 2. Multi-Object Tracking
- **Custom SORT (Simple Online and Realtime Tracking)**:
  - **Kalman Filter**: Models object motion across frames. Tracks states: center $(x,y)$, scale/area $(s)$, aspect ratio $(r)$, and their velocities.
  - **Hungarian Association**: Employs `scipy.optimize.linear_sum_assignment` to associate predictions with new detections based on Intersection-over-Union (IoU) distance.
  - **Track Management**: Spawns new tracks, updates current ones, and deletes tracks when objects disappear (max age).
- **Native ByteTrack & BoT-SORT**:
  - Leverages Ultralytics' highly-optimized state-of-the-art native tracking algorithms.

### 3. Fading Trails & Sci-Fi HUD
- Draws fading movement trails behind tracked targets.
- Displays bounding boxes with tech-inspired corner-bracket styles.
- Draws glassy-styled tags containing ID labels and confidence.
- Overlays a system HUD displaying current FPS, active count, and total detections.

---

## Directory Structure

```
object-detection-tracking/
├── requirements.txt            # Package dependencies
├── README.md                   # Setup and usage guide
├── main.py                     # CLI & OpenCV GUI application
├── app.py                      # Streamlit interactive dashboard
├── core/
│   ├── __init__.py
│   ├── detector.py             # YOLO detector wrapper class
│   ├── tracker_sort.py         # Custom SORT Kalman + Hungarian tracking
│   ├── tracker_bridge.py       # Wrapper to toggle custom/native trackers
│   └── utils.py                # Graphical overlays, FPS, and trails
└── data/
    └── sample.mp4              # Sample traffic video
```

---

## Installation & Setup

1. **Clone or navigate** to the project directory:
   ```bash
   cd object-detection-tracking
   ```

2. **Create a Python Virtual Environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. High-Performance OpenCV CLI (`main.py`)
Run real-time tracking on your webcam or a video file.

```bash
# Run on webcam (index 0) using custom SORT tracker
python main.py --source 0 --tracker sort

# Run on the downloaded sample video using ByteTrack
python main.py --source data/sample.mp4 --tracker bytetrack

# Run on a custom video, filtering only for 'person' (ID 0) and 'car' (ID 2)
python main.py --source data/sample.mp4 --tracker botsort --classes 0,2 --output data/output.mp4
```

#### CLI Command Options:
- `--source`: Webcam index (e.g. `0`) or path to video file (default: `0`).
- `--tracker`: `sort`, `bytetrack`, or `botsort` (default: `sort`).
- `--model`: YOLO model name (e.g. `yolov8n.pt`, `yolov8s.pt`, `yolov11n.pt`) (default: `yolov8n.pt`).
- `--conf`: Confidence threshold for detections (default: `0.35`).
- `--iou-threshold`: NMS/association overlap threshold (default: `0.45`).
- `--classes`: Comma-separated class IDs to detect (e.g. `0,2` for person, car). Default detects all.
- `--output`: Path to write output processed MP4 video file.
- `--device`: Target device: `cpu`, `cuda`, `mps` (default: auto-detected).

#### Live Keyboard Controls:
* `q` or `ESC` : Quit and close window.
* `p` or `SPACE` : Pause / Resume video.
* `t` : Cycle through tracking algorithms (`SORT` ➔ `ByteTrack` ➔ `BoT-SORT`).
* `h` : Toggle HUD display overlay on/off.
* `l` : Toggle movement trails.

---

### 2. Interactive Web Dashboard (`app.py`)
Run a modern, visual web dashboard to adjust parameters in real-time, view stats, see charts, and download CSV reports.

```bash
streamlit run app.py
```

The browser will open automatically at: `http://localhost:8501`

#### Web Dashboard Features:
- **Interactive Control Panel**: Switch input source (webcam/upload/sample), tracking model, confidence levels, and active tracking classes.
- **Live Telemetry metrics**: Visual display of real-time FPS, Active Targets, and Cumulative unique count.
- **Interactive Analytics**: Graphical plots of active count timeline and class distribution bar charts.
- **Export Reports**: View logs of targets including dimensions, coordinates, and timestamps, and download them as a CSV.
