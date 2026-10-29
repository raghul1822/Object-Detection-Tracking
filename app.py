import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from core.detector import YOLODetector
from core.tracker_bridge import TrackerBridge
from core.utils import (
    get_color_for_id,
    get_color_for_class,
    draw_corner_rect,
    draw_glassy_label,
    draw_fading_trail,
    FPSCalculator
)

# Page configuration
st.set_page_config(
    page_title="Aurora AI Vision - Object Detection & Tracking",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description
st.markdown("<h1 id='aurora-ai-vision' class='title-gradient'>🧠 Aurora AI Vision</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family: \"Inter\", sans-serif; font-size: 16px !important; font-weight: 400 !important; color: #9BAEC8; margin-bottom: 25px;'>Real-time Multi-Object Detection & AI Telemetry Tracking Dashboard</p>", unsafe_allow_html=True)

# Create directories if they do not exist
os.makedirs("data", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Cache the detector so model loading happens only once per selected model
@st.cache_resource
def load_detector(model_name):
    return YOLODetector(model_name=model_name)

# Default adapter for class filters before a detector is loaded
DEFAULT_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    44: "bottle",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    53: "pizza",
    56: "chair",
    57: "couch",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    67: "cell phone"
}

# Default Sample Video URL/Path (We will look for it or create a placeholder)
SAMPLE_VIDEO_PATH = "data/sample.mp4"

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/nolan/128/vision.png", width=80)
st.sidebar.header("Mission Control")

# 1. Source Selection
source_type = st.sidebar.selectbox(
    "1. Choose Input Source",
    options=["Sample Video", "Upload Video File", "Live Webcam (Local)"]
)

video_source = None
if source_type == "Sample Video":
    if os.path.exists(SAMPLE_VIDEO_PATH):
        video_source = SAMPLE_VIDEO_PATH
    else:
        st.sidebar.warning("Sample video file not found in `data/sample.mp4`. Please upload a video or run the CLI.")
        video_source = None
elif source_type == "Upload Video File":
    uploaded_file = st.sidebar.file_uploader("Upload MP4/AVI Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save temp file
        temp_path = os.path.join("temp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_source = temp_path
else:
    # Webcam index
    webcam_index = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=0, step=1)
    video_source = int(webcam_index)

# 2. Tracking Options
tracker_type = st.sidebar.selectbox(
    "2. Tracking Algorithm",
    options=["SORT (Custom)", "ByteTrack (Native)", "BoT-SORT (Native)"],
    index=0
)
tracker_name_map = {
    "SORT (Custom)": "sort",
    "ByteTrack (Native)": "bytetrack",
    "BoT-SORT (Native)": "botsort"
}
selected_tracker = tracker_name_map[tracker_type]

# 3. Model Size Options
model_size = st.sidebar.selectbox(
    "3. YOLO Model Size",
    options=["YOLOv8n (Nano - Fast)", "YOLOv8s (Small - Balanced)", "YOLOv11n (New - Optimized)"],
    index=0
)
model_name_map = {
    "YOLOv8n (Nano - Fast)": "yolov8n.pt",
    "YOLOv8s (Small - Balanced)": "yolov8s.pt",
    "YOLOv11n (New - Optimized)": "yolov11n.pt"
}
selected_model = model_name_map[model_size]

# 4. Parameters
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.35, step=0.05)
show_trails = st.sidebar.toggle("Render Trails / Historical Paths", value=True)

# 5. Class filters
# Use a lightweight default class list before the user starts processing.
if "detector_classes" not in st.session_state:
    st.session_state["detector_classes"] = DEFAULT_CLASS_NAMES
    st.session_state["current_model_name"] = selected_model

# If model selection changed before processing, keep the previous cached default
if st.session_state.get("current_model_name") != selected_model and not st.session_state.get("run_pipeline", False):
    st.session_state["detector_classes"] = DEFAULT_CLASS_NAMES
    st.session_state["current_model_name"] = selected_model

class_names_dict = st.session_state["detector_classes"]
class_options = list(class_names_dict.values())
selected_classes = st.sidebar.multiselect(
    "Filter Detection Classes",
    options=class_options,
    default=["person", "car", "motorcycle", "bicycle", "dog", "cat"] if "person" in class_options else class_options[:5]
)

# Convert string classes back to IDs
selected_class_ids = [k for k, v in class_names_dict.items() if v in selected_classes] if selected_classes else None

# Action buttons
start_btn = st.sidebar.button("🚀 Start Processing", use_container_width=True)
stop_btn = st.sidebar.button("🛑 Stop / Reset", use_container_width=True)

# Set execution flags
if "run_pipeline" not in st.session_state:
    st.session_state["run_pipeline"] = False

if start_btn:
    if video_source is None:
        st.error("Please specify a valid video source or upload a file first.")
    else:
        st.session_state["run_pipeline"] = True
        st.rerun()

if stop_btn:
    st.session_state["run_pipeline"] = False
 # ----------------- MAIN APP BODY -----------------
col_video, col_metrics = st.columns([3, 1])

with col_video:
    st.markdown("<h3 class='section-header'>Live Detection Feed</h3>", unsafe_allow_html=True)
    st.markdown("<div class='video-panel'>", unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not st.session_state["run_pipeline"]:
        # Show an attractive placeholder image or video details
        st.info("Waiting for Input... Configure the sidebar settings and click 'Start Processing' to begin.")
        
        # Display instructions/features
        st.markdown("""
        ### Features Supported:
        - **Real-Time Video Analytics**: Process files or webcam frames with high efficiency.
        - **Advanced Track Association**: Compares custom SORT Kalman prediction and Hungarian assignment with state-of-the-art Deep SORT / ByteTrack models.
        - **Export Tracking Reports**: Track individual object coordinate paths, lifespan, and speeds, then export.
        - **Modern Visualizations**: Premium overlays including corner-bracket boxes, glassy text plates, and dynamic movement trails.
        """)

# Columns for Real-time metrics
with col_metrics:
    st.markdown("<h3 class='section-header'>Performance Monitor</h3>", unsafe_allow_html=True)
    
    # Custom HTML Metric Cards
    fps_metric = st.empty()
    active_metric = st.empty()
    total_metric = st.empty()
    
    fps_metric.markdown("""
        <div class="metric-card card-1">
            <div class="metric-val">0.0</div>
            <div class="metric-title">FPS</div>
        </div>
    """, unsafe_allow_html=True)
    
    active_metric.markdown("""
        <div class="metric-card card-2">
            <div class="metric-val">0</div>
            <div class="metric-title">Objects Tracked</div>
        </div>
    """, unsafe_allow_html=True)
    
    total_metric.markdown("""
        <div class="metric-card card-3">
            <div class="metric-val">0</div>
            <div class="metric-title">Unique Objects</div>
        </div>
    """, unsafe_allow_html=True)

# Tabs for Charts and Logs
tab_charts, tab_logs = st.tabs(["📈 Real-Time Charts", "📋 Detailed Tracker Logs"])

with tab_charts:
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        chart1_placeholder = st.empty()
    with col_chart2:
        chart2_placeholder = st.empty()

with tab_logs:
    log_placeholder = st.empty()
    download_placeholder = st.empty()

# ----------------- INFERENCE PIPELINE LOOP -----------------
if st.session_state["run_pipeline"] and video_source is not None:
    # Initialize Detector lazily when processing starts
    detector = load_detector(selected_model)
    class_names = detector.get_class_names()
    st.session_state["detector_classes"] = class_names
    st.session_state["current_model_name"] = selected_model
    
    # Initialize Tracker Bridge
    tracker = TrackerBridge(
        tracker_type=selected_tracker,
        max_age=15,
        min_hits=3,
        iou_threshold=0.3
    )
    
    # Open Capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error(f"Error: Could not open video source '{video_source}'")
        st.session_state["run_pipeline"] = False
        st.rerun()
        
    fps_calc = FPSCalculator()
    unique_ids_tracker = set()
    
    # Historical lists for plotting
    time_history = []
    active_count_history = []
    class_frequency = {}
    
    # Tracking logs dataframe fields
    log_data = [] # List of dicts
    
    start_time = time.time()
    
    # Processing Loop
    frame_idx = 0
    while cap.isOpened() and st.session_state["run_pipeline"]:
        ret, frame = cap.read()
        if not ret:
            st.info("End of video file or stream reached.")
            break
            
        frame_idx += 1
        elapsed_seconds = round(time.time() - start_time, 2)
        
        # Core Update (Detection + Tracking)
        tracks = tracker.update(
            frame,
            detector,
            conf_threshold=conf_threshold,
            iou_threshold=0.45,
            classes=selected_class_ids
        )
        
        active_ids = []
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, score = track
            track_id = int(track_id)
            class_id = int(class_id)
            
            active_ids.append(track_id)
            unique_ids_tracker.add(track_id)
            
            # Label name
            class_label = class_names.get(class_id, f"Class {class_id}")
            
            # Increment class frequencies
            class_frequency[class_label] = class_frequency.get(class_label, 0) + 1
            
            # Log active tracker state
            log_data.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "Elapsed (s)": elapsed_seconds,
                "Frame": frame_idx,
                "Track ID": track_id,
                "Class": class_label,
                "Confidence": round(score, 2),
                "Center X": int((x1 + x2) / 2),
                "Center Y": int((y1 + y2) / 2),
                "Width": int(x2 - x1),
                "Height": int(y2 - y1)
            })
            
            # Bounding box color based on class name
            color = get_color_for_class(class_label, track_id=track_id)
            
            # Compute track age and alpha for fade-in effect (0.0 to 1.0 over first 5 frames = 150ms)
            age = tracker.get_track_age(track_id)
            alpha = min(1.0, age / 5.0)
            
            # Render visual trails
            if show_trails:
                trail = tracker.get_trail(track_id)
                draw_fading_trail(frame, trail, color, max_trail_len=30)
                
            # Render bounding boxes and labels with alpha fade-in
            draw_corner_rect(frame, [x1, y1, x2, y2], color, thickness=2, alpha=alpha)
            label_text = f"#{track_id} {class_label} {score:.2f}"
            draw_glassy_label(frame, label_text, (int(x1) + 2, int(y1) - 4), color, alpha=alpha)
            
        # Update FPS
        fps_calc.update()
        fps = fps_calc.get_fps()
        
        # Append history for plotting (Limit to last 100 entries to prevent memory bloating)
        time_history.append(elapsed_seconds)
        active_count_history.append(len(active_ids))
        if len(time_history) > 100:
            time_history.pop(0)
            active_count_history.pop(0)
            
        # Update Live Metric Cards
        fps_metric.markdown(f"""
            <div class="metric-card card-1">
                <div class="metric-val">{fps:.1f}</div>
                <div class="metric-title">FPS</div>
            </div>
        """, unsafe_allow_html=True)
        
        active_metric.markdown(f"""
            <div class="metric-card card-2">
                <div class="metric-val">{len(active_ids)}</div>
                <div class="metric-title">Objects Tracked</div>
            </div>
        """, unsafe_allow_html=True)
        
        total_metric.markdown(f"""
            <div class="metric-card card-3">
                <div class="metric-val">{len(unique_ids_tracker)}</div>
                <div class="metric-title">Unique Objects</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Render image to dashboard
        # Convert BGR (OpenCV default) to RGB (Streamlit default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update Charts and tables periodically (every 5 frames to reduce UI lag)
        if frame_idx % 5 == 0:
            # 1. Line Chart: Active count over time
            df_line = pd.DataFrame({
                "Elapsed Time (s)": time_history,
                "Active Object Count": active_count_history
            })
            fig_line = px.line(
                df_line, x="Elapsed Time (s)", y="Active Object Count",
                title=f"Active Object Count Over Time (Frame {frame_idx})",
                template="plotly_dark"
            )
            fig_line.update_traces(line_color="#00FFCC")
            fig_line.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            chart1_placeholder.plotly_chart(fig_line, use_container_width=True)
            
            # 2. Bar Chart: Class distribution
            if len(class_frequency) > 0:
                df_bar = pd.DataFrame({
                    "Object Class": list(class_frequency.keys()),
                    "Detection Count": list(class_frequency.values())
                }).sort_values(by="Detection Count", ascending=False)
                fig_bar = px.bar(
                    df_bar, x="Object Class", y="Detection Count",
                    title=f"Detection Frequency by Class (Frame {frame_idx})",
                    template="plotly_dark"
                )
                fig_bar.update_traces(marker_color="#1E90FF")
                fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                chart2_placeholder.plotly_chart(fig_bar, use_container_width=True)
                
            # 3. Log Table
            if len(log_data) > 0:
                df_logs = pd.DataFrame(log_data).tail(15) # Show last 15 entries
                log_placeholder.dataframe(df_logs, use_container_width=True)
                
        # Small delay to keep UI responsive
        time.sleep(0.01)
        
    cap.release()
    st.session_state["run_pipeline"] = False
    
    # Save final logs CSV export
    if len(log_data) > 0:
        df_logs_full = pd.DataFrame(log_data)
        csv_data = df_logs_full.to_csv(index=False)
        download_placeholder.download_button(
            label="📥 Download Full Tracking Log (CSV)",
            data=csv_data,
            file_name=f"tracking_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
