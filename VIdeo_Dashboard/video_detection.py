import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
import os
import time
import requests


st.set_page_config(
    page_title="Video Detection Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        padding: 0.2rem 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }
    .metric-card, .detection-box {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'person_counter' not in st.session_state:
    st.session_state.person_counter = 1
if 'person_registry' not in st.session_state:
    st.session_state.person_registry = []
if 'age_gender_cache' not in st.session_state:
    st.session_state.age_gender_cache = {}

CAFFE_MODELS = {
    "deploy_gender.prototxt": "https://github.com/raunaqness/Gender-and-Age-Detection-OpenCV-Caffe/raw/master/data/deploy_gender.prototxt",
    "gender_net.caffemodel": "https://github.com/raunaqness/Gender-and-Age-Detection-OpenCV-Caffe/raw/master/data/gender_net.caffemodel",
    "deploy_age.prototxt": "https://github.com/raunaqness/Gender-and-Age-Detection-OpenCV-Caffe/raw/master/data/deploy_age.prototxt",
    "age_net.caffemodel": "https://github.com/raunaqness/Gender-and-Age-Detection-OpenCV-Caffe/raw/master/data/age_net.caffemodel",
}

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

models_dir = os.path.join(os.path.dirname(__file__), "caffe_models")
os.makedirs(models_dir, exist_ok=True)

for filename, url in CAFFE_MODELS.items():
    dest_path = os.path.join(models_dir, filename)
    if not os.path.exists(dest_path):
        with st.spinner(f"Downloading {filename}..."):
            try:
                download_file(url, dest_path)
                st.success(f"Downloaded {filename}")
            except Exception as e:
                st.warning(f"Failed to download {filename}: {e}")

@st.cache_resource
def load_yolo_model(model_path: str):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

@st.cache_resource
def load_caffe_models(gender_proto: str, gender_model: str, age_proto: str, age_model: str):
    """Load Caffe models for age and gender detection"""
    try:
        gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
        age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        gender_list = ['Male', 'Female']
        age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        return gender_net, age_net, gender_list, age_list
    except Exception as e:
        st.error(f"Error loading Caffe models: {e}")
        return None, None, None, None

def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int] = (0, 255, 0)):
    """Draw label with background on frame"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - th - 4), (x + tw, y), color, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y - 2), font, font_scale, (0, 0, 0), thickness)

def process_frame(frame: np.ndarray, model: YOLO, detect_objects: bool, detect_persons: bool, 
                 detect_gender: bool, detect_age: bool, gender_net=None, age_net=None, 
                 gender_list=None, age_list=None, resize_dim=(416, 416)):
    """Process a single frame for detections"""
    frame_resized = cv2.resize(frame, resize_dim)
    results = model(frame_resized, verbose=False)[0]
    detections = results.boxes
    names = model.names
    
    h_ratio = frame.shape[0] / resize_dim[1]
    w_ratio = frame.shape[1] / resize_dim[0]
    
    detection_data = []
    
    for i in range(len(detections.cls)):
        label = names[int(detections.cls[i])]
        confidence = float(detections.conf[i])
        
        # Filtering logic:
        if detect_persons and label != "person":
            continue  # Only show people if 'Person Only Mode' is ON
        if not detect_persons and not detect_objects:
            pass  # If both are off, show all objects
        elif detect_objects and not detect_persons:
            pass  # Show all objects
        # If both are ON, show only people (Person Only Mode takes precedence)
        
        box = detections.xyxy[i].cpu().numpy().astype(int)
        xmin, ymin, xmax, ymax = box
        xmin = int(xmin * w_ratio)
        xmax = int(xmax * w_ratio)
        ymin = int(ymin * h_ratio)
        ymax = int(ymax * h_ratio)
        width, height = xmax - xmin, ymax - ymin
        
        bbox_center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
        matched = False
        person_label = label
        gender = age = "-"
        
        if label == 'person':
            # Person tracking logic
            for pid, center in st.session_state.person_registry:
                if abs(center[0] - bbox_center[0]) < 50 and abs(center[1] - bbox_center[1]) < 50:
                    person_label = f'person{pid}'
                    matched = True
                    break
            if not matched:
                person_label = f'person{st.session_state.person_counter}'
                st.session_state.person_registry.append((st.session_state.person_counter, bbox_center))
                st.session_state.person_counter += 1
            # Age and gender detection
            roi = frame[ymin:ymax, xmin:xmax]
            if roi.size > 0:
                if person_label in st.session_state.age_gender_cache:
                    gender, age = st.session_state.age_gender_cache[person_label]
                else:
                    try:
                        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (227, 227)), 1.0, (227, 227),
                                                     (78.4263377603, 87.7689143744, 114.895847746),
                                                     swapRB=False)
                        if detect_gender and gender_net is not None and gender_list is not None:
                            gender_net.setInput(blob)
                            gender_preds = gender_net.forward()
                            gender = gender_list[gender_preds[0].argmax()]
                        if detect_age and age_net is not None and age_list is not None:
                            age_net.setInput(blob)
                            age_preds = age_net.forward()
                            age = age_list[age_preds[0].argmax()]
                        st.session_state.age_gender_cache[person_label] = (gender, age)
                    except Exception as e:
                        gender, age = "?", "?"
        # Draw bounding box and label
        color = (255, 200, 0) if label == 'person' else (0, 255, 0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2, lineType=cv2.LINE_AA)
        label_display = f"{person_label if label == 'person' else label}"
        if label == 'person':
            if detect_gender and gender != "-":
                label_display += f" | G: {gender}"
            if detect_age and age != "-":
                label_display += f" | A: {age}"
        draw_label(frame, label_display, xmin, ymin, color)
        # Store detection data
        detection_data.append({
            'class': person_label if label == 'person' else label,
            'confidence': round(confidence, 2),
            'gender': gender,
            'age': age,
            'x': xmin,
            'y': ymin,
            'width': width,
            'height': height
        })
    return frame, detection_data

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎥 Video Detection Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Select YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    
    # Detection options
    st.sidebar.markdown("### 🔍 Detection Options")
    detect_objects = st.sidebar.checkbox("Detect All Objects", value=True)
    detect_persons = st.sidebar.checkbox("Person Only Mode", value=False)
    
    # Check if Caffe models are available for age/gender detection
    models_dir = "caffe_models"
    models_available = os.path.exists(models_dir) and all(
        os.path.exists(os.path.join(models_dir, f)) 
        for f in ["deploy_gender.prototxt", "gender_net.caffemodel", "deploy_age.prototxt", "age_net.caffemodel"]
    )
    
    if not models_available:
        detect_gender = st.sidebar.checkbox("Gender Detection", value=False, disabled=True)
        detect_age = st.sidebar.checkbox("Age Detection", value=False, disabled=True)
        st.sidebar.info("💡 Enable age/gender detection by downloading models")
    else:
        detect_gender = st.sidebar.checkbox("Gender Detection", value=False)
        detect_age = st.sidebar.checkbox("Age Detection", value=False)
    
    # Resize options
    st.sidebar.markdown("### 📐 Processing Options")
    resize_width = st.sidebar.slider("Resize Width", 320, 640, 416, 32)
    resize_height = st.sidebar.slider("Resize Height", 320, 640, 416, 32)
    resize_dim = (resize_width, resize_height)
    
    # Load models
    if model_option is None:
        st.error("Please select a YOLO model.")
        return
    
    model = load_yolo_model(model_option)
    if model is None:
        st.error("Failed to load YOLO model. Please check the model file.")
        return
    
    gender_net = age_net = gender_list = age_list = None
    if detect_gender or detect_age:
        st.sidebar.markdown("### 👥 Age/Gender Models")
        
        # Check if models directory exists
        models_dir = "caffe_models"
        if not os.path.exists(models_dir):
            st.sidebar.warning("⚠️ Caffe models not found!")
            st.sidebar.info("📥 To enable age/gender detection, run: `python download_caffe_models.py`")
            detect_gender = False
            detect_age = False
        else:
            # Use default paths in caffe_models directory
            gender_proto = f"{models_dir}/deploy_gender.prototxt"
            gender_model = f"{models_dir}/gender_net.caffemodel"
            age_proto = f"{models_dir}/deploy_age.prototxt"
            age_model = f"{models_dir}/age_net.caffemodel"
            
            # Check if files exist
            missing_files = []
            for path in [gender_proto, gender_model, age_proto, age_model]:
                if not os.path.exists(path):
                    missing_files.append(os.path.basename(path))
            
            if missing_files:
                st.sidebar.error(f"❌ Missing model files: {', '.join(missing_files)}")
                st.sidebar.info("📥 Run: `python download_caffe_models.py` to download models")
                detect_gender = False
                detect_age = False
            else:
                try:
                    gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
                    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
                    gender_list = ['Male', 'Female']
                    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                    st.sidebar.success("✅ Age/Gender models loaded automatically!")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to load models: {e}")
                    detect_gender = False
                    detect_age = False
    
    # Main content area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("### 📹 Video Input")
        
        # Video source selection
        video_source = st.radio(
            "Select Video Source",
            ["Webcam", "Upload Video File", "Video URL"],
            horizontal=True
        )
        
        if video_source == "Webcam":
            # Webcam input
            if st.button("🎥 Start Webcam Detection", type="primary"):
                st.session_state.processing = True
                st.session_state.detection_results = []
                st.session_state.person_counter = 1
                st.session_state.person_registry = []
                st.session_state.age_gender_cache = {}
                
                # Webcam processing
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Unable to open webcam")
                    return
                
                # Create placeholder for video
                video_placeholder = st.empty()
                stats_placeholder = st.empty()
                # Create stop button ONCE
                stop = st.button("⏹️ Stop Detection", key="stop_webcam")
                try:
                    while st.session_state.processing:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame
                        processed_frame, detections = process_frame(
                            frame, model, detect_objects, detect_persons,
                            detect_gender, detect_age, gender_net, age_net,
                            gender_list, age_list, resize_dim
                        )
                        
                        # Debug info
                        if detect_gender or detect_age:
                            st.sidebar.markdown(f"**Debug:** Models loaded - Gender: {gender_net is not None}, Age: {age_net is not None}")
                            if detections:
                                for det in detections:
                                    if 'person' in det['class']:
                                        st.sidebar.markdown(f"**Person:** {det['gender']} | {det['age']}")

                        # Update detection results
                        if detections:
                            st.session_state.detection_results.extend(detections)
                        
                        # Convert frame to RGB for display
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB")
                        
                        # Display stats
                        with stats_placeholder.container():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Detections", len(st.session_state.detection_results))
                            with col2:
                                st.metric("People Detected", len([d for d in st.session_state.detection_results if 'person' in d['class']]))
                            with col3:
                                st.metric("Person Counter", st.session_state.person_counter - 1)
                        time.sleep(0.1)
                        if stop:
                            st.session_state.processing = False
                            break
                finally:
                    cap.release()
        
        elif video_source == "Upload Video File":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for processing"
            )
            
            if uploaded_file is not None:
                if st.button("🎬 Process Video File", type="primary"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name
                    
                    try:
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            st.error("Unable to open video file")
                            return
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        st.info(f"Video Info: {total_frames} frames, {fps:.2f} FPS")
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Video processing
                        frame_count = 0
                        st.session_state.detection_results = []
                        st.session_state.person_counter = 1
                        st.session_state.person_registry = []
                        st.session_state.age_gender_cache = {}
                        
                        video_placeholder = st.empty()
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_count}/{total_frames}")
                            
                            # Process every 5th frame for performance
                            if frame_count % 5 == 0:
                                processed_frame, detections = process_frame(
                                    frame, model, detect_objects, detect_persons,
                                    detect_gender, detect_age, gender_net, age_net,
                                    gender_list, age_list, resize_dim
                                )
                                if detections:
                                    st.session_state.detection_results.extend(detections)
                                # Show sample frame
                                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(frame_rgb, channels="RGB")
                        
                        cap.release()
                        status_text.text("Processing complete!")
                        
                    finally:
                        # Clean up temporary file
                        os.unlink(video_path)
        
        else:  # Video URL
            video_url = st.text_input("Enter Video URL", placeholder="https://example.com/video.mp4")
            if video_url:
                if st.button("🌐 Process Video URL", type="primary"):
                    st.session_state.processing = True
                    st.session_state.detection_results = []
                    st.session_state.person_counter = 1
                    st.session_state.person_registry = []
                    st.session_state.age_gender_cache = {}

                    cap = cv2.VideoCapture(video_url)
                    if not cap.isOpened():
                        st.error("Unable to open video stream. Please check the URL.")
                        return

                    video_placeholder = st.empty()
                    stats_placeholder = st.empty()
                    stop = st.button("⏹️ Stop Detection", key="stop_url")

                    try:
                        while st.session_state.processing:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("Stream ended or unavailable.")
                                break

                            processed_frame, detections = process_frame(
                                frame, model, detect_objects, detect_persons,
                                detect_gender, detect_age, gender_net, age_net,
                                gender_list, age_list, resize_dim
                            )
                            if detections:
                                st.session_state.detection_results.extend(detections)
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(frame_rgb, channels="RGB")
                            with stats_placeholder.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Detections", len(st.session_state.detection_results))
                                with col2:
                                    st.metric("People Detected", len([d for d in st.session_state.detection_results if 'person' in d['class']]))
                                with col3:
                                    st.metric("Person Counter", st.session_state.person_counter - 1)
                            time.sleep(0.1)
                            if stop:
                                st.session_state.processing = False
                                break
                    finally:
                        cap.release()
    
    with col2:
        st.markdown("### 📊 Detection Results")
        if st.session_state.detection_results:
            total_detections = len(st.session_state.detection_results)
            people_detected = len([d for d in st.session_state.detection_results if 'person' in d['class']])
            avg_confidence = np.mean([d['confidence'] for d in st.session_state.detection_results])
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Summary</h4>
                <p><strong>Total Detections:</strong> {total_detections}</p>
                <p><strong>People Detected:</strong> {people_detected}</p>
                <p><strong>Avg Confidence:</strong> {avg_confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recent detections
            st.markdown("### 🔍 Recent Detections")
            recent_detections = st.session_state.detection_results[-3:]  # Last 3 detections
            
            for detection in recent_detections:
                st.markdown(f"""
                <div class="detection-box">
                    <p><strong>{detection['class']}</strong> (Conf: {detection['confidence']})</p>
                    <p>Gender: {detection['gender']} | Age: {detection['age']}</p>
                    <p>Position: ({detection['x']}, {detection['y']})</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Export options
            st.markdown("### 💾 Export Data")
            if st.button("📄 Export to CSV"):
                df = pd.DataFrame(st.session_state.detection_results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="detection_results.csv",
                    mime="text/csv"
                )
            
            if st.button("🗑️ Clear Results"):
                st.session_state.detection_results = []
                st.session_state.person_counter = 1
                st.session_state.person_registry = []
                st.session_state.age_gender_cache = {}
                st.rerun()
        else:
            st.info("No detections yet. Start processing a video to see results here.")

if __name__ == "__main__":
    main() 
