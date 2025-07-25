# type: ignore #Modified script to support both single file analysis and comparative assessment modes
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
import tempfile
import os
from datetime import datetime
from pyulog import ULog
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Motor Data Vantage",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants and configurations
TOPIC_ASSESSMENT_PAIRS = [
    ("vehicle_local_position", "Actualposition"),
    ("vehicle_local_position_setpoint", "Setpointposition"),
    ("vehicle_local_position_setpoint", "Thrust"),
    ("vehicle_torque_setpoint", "Torque"),
    ("px4io_status", "Control"),
    ("battery_status", "Battery"),
]

ASSESSMENT_Y_AXIS_MAP = {
    "Actualposition": ["x", "y", "z"],
    "Setpointposition": ["x", "y", "z"],
    "Thrust": ["thrust[0]", "thrust[1]", "thrust[2]", "thrust[3]", "thrust[4]", "thrust[5]"],
    "Torque": ["xyz[0]", "xyz[1]", "xyz[2]"],
    "Control": ["pwm[0]", "pwm[1]", "pwm[2]", "pwm[3]", "pwm[4]", "pwm[5]"],
    "Battery": ["voltage_v", "current_average_a", "discharged_mah"],
}

COLUMN_DISPLAY_NAMES = {
    "pwm[0]": "Motor 1 pwm",
    "pwm[1]": "Motor 2 pwm",
    "pwm[2]": "Motor 3 pwm",
    "pwm[3]": "Motor 4 pwm",
    "pwm[4]": "Motor 5 pwm",
    "pwm[5]": "Motor 6 pwm",
    "thrust[0]": "Thrust Motor 1",
    "thrust[1]": "Thrust Motor 2",
    "thrust[2]": "Thrust Motor 3",
    "thrust[3]": "Thrust Motor 4",
    "thrust[4]": "Thrust Motor 5",
    "thrust[5]": "Thrust Motor 6",
    "xyz[0]": "Torque x",
    "xyz[1]": "Torque y",
    "xyz[2]": "Torque z",
    "voltage_v": "Battery Voltage",
    "current_average_a": "Current",
    "discharged_mah": "Discharged Capacity",
}

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'files_submitted' not in st.session_state:
    st.session_state.files_submitted = False
if 'show_upload_area' not in st.session_state:
    st.session_state.show_upload_area = True
if 'upload_opened_by_plus' not in st.session_state:
    st.session_state.upload_opened_by_plus = False
if 'single_file_selection' not in st.session_state:
    st.session_state.single_file_selection = "None"
if 'selected_single_file' not in st.session_state:
    st.session_state.selected_single_file = None
if 'selected_single_file_content' not in st.session_state:
    st.session_state.selected_single_file_content = None
if 'selected_assessment' not in st.session_state:
    st.session_state.selected_assessment = "None"
if 'last_single_topic' not in st.session_state:
    st.session_state.last_single_topic = None

# Add new session state variables for the new file upload interface
if 'upload_source' not in st.session_state:
    st.session_state.upload_source = "desktop"  # desktop
if 'show_file_preview' not in st.session_state:
    st.session_state.show_file_preview = False
if 'file_rename_mode' not in st.session_state:
    st.session_state.file_rename_mode = {}
if 'file_share_mode' not in st.session_state:
    st.session_state.file_share_mode = {}
if "share_all_mode" not in st.session_state:
    st.session_state.share_all_mode = False

# Initialize global variables
selected_assessment = "None"

# --- Ensure session state keys for comparative analysis file selection are initialized at the very top of main() ---
if 'benchmark_file_selection' not in st.session_state:
    st.session_state['benchmark_file_selection'] = 'None'
if 'target_file_selection' not in st.session_state:
    st.session_state['target_file_selection'] = 'None'
if 'selected_bench' not in st.session_state:
    st.session_state['selected_bench'] = 'None'
if 'selected_val' not in st.session_state:
    st.session_state['selected_val'] = 'None'
if 'selected_bench_content' not in st.session_state:
    st.session_state['selected_bench_content'] = None
if 'selected_val_content' not in st.session_state:
    st.session_state['selected_val_content'] = None
if 'selected_assessment' not in st.session_state:
    st.session_state['selected_assessment'] = 'None'
if 'b_df' not in st.session_state:
    st.session_state['b_df'] = None
if 'v_df' not in st.session_state:
    st.session_state['v_df'] = None
if 'mode_switched' not in st.session_state:
    st.session_state['mode_switched'] = False

# Function to change page
def change_page(page):
    if page == 'home':
        # Store the current analysis type and data source before going back
        st.session_state.previous_analysis_type = st.session_state.analysis_type
        st.session_state.previous_data_source = st.session_state.data_source
    st.session_state.current_page = page

# Utility functions
def load_csv(file):
    if isinstance(file, str):
        with open(file, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(10)]
        file_obj = file
    else:
        file.seek(0)
        lines = [file.readline().decode('utf-8') for _ in range(10)]
        file.seek(0)
        file_obj = file
    # Find the header row index
    header_row = None
    for i, line in enumerate(lines):
        if "TestrecordId" in line or "Timestamp (hh:mm:ss)" in line or "Speedmms" in line:
            header_row = i
            break
    if header_row is None:
        header_row = 4
    try:
        return pd.read_csv(file_obj, encoding='utf-8', skiprows=header_row, header=0)
    except Exception:
        if not isinstance(file, str):
            file.seek(0)
        return pd.read_csv(file_obj, encoding='utf-8-sig', skiprows=header_row, header=0)

def load_ulog(file, key_suffix=""):
    ALLOWED_TOPICS = set(t for t, _ in TOPIC_ASSESSMENT_PAIRS)
    
    # Create a temporary file to store the content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ulg') as tmp_file:
        try:
            # If file is a string (path), read directly
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    tmp_file.write(f.read())
            else:
                # If file is a file object, write its content
                file.seek(0)
                tmp_file.write(file.read())
            tmp_file.flush()
            
            # Process the ULog file
            ulog = ULog(tmp_file.name)
            if not ulog.data_list:
                st.warning("⚠️ No data found in the ULog file")
                return {}, []
                
            extracted_dfs = {}
            for msg in ulog.data_list:
                if msg.data:  # Only process messages with data
                    df = pd.DataFrame(msg.data)
                    if not df.empty:
                        extracted_dfs[msg.name] = df
            
            filtered_dfs = {topic: df for topic, df in extracted_dfs.items() if topic in ALLOWED_TOPICS}
            if not filtered_dfs:
                st.warning("⚠️ No extractable topics found in ULog file")
                return {}, []
                
            topic_names = ["None"] + list(filtered_dfs.keys())
            return filtered_dfs, topic_names
            
        except Exception as e:
            st.error(f"Error processing ULog file: {str(e)}")
            return {}, []
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_file.name)
            except:
                pass

def get_axis_title(axis_name):
    if axis_name == 'timestamp_seconds':
        return 'TIME(secs)'
    return axis_name

def get_timestamp_ticks(data):
    """Generate evenly spaced timestamp ticks."""
    if data is None or len(data) == 0:
        return [], []
    try:
        data_min = float(data.min())
        data_max = float(data.max())
        data_range = data_max - data_min
        spacing = get_tick_spacing(data_range)
        ticks = np.arange(data_min, data_max + spacing, spacing)
        return ticks, [format_seconds_to_mmss(float(t)) for t in ticks]
    except Exception as e:
        st.error(f"Error generating timestamp ticks: {str(e)}")
        return [], []

def get_tick_spacing(data_range):
    """Get appropriate tick spacing based on data range."""
    if data_range <= 10:
        return 1
    elif data_range <= 60:
        return 10
    elif data_range <= 300:
        return 30
    elif data_range <= 600:
        return 60
    else:
        return 120

def format_seconds_to_mmss(seconds):
    """Format seconds to MM:SS format."""
    try:
        minutes = int(float(seconds) // 60)
        remaining_seconds = int(float(seconds) % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    except Exception as e:
        st.error(f"Error formatting seconds: {str(e)}")
        return "00:00"

def mmss_to_seconds(mmss_str):
    """Convert MM:SS format to seconds."""
    try:
        if ':' in mmss_str:
            minutes, seconds = mmss_str.split(':')
            return int(minutes) * 60 + int(seconds)
        else:
            return float(mmss_str)
    except:
        return 0.0

def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format."""
    try:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"
    except:
        return "00:00"

def ensure_seconds_column(df):
    """Ensure timestamp_seconds column exists in the dataframe."""
    if 'timestamp_seconds' not in df.columns:
        if 'timestamp' in df.columns:
            df['timestamp_seconds'] = df['timestamp'] / 1e6  # Convert microseconds to seconds
        elif 'Timestamp (hh:mm:ss)' in df.columns:
            df['timestamp_seconds'] = df['Timestamp (hh:mm:ss)'].apply(lambda x: mmss_to_seconds(str(x)))
        else:
            # Create a simple index-based timestamp
            df['timestamp_seconds'] = range(len(df))
    
    # Check if timestamp_seconds column has valid data (not all NaN)
    if 'timestamp_seconds' in df.columns:
        if df['timestamp_seconds'].isna().all() or df['timestamp_seconds'].isnull().all():
            df = df.drop('timestamp_seconds', axis=1)
    
    return df

def load_data(file, filetype, key_suffix):
    """Load data from various file types."""
    try:
        if filetype == ".csv":
            df = load_csv(file)
        elif filetype == ".xlsx":
            df = pd.read_excel(file)
        else:
            st.error(f"Unsupported file type: {filetype}")
            return None, None
        
        if df is not None and not df.empty:
            df = ensure_seconds_column(df)
            return df, None
        else:
            st.error("No data found in file")
            return None, None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def get_display_name(col):
    """Get a display-friendly name for a column."""
    if col == 'timestamp_seconds':
        return 'Time (secs)'
    elif col == 'Index':
        return 'Index'
    else:
        return col

# Main application
def main():
    # Fixed header with improved structure
    st.markdown("""
    <style>
    .fixed-header {
        top: 18px;
        left: 18px;
        z-index: 1001;
        background: #fff;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(44, 62, 80, 0.10);
        padding: 16px 28px 14px 22px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        min-width: 260px;
        max-width: 380px;
        border: 1px solid #e0e0e0;
    }
    .fixed-header h1 {
        color: #2E86C1;
        margin: 0 0 2px 0;
        font-size: 1.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        line-height: 1.1;
        font-weight: 700;
    }
    .fixed-header .rocket-icon {
        font-size: 1.7rem;
        line-height: 1;
    }
    .fixed-header p {
        color: #666;
        margin: 0;
        font-size: 0.98rem;
        line-height: 1.2;
        font-weight: 400;
    }
    /* Add padding to main content to prevent overlap with fixed header */
    .main .block-container {
        padding-top: 40px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fixed-header">
        <h1><span class="rocket-icon">🚀</span> Motor Data Vantage </h1>
        <!-- <p>Advanced data analysis and visualization platform</p> -->
    </div>
    """, unsafe_allow_html=True)

    # File Upload Section
    if st.session_state.show_upload_area:
        st.markdown("""
        <style>
        .upload-section {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 2px solid #e9ecef;
            transition: all 0.3s ease;
        }
        .upload-section.active {
            border-color: #007bff;
            background: #f0f8ff;
            box-shadow: 0 4px 12px rgba(0, 123, 255, 0.15);
        }
        .upload-section:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }
        .file-preview-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        .file-preview-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-color: #007bff;
            transform: translateY(-1px);
        }
        .file-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        .file-action-btn {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .file-action-btn:hover {
            background: #e9ecef;
            border-color: #adb5bd;
        }
        .file-action-btn.primary {
            background: #007bff;
            color: white;
            border-color: #007bff;
        }
        .file-action-btn.primary:hover {
            background: #0056b3;
        }
        .file-action-btn.danger {
            background: #dc3545;
            color: white;
            border-color: #dc3545;
        }
        .file-action-btn.danger:hover {
            background: #c82333;
        }
        .upload-zone {
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            background: white;
            transition: all 0.2s;
            cursor: pointer;
        }
        .upload-zone:hover {
            border-color: #007bff;
            background: #f8f9ff;
            transform: scale(1.02);
        }
        .upload-zone.dragover {
            border-color: #007bff;
            background: #e3f2fd;
        }
        .file-stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .file-stats h6 {
            color: rgba(255,255,255,0.9);
            margin-bottom: 8px;
        }
        .file-stats .stat-value {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .file-stats .stat-label {
            font-size: 12px;
            opacity: 0.8;
        }
        .bulk-actions {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .bulk-actions h6 {
            color: #495057;
            margin-bottom: 10px;
        }
        .tab-content {
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .file-type-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .file-type-badge.csv {
            background: #d4edda;
            color: #155724;
        }
        .file-type-badge.ulg {
            background: #d1ecf1;
            color: #0c5460;
        }
        .video-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .video-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 12px;
            color: white;
        }
        .video-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: white;
        }
        .video-description {
            font-size: 0.9rem;
            opacity: 0.9;
            color: white;
            margin-bottom: 12px;
        }
        .video-placeholder {
            padding: 20px;
            text-align: center;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            color: white;
        }
        .upload-section-enhanced {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        .github-section-enhanced {
            background: linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #b3d8fd;
            box-shadow: 0 2px 8px rgba(0, 123, 255, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: center; color: #2E86C1; margin-bottom: 30px;'>📁 File Management</h3>", unsafe_allow_html=True)
        github_col, upload_col, video_col = st.columns([0.35, 0.35, 0.3])
        with github_col:
            st.markdown("""
            <div class="github-section-enhanced">
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 8px;'>
                    <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='22' style='margin-right: 4px;'/>
                    <span style='font-size: 1.08rem; font-weight: 700; color: #24292f;'>GitHub</span>
                    <span style='font-size: 0.98rem; color: #2980b9; margin-left: 6px;'>(<a style='color:#2980b9; text-decoration:underline; cursor:pointer;' href='#'>.csv, .ulg</a>)</span>
                </div>
                <div style='font-size: 0.98rem; color: #444; margin-bottom: 12px;'>
                    Paste a <b>GitHub <span style='font-weight:700;'>raw/blob/folder URL</span></b> to fetch files.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.get("clear_github_url_input", False):
                st.session_state.github_url_input = ""
                st.session_state.clear_github_url_input = False
            github_col, fetch_col  = st.columns([5, 1])
            with fetch_col:
                fetch_github = st.button("Fetch", key="fetch_github_btn", use_container_width=True)
            with github_col:
                github_url = st.text_input("GitHub URL (raw, blob, or folder)", key="github_url_input", label_visibility="collapsed", placeholder="e.g. https://github.com/user/repo/blob/main/data.csv")
            if fetch_github and github_url:
                st.warning("GitHub fetch is a placeholder in this demo. Implement fetch logic as in Final.py if needed.")
        with upload_col:
            # File uploader with increased height
            st.markdown("""
            <style>
            section[data-testid="stFileUploader"] > div {
                min-height: 120px !important;
                height: 120px !important;
                display: flex;
                align-items: center;
            }
            </style>
            """, unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Choose files to upload", 
                type=["csv", "ulg"], 
                key="desktop_uploader", 
                label_visibility="collapsed", 
                accept_multiple_files=True,
                help="Drag and drop files here or click to browse")
        with video_col:            
            # Video player
            try:
                with open("streamlit-assessment_dashboard.mp4", "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes, start_time=0)
            except FileNotFoundError:
                st.markdown("""
                <div class="video-placeholder">
                    <div style='font-size: 2rem; margin-bottom: 10px;'>🎥</div>
                    <div style='font-size: 0.9rem; margin-bottom: 5px;'>Tutorial video not found</div>
                    <div style='font-size: 0.8rem; opacity: 0.8;'>Add 'streamlit-assessment_dashboard.webm' to your project</div>
                </div>
                """, unsafe_allow_html=True)
        # Process uploaded files
        if uploaded_files:
            new_files_added = False
            existing_names = [f.name for f in st.session_state.uploaded_files]
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in existing_names:
                    st.session_state.uploaded_files.append(uploaded_file)
                    new_files_added = True
            if new_files_added:
                st.rerun()  # Force UI refresh to match current analysis mode
        # Show file preview section if files are uploaded
        if st.session_state.uploaded_files:
            st.markdown("<h4 style='margin-top: 0px; color: #495057;'>📋 File Preview & Management</h4>", unsafe_allow_html=True)
            
            # Two-column layout for file preview
            preview_col, actions_col = st.columns([0.7, 0.3])
            
            with preview_col:
                st.markdown("<h5 style='color: #6c757d; margin-bottom: 15px;'>📎 Uploaded Files</h5>", unsafe_allow_html=True)
                for i, file in enumerate(st.session_state.uploaded_files):
                    file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                    file_type_badge = f"<span class='file-type-badge {file_ext}'>{file_ext}</span>"
                    
                    # Use columns to align file name/details and action buttons in a single row
                    file_cols = st.columns([12, 1, 1, 1, 1, 1])  # Add extra column for preview
                    with file_cols[0]:
                        st.markdown(f"""
                        <div style="font-weight: 600; color: #495057;">
                            📄 {file.name}
                            <span style="font-size: 12px; color: #6c757d; margin-left: 10px;">
                                Size: {file.size / (1024*1024):.1f} MB | Type: {file.type or 'Unknown'} {file_type_badge}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    with file_cols[1]:
                        if st.button("🔍", key=f"preview_btn_{i}", use_container_width=True, help="Quick Preview"):
                            st.session_state[f"preview_mode_{i}"] = not st.session_state.get(f"preview_mode_{i}", False)
                            st.rerun()
                    with file_cols[2]:
                        if st.button("✏️", key=f"rename_btn_{i}", use_container_width=True, help="Rename"):
                            st.session_state.file_rename_mode[i] = not st.session_state.file_rename_mode.get(i, False)
                            st.rerun()
                    with file_cols[3]:
                        if st.button("➦", key=f"share_btn_{i}", use_container_width=True, help="Share"):
                            st.session_state.file_share_mode[i] = not st.session_state.file_share_mode.get(i, False)
                            st.rerun()
                    with file_cols[4]:
                        file.seek(0)
                        st.download_button(
                            label="⬇️",
                            data=file.read(),
                            file_name=file.name,
                            mime=file.type or "application/octet-stream",
                            key=f"download_btn_{i}",
                            use_container_width=True,
                            help="Download"
                        )
                        file.seek(0)
                    with file_cols[5]:
                        if st.button("🗑️", key=f"remove_btn_{i}", use_container_width=True, help="Remove"):
                            st.session_state.uploaded_files.pop(i)
                            st.rerun()
                    # Quick Preview UI
                    if st.session_state.get(f"preview_mode_{i}", False):
                        with st.expander("Quick Preview", expanded=True):
                            file.seek(0)
                            file_ext = file.name.split('.')[-1].lower() if '.' in file.name else 'unknown'
                            if file_ext == "csv":
                                try:
                                    df = pd.read_csv(file, nrows=5)
                                    st.dataframe(df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Could not preview CSV: {e}")
                            elif file_ext == "ulg":
                                size_mb = file.size / (1024 * 1024)
                                st.info("ULG preview: Only file name, size, and type shown.")
                                st.write({"Name": file.name, "Size (MB)": f"{size_mb:.2f} MB", "Type": file.type})
                            else:
                                st.warning("Preview not supported for this file type.")
                            file.seek(0)
                    if st.session_state.file_rename_mode.get(i, False):
                        with st.container():
                            st.markdown("**✏️ Rename File**")
                            col_rename1, col_rename2, col_rename3 = st.columns([2, 1, 1])
                            with col_rename1:
                                new_name = st.text_input(
                                    "New file name:", 
                                    value=file.name,
                                    key=f"rename_input_{i}"
                                )
                            with col_rename2:
                                if st.button("✅ Save", key=f"save_rename_{i}", use_container_width=True):
                                    if new_name and new_name != file.name:
                                        file.name = new_name
                                        st.success(f"File renamed to: {new_name}")
                                    st.session_state.file_rename_mode[i] = False
                                    st.rerun()
                            with col_rename3:
                                if st.button("❌ Cancel", key=f"cancel_rename_{i}", use_container_width=True):
                                    st.session_state.file_rename_mode[i] = False
                                    st.rerun()
                    if st.session_state.file_share_mode.get(i, False):
                        with st.container():
                            st.markdown("**➦ Share File**")
                            share_option = st.selectbox(
                                "Select sharing option:",
                                ["Public Link", "Email"],
                                key=f"share_option_{i}"
                            )
                            email_address = None
                            if share_option == "Email":
                                st.info(f"🔧 {share_option} sharing is a Work in Progress.")
                            elif share_option in ["Public Link"]:
                                st.info(f"🔧 {share_option} sharing is a Work in Progress.")
            with actions_col:
                total_files = len(st.session_state.uploaded_files)
                total_size = sum(f.size for f in st.session_state.uploaded_files) / (1024*1024) # MB
                
                st.markdown(f"""
                <div class="file-stats">
                    <h6>📊 File Statistics</h6>
                    <div class="stat-value">{total_files}</div>
                    <div class="stat-label">Total Files</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="file-stats">
                    <h6>💾 Storage</h6>
                    <div class="stat-value">{total_size:.1f}</div>
                    <div class="stat-label">Total Size (MB)</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("➦ Share All", use_container_width=True):
                    st.session_state.share_all_mode = not st.session_state.get("share_all_mode", False)

                if st.session_state.get("share_all_mode", False):
                    st.info(f"🔧 Share All Files is a Work in Progress.")
                
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.uploaded_files.clear()
                    st.rerun()

                # Submit files button with enhanced styling
                if st.button("✅ Submit Files for Analysis", type="primary", use_container_width=True):
                    st.session_state.files_submitted = True
                    st.session_state.show_upload_area = False
                    st.session_state.upload_opened_by_plus = False
                    st.rerun()
        else:
            # Empty state
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size: 48px; margin-bottom: 20px;">📁</div>
                <h4 style="color: #6c757d; margin-bottom: 10px;">No files uploaded yet</h4>
                <p style="color: #adb5bd; margin-bottom: 20px;">Upload your CSV or ULG files to begin analysis</p>
                <p style="font-size: 12px; color: #ced4da;">Supported formats: .csv, .ulg</p>
            </div>
            """, unsafe_allow_html=True)

    # Analysis Type Selection - only show after files are submitted
    if st.session_state.files_submitted:
        col1, col2 = st.columns([8, 0.75])
        with col1:
            analysis_type = st.radio(
                "Choose the type of analysis you want to perform",
                ["Single File Analysis", "Comparative Analysis"],
                index=0,
                horizontal=True
            )
        with col2:
            if st.session_state.files_submitted and not st.session_state.show_upload_area:
                if st.button("➕ UPLOAD", type="primary", use_container_width=True, help="Upload or manage files"):
                    st.session_state.show_upload_area = True
                    st.session_state.upload_opened_by_plus = True
                    st.rerun()
        st.session_state.analysis_type = analysis_type

        if analysis_type == "Single File Analysis":
            # Single File Analysis
            file_ext = None
            file_options = ["None"] + [f.name for f in st.session_state.uploaded_files]
            if st.session_state.single_file_selection not in file_options:
                st.session_state.single_file_selection = "None"
                st.session_state.selected_single_file = None
            selected_file = st.session_state.single_file_selection
            if selected_file != "None":
                file_ext = os.path.splitext(selected_file)[-1].lower()
            # Dynamic columns: 3 for ulg, 2 for csv
            if file_ext == ".ulg":
                col1, col_mid, col2 = st.columns([0.9, 0.25, 1])
            else:
                col1, col2 = st.columns([1, 1])
                col_mid = None
            with col1:
                new_selected_file = st.selectbox(
                    "Select File", 
                    file_options,
                    key="file_selector",
                    index=file_options.index(st.session_state.single_file_selection) if st.session_state.single_file_selection in file_options else 0
                )
                if new_selected_file != st.session_state.single_file_selection:
                    st.session_state.single_file_selection = new_selected_file
                    st.session_state.selected_single_file = new_selected_file if new_selected_file != "None" else None
                    st.session_state["previous_file"] = new_selected_file
                    st.rerun()
                selected_file = st.session_state.single_file_selection
                st.session_state.selected_single_file = selected_file if selected_file != "None" else None
                # Reset Y-axis when switching files
                if selected_file != st.session_state.get("previous_file", "None"):
                    if "x_min_grid_mmss" in st.session_state:
                        del st.session_state["x_min_grid_mmss"]
                    if "x_max_grid_mmss" in st.session_state:
                        del st.session_state["x_max_grid_mmss"]
                    if "x_min_grid" in st.session_state:
                        del st.session_state["x_min_grid"]
                    if "x_max_grid" in st.session_state:
                        del st.session_state["x_max_grid"]
                    if "reset_x_pressed" in st.session_state:
                        del st.session_state["reset_x_pressed"]
                    st.session_state["previous_file"] = selected_file
                if selected_file != "None" and st.session_state.uploaded_files:
                    try:
                        file = [f for f in st.session_state.uploaded_files if f.name == selected_file][0]
                        file.seek(0)
                        st.session_state.selected_single_file_content = file.read()
                        file.seek(0)
                    except Exception as e:
                        st.error("Error loading file")
                if selected_file == "None":
                    st.info("📋 Please select a file to begin Single File Analysis")
                    st.stop()
            
                motor_type = None
                if file_ext == ".ulg" and col_mid is not None:
                    with col_mid:
                        motor_type = st.radio(
                            "Drone Type",
                            ["Quad", "Hexa"],
                            index=0,  # Default to Quad
                            key="motor_type_single",
                            horizontal=True
                        )
                
                with col2:
                    topic_options = []
                    loaded_data = None
                    if selected_file != "None":
                        file_content = st.session_state.get('selected_single_file_content')
                        if file_content:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                                if isinstance(file_content, str):
                                    tmp_file.write(file_content.encode('utf-8'))
                                else:
                                    tmp_file.write(file_content)
                                tmp_file.flush()
                                try:
                                    if file_ext == ".ulg":
                                        dfs, topics = load_ulog(tmp_file.name)
                                        loaded_data = dfs
                                        topic_options = [a for _, a in TOPIC_ASSESSMENT_PAIRS if _ in dfs]
                                    else:
                                        df, _ = load_data(tmp_file.name, file_ext, "")
                                        loaded_data = df
                                        topic_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'timestamp_seconds']
                                except Exception as e:
                                    st.error(f"Error processing file: {str(e)}")
                                finally:
                                    try:
                                        os.unlink(tmp_file.name)
                                    except:
                                        pass
                
                    if file_ext == ".ulg":
                        selected_topic = st.selectbox(
                            "Select Topic",
                            topic_options,
                            key="ulg_topic_single"
                        )
                    else:
                        selected_topics = st.multiselect(
                            "Select up to 4 columns to plot",
                            topic_options,
                            default=topic_options[:1],
                            max_selections=4,
                            key="multi_col_select"
                        )
                
                    # Prepare data for plotting
                    plot_data = []
                    filtered_dfs_for_data_tab = []
                    if file_ext == ".ulg":
                        assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
                        topic = assessment_to_topic.get(selected_topic)
                        if topic and topic in loaded_data:
                            df = loaded_data[topic].copy()
                            df = ensure_seconds_column(df)
                            y_cols = [col for col in ASSESSMENT_Y_AXIS_MAP.get(selected_topic, []) if col in df.columns]
                            if not y_cols:
                                st.info("No relevant columns found for this topic.")
                                st.stop()
                            filtered_df = df[['timestamp_seconds'] + y_cols].dropna()
                            
                            # Convert to relative time starting from 00:00
                            if 'timestamp_seconds' in filtered_df.columns and len(filtered_df) > 0:
                                min_time = filtered_df['timestamp_seconds'].min()
                                filtered_df['timestamp_seconds'] = filtered_df['timestamp_seconds'] - min_time
                            
                            stats = filtered_df[y_cols].agg(['mean', 'min', 'max']).T
                            plot_data = [{
                                "df": filtered_df,
                                "y_col": y_col,
                                "stats": stats.loc[y_col],
                                "topic": selected_topic
                            } for y_col in y_cols]
                            filtered_dfs_for_data_tab = [filtered_df]
                        else:
                            st.info("Select a valid topic to plot.")
                            st.stop()
                    else:
                        for topic in selected_topics:
                            df = loaded_data.copy() if loaded_data is not None else None
                            if df is not None:
                                df = ensure_seconds_column(df)
                                y_cols = [topic]
                                # For CSV files, always use index as x-axis
                                filtered_df = df[y_cols].dropna().copy()
                                filtered_df['Index'] = filtered_df.index
                                x_col = 'Index'
                                stats = filtered_df[y_cols].agg(['mean', 'min', 'max']).T
                                plot_data.append({
                                    "df": filtered_df,
                                    "y_cols": y_cols,
                                    "stats": stats,
                                    "topic": topic,
                                    "x_col": x_col
                                })
                                filtered_dfs_for_data_tab.append(filtered_df)
                            else:
                                plot_data.append(None)
                                filtered_dfs_for_data_tab.append(None)
            # Tabs for Plot and Data
            tab_plot, tab_data = st.tabs(["📊 PLOT", "📋 DATA"])
            with tab_plot:
                # Create main layout with plots on left and parameters on right
                plot_col, param_col = st.columns([8.5, 2.5])
                
                with param_col:
                    # Parameter controls section
                    st.markdown("#### 📝 Parameters")
                    
                    # Get actual min/max values for axis limits
                    x_min_actual = 0.0
                    x_max_actual = 100.0
                    
                    if plot_data and len(plot_data) > 0:
                        # Get x-axis range from first plot data
                        first_df = plot_data[0]["df"]
                        if 'timestamp_seconds' in first_df.columns:
                            x_min_actual = float(first_df['timestamp_seconds'].min())
                            x_max_actual = float(first_df['timestamp_seconds'].max())
                        elif 'Index' in first_df.columns:
                            # For CSV files, use Index column
                            x_min_actual = float(first_df['Index'].min())
                            x_max_actual = float(first_df['Index'].max())
                        else:
                            # Fallback to index-based values
                            x_min_actual = 0.0
                            x_max_actual = float(len(first_df) - 1)
                    
                    # Z-Score threshold for anomaly detection
                    st.markdown("<span style='font-size:0.9rem; color:#444; font-weight:500;'>Z-Score Threshold</span>", unsafe_allow_html=True)
                    z_threshold = st.slider("", 1.0, 5.0, 3.0, 0.1, help="Threshold for detecting abnormal data points", label_visibility="collapsed")
                    
                    # X-Axis limits
                    st.markdown(f"<span style='font-size:0.9rem; color:#444; font-weight:500;'>X-Axis Limits</span>", unsafe_allow_html=True)
                    x_min_col, x_max_col, x_reset_col = st.columns([6, 6, 2])
                    with x_min_col:
                        if file_ext == ".ulg":
                            # Check if reset was pressed
                            if st.session_state.get("reset_x_pressed", False):
                                x_min_default = seconds_to_mmss(x_min_actual)
                                st.session_state["reset_x_pressed"] = False
                            else:
                                x_min_default = st.session_state.get("x_min_grid_mmss", seconds_to_mmss(x_min_actual))
                            st.markdown("<span style='font-size:0.8rem; color:#666;'>Start (MM:SS)</span>", unsafe_allow_html=True)
                            x_min = st.text_input("", value=x_min_default, key="x_min_grid_mmss", label_visibility="collapsed")
                            # Convert MM:SS to seconds for processing
                            x_min_seconds = mmss_to_seconds(x_min) if x_min else x_min_actual
                        else:
                            # Check if reset was pressed
                            if st.session_state.get("reset_x_pressed", False):
                                x_min_default = float(x_min_actual)
                                st.session_state["reset_x_pressed"] = False
                            else:
                                x_min_default = st.session_state.get("x_min_grid", float(x_min_actual))
                            st.markdown("<span style='font-size:0.8rem; color:#666;'>Start</span>", unsafe_allow_html=True)
                            x_min = st.number_input("", value=x_min_default, format="%.2f", key="x_min_grid", step=1.0, label_visibility="collapsed")
                            x_min_seconds = float(x_min)
                    with x_max_col:
                        if file_ext == ".ulg":
                            # Check if reset was pressed
                            if st.session_state.get("reset_x_pressed", False):
                                x_max_default = seconds_to_mmss(x_max_actual)
                                st.session_state["reset_x_pressed"] = False
                            else:
                                x_max_default = st.session_state.get("x_max_grid_mmss", seconds_to_mmss(x_max_actual))
                            st.markdown("<span style='font-size:0.8rem; color:#666;'>End (MM:SS)</span>", unsafe_allow_html=True)
                            x_max = st.text_input("", value=x_max_default, key="x_max_grid_mmss", label_visibility="collapsed")
                            # Convert MM:SS to seconds for processing
                            x_max_seconds = mmss_to_seconds(x_max) if x_max else x_max_actual
                        else:
                            # Check if reset was pressed
                            if st.session_state.get("reset_x_pressed", False):
                                x_max_default = float(x_max_actual)
                                st.session_state["reset_x_pressed"] = False
                            else:
                                x_max_default = st.session_state.get("x_max_grid", float(x_max_actual))
                            st.markdown("<span style='font-size:0.8rem; color:#666;'>End</span>", unsafe_allow_html=True)
                            x_max = st.number_input("", value=x_max_default, format="%.2f", key="x_max_grid", step=1.0, label_visibility="collapsed")
                            x_max_seconds = float(x_max)
                    with x_reset_col:
                        st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
                        if st.button("↺", key="reset_x_grid", help="Reset X-axis range"):
                            # Set flag to reset values
                            st.session_state["reset_x_pressed"] = True
                            # Clear the stored values
                            if "x_min_grid_mmss" in st.session_state:
                                del st.session_state["x_min_grid_mmss"]
                            if "x_max_grid_mmss" in st.session_state:
                                del st.session_state["x_max_grid_mmss"]
                            if "x_min_grid" in st.session_state:
                                del st.session_state["x_min_grid"]
                            if "x_max_grid" in st.session_state:
                                del st.session_state["x_max_grid"]
                            st.rerun()
                    
                    # Abnormal Points Summary
                    st.markdown("---")
                    st.markdown("#### 🔴 Anomaly Detection")
                    
                    if plot_data and len(plot_data) > 0:
                        total_abnormal = 0
                        total_points = 0
                        abnormal_details = []
                        
                        for idx, pdata in enumerate(plot_data):
                            if pdata is not None:
                                df = pdata["df"]
                                if "y_col" in pdata:
                                    # ULG file structure
                                    y_col = pdata["y_col"]
                                    y_cols = [y_col]
                                    plot_name = f"{pdata.get('topic', 'Data')} - {y_col}"
                                else:
                                    # Non-ULG file structure
                                    y_cols = pdata["y_cols"]
                                    plot_name = pdata.get("topic", f"Plot {idx+1}")
                                
                                plot_abnormal_count = 0
                                plot_total_points = 0
                                
                                for y_col in y_cols:
                                    if y_col in df.columns:
                                        # Calculate z-scores
                                        mean_val = df[y_col].mean()
                                        std_val = df[y_col].std()
                                        if std_val > 0:
                                            z_scores = np.abs((df[y_col] - mean_val) / std_val)
                                            abnormal_mask = z_scores > z_threshold
                                            plot_abnormal_count += abnormal_mask.sum()
                                            plot_total_points += len(df[y_col])
                                
                                if plot_abnormal_count > 0:
                                    percentage = (plot_abnormal_count / plot_total_points) * 100 if plot_total_points > 0 else 0
                                    abnormal_details.append({
                                        "name": plot_name,
                                        "count": plot_abnormal_count,
                                        "total": plot_total_points,
                                        "percentage": percentage
                                    })
                                    total_abnormal += plot_abnormal_count
                                    total_points += plot_total_points
                        
                        if total_abnormal > 0:
                            overall_percentage = (total_abnormal / total_points) * 100 if total_points > 0 else 0
                            
                            # Summary box
                            st.markdown(f"""
                            <div style='background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 10px; margin: 6px 0;'>
                                <div style='font-weight: bold; color: #856404; margin-bottom: 6px; font-size: 0.9rem;'>
                                    📊 Summary (Z-Score > {z_threshold})
                                </div>
                                <div style='font-size: 0.95rem; color: #d63031;'>
                                    <strong>{total_abnormal:,}</strong> abnormal points out of <strong>{total_points:,}</strong> total
                                </div>
                                <div style='font-size: 0.8rem; color: #6c5ce7;'>
                                    <strong>{overall_percentage:.1f}%</strong> of data points are abnormal
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detailed breakdown
                            st.markdown("**📋 Breakdown by Plot:**")
                            for idx, detail in enumerate(abnormal_details):
                                severity_color = "#e74c3c" if detail["percentage"] > 5 else "#f39c12" if detail["percentage"] > 2 else "#27ae60"
                                
                                # Create columns for the breakdown and button
                                col1, col2 = st.columns([4, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style='margin: 4px 0; padding: 6px; border-left: 3px solid {severity_color}; background-color: #f8f9fa;'>
                                        <div style='font-weight: 600; color: #2c3e50;'>{detail["name"]}</div>
                                        <div style='font-size: 0.9em; color: #7f8c8d;'>
                                            {detail["count"]:,} abnormal / {detail["total"]:,} total ({detail["percentage"]:.1f}%)
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    if st.button("📊 View", key=f"view_abnormal_{idx}", help="View abnormal data points"):
                                        st.session_state[f"show_abnormal_table_{idx}"] = not st.session_state.get(f"show_abnormal_table_{idx}", False)
                                        st.rerun()
                                
                                # Show abnormal data table if button is pressed
                                if st.session_state.get(f"show_abnormal_table_{idx}", False):
                                    st.markdown("**🔍 Abnormal Data Points:**")
                                    
                                    # Find the corresponding plot data
                                    plot_idx = None
                                    for i, pdata in enumerate(plot_data):
                                        if pdata is not None:
                                            if "y_col" in pdata:
                                                plot_name = f"{pdata.get('topic', 'Data')} - {pdata['y_col']}"
                                            else:
                                                plot_name = pdata.get("topic", f"Plot {i+1}")
                                            
                                            if plot_name == detail["name"]:
                                                plot_idx = i
                                                break
                                    
                                    if plot_idx is not None:
                                        pdata = plot_data[plot_idx]
                                        df = pdata["df"]
                                        
                                        if "y_col" in pdata:
                                            # ULG file structure
                                            y_col = pdata["y_col"]
                                            y_cols = [y_col]
                                            x_col = "timestamp_seconds"
                                        else:
                                            # Non-ULG file structure
                                            y_cols = pdata["y_cols"]
                                            x_col = pdata.get("x_col", "timestamp_seconds")
                                        
                                        # Collect abnormal data points
                                        abnormal_data = []
                                        for y_col in y_cols:
                                            if y_col in df.columns:
                                                mean_val = df[y_col].mean()
                                                std_val = df[y_col].std()
                                                if std_val > 0:
                                                    z_scores = np.abs((df[y_col] - mean_val) / std_val)
                                                    abnormal_mask = z_scores > z_threshold
                                                    
                                                    if abnormal_mask.any():
                                                        abnormal_df = df[abnormal_mask].copy()
                                                        abnormal_df['Z_Score'] = z_scores[abnormal_mask]
                                                        
                                                        # Select relevant columns for display (without Column)
                                                        display_cols = [x_col, y_col, 'Z_Score']
                                                        abnormal_df_display = abnormal_df[display_cols].copy()
                                                        
                                                        # Format timestamp for ULG files
                                                        if file_ext == ".ulg" and x_col == "timestamp_seconds":
                                                            abnormal_df_display['Time'] = abnormal_df_display[x_col].apply(seconds_to_mmss)
                                                            display_cols = ['Time', y_col, 'Z_Score']
                                                            abnormal_df_display = abnormal_df_display[display_cols]
                                                        
                                                        abnormal_data.append(abnormal_df_display)
                                
                                        if abnormal_data:
                                            # Combine all abnormal data
                                            combined_abnormal = pd.concat(abnormal_data, ignore_index=True)
                                            combined_abnormal = combined_abnormal.sort_values('Z_Score', ascending=False)
                                            
                                            # Limit to first 100 rows for performance
                                            if len(combined_abnormal) > 100:
                                                st.info(f"Showing first 100 of {len(combined_abnormal)} abnormal points")
                                                combined_abnormal = combined_abnormal.head(100)
                                            
                                            st.dataframe(combined_abnormal, use_container_width=True)
                                            
                                            # Download button for abnormal data
                                            csv = combined_abnormal.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download Abnormal Data",
                                                data=csv,
                                                file_name=f"abnormal_data_{detail['name'].replace(' ', '_').replace('-', '_')}.csv",
                                                mime="text/csv"
                                            )
                                        else:
                                            st.info("No abnormal data points found for this plot")
                                    
                                    st.markdown("---")
                        else:
                            st.markdown("""
                            <div style='background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 12px; margin: 8px 0;'>
                                <div style='font-weight: bold; color: #155724;'>
                                    ✅ No Abnormal Points Detected
                                </div>
                                <div style='font-size: 0.9em; color: #0f5132; margin-top: 4px;'>
                                    All data points are within normal range (Z-Score ≤ {z_threshold})
                                </div>
                            </div>
                            """.format(z_threshold=z_threshold), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 12px; margin: 8px 0;'>
                            <div style='font-weight: bold; color: #721c24;'>
                                ⚠️ No Data Available
                            </div>
                            <div style='font-size: 0.9em; color: #721c24; margin-top: 4px;'>
                                Please select a file and topic to analyze
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                with plot_col:
                    # Plot grid size based on motor type and user selection
                    n_plots = len([p for p in plot_data if p is not None])
                    if n_plots == 0:
                        st.info("Select at least one topic/column with valid data to plot.")
                        st.stop()
                    
                    # Determine grid layout
                    if file_ext == ".ulg" and motor_type == "Hexa":
                        rows, cols = 3, 2
                        plot_data_to_show = plot_data  # Show all for Hexa
                    else:
                        rows, cols = 2, 2
                        plot_data_to_show = plot_data[:4]  # Only show up to 4 plots for Quad
                    
                    plot_idx = 0
                    for i in range(rows):
                        row_cols = st.columns(cols)
                        for j in range(cols):
                            if plot_idx >= len(plot_data_to_show):
                                continue
                            pdata = plot_data_to_show[plot_idx]
                            if pdata is not None:
                                with row_cols[j]:
                                    df = pdata["df"].copy()  # Create a copy to avoid modifying original
                                    
                                    # Apply x-axis filtering based on limits
                                    if file_ext == ".ulg" and 'timestamp_seconds' in df.columns:
                                        # Filter by timestamp range for ULG files
                                        df = df[(df['timestamp_seconds'] >= x_min_seconds) & (df['timestamp_seconds'] <= x_max_seconds)]
                                    elif file_ext != ".ulg" and 'Index' in df.columns:
                                        # Filter by index range for CSV files
                                        df = df[(df['Index'] >= x_min_seconds) & (df['Index'] <= x_max_seconds)]
                                    
                                    # Handle both ULG (y_col) and non-ULG (y_cols) data structures
                                    if "y_col" in pdata:
                                        # ULG file structure
                                        y_col = pdata["y_col"]
                                        y_cols = [y_col]
                                        x_col = "timestamp_seconds"
                                        title_text = f"{pdata.get('topic', 'Plot')} - {y_col}"
                                    else:
                                        # Non-ULG file structure
                                        y_cols = pdata["y_cols"]
                                        x_col = pdata.get("x_col", "timestamp_seconds")
                                        title_text = pdata.get("topic", "Plot")
                                    
                                    # Determine plot mode based on user selection
                                    mode = 'lines' # Default to lines
                                    
                                    fig = go.Figure()
                                    for y_col in y_cols:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=df[x_col],
                                                y=df[y_col],
                                                mode=mode,
                                                name=f"{y_col}"
                                            )
                                        )
                                    
                                    # Add anomaly detection if z_threshold is set
                                    if z_threshold > 0:
                                        for y_col in y_cols:
                                            # Calculate z-scores
                                            mean_val = df[y_col].mean()
                                            std_val = df[y_col].std()
                                            z_scores = np.abs((df[y_col] - mean_val) / std_val)
                                            abnormal_mask = z_scores > z_threshold
                                            
                                            if abnormal_mask.any():
                                                abnormal_points = df[abnormal_mask]
                                                fig.add_trace(
                                                    go.Scatter(
                                                        x=abnormal_points[x_col],
                                                        y=abnormal_points[y_col],
                                                        mode='markers',
                                                        marker=dict(color='red', size=8),
                                                        name=f'{y_col} (Abnormal)',
                                                        showlegend=False
                                                    )
                                                )
                                    
                                    fig.update_layout(
                                        title_text=title_text,
                                        title_x=0.4,
                                        height=350,
                                        width=550,
                                        showlegend=True,
                                        margin=dict(t=30, b=40, l=40, r=40),
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=-.3,  # Move legend a bit lower
                                            xanchor="left",
                                            x=0.0
                                        ),
                                    )
                                    
                                    # Set proper x-axis title and formatting
                                    if file_ext == ".ulg" and x_col == "timestamp_seconds":
                                        x_title = "Time (MM:SS)"
                                        # Add time formatting for ULG files
                                        if len(df[x_col]) > 0:
                                            tick_vals, tick_texts = get_timestamp_ticks(df[x_col])
                                            fig.update_xaxes(
                                                tickvals=tick_vals,
                                                ticktext=tick_texts,
                                                title_text=x_title,
                                                type='linear'
                                            )
                                    else:
                                        x_title = "Index" if x_col == "Index" else "Time (secs)"
                                        fig.update_xaxes(title_text=x_title)
                                    
                                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{plot_idx}")
                                    # Show a single row of stats for all y_cols below the plot
                                    stats_line = ""
                                    for y_col in y_cols:
                                        if "y_col" in pdata:
                                            # ULG file structure - stats is already for the single column
                                            stats = pdata["stats"]
                                        else:
                                            # Non-ULG file structure - get stats for this column
                                            stats = pdata["stats"].loc[y_col]
                                        mean_val = stats["mean"]
                                        min_val = stats["min"]
                                        max_val = stats["max"]
                                        stats_line += f" Mean: {mean_val:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f} &nbsp; &nbsp; "
                                    st.markdown(
                                        f"<div style='text-align:center; font-size:1.05em; margin-top:-2.5em; margin-bottom:1.5em;'>{stats_line}</div>",
                                        unsafe_allow_html=True
                                    )
                            plot_idx += 1
            with tab_data:
                # Show the filtered data table(s)
                if file_ext == ".ulg":
                    for filtered_df in filtered_dfs_for_data_tab:
                        if filtered_df is not None:
                            st.dataframe(filtered_df, use_container_width=True)
                else:
                    # For CSV files, show a single consolidated table with all selected columns
                    if filtered_dfs_for_data_tab and any(df is not None for df in filtered_dfs_for_data_tab):
                        # Get the first valid dataframe to use as base
                        base_df = next(df for df in filtered_dfs_for_data_tab if df is not None)
                        
                        # Create a consolidated dataframe with all selected columns
                        consolidated_df = base_df[['Index']].copy()
                        
                        # Add all selected columns to the consolidated dataframe
                        for filtered_df in filtered_dfs_for_data_tab:
                            if filtered_df is not None:
                                # Get the column name (excluding 'Index')
                                y_cols = [col for col in filtered_df.columns if col != 'Index']
                                for col in y_cols:
                                    if col not in consolidated_df.columns:
                                        consolidated_df[col] = filtered_df[col]
                        
                        st.dataframe(consolidated_df, use_container_width=True)
                        
                        # Show summary statistics for all columns
                        st.markdown("### 📊 Summary Statistics")
                        numeric_cols = [col for col in consolidated_df.columns if col != 'Index' and pd.api.types.is_numeric_dtype(consolidated_df[col])]
                        if numeric_cols:
                            summary_stats = consolidated_df[numeric_cols].describe()
                            st.dataframe(summary_stats, use_container_width=True)

        else:  # Comparative Analysis
            # Add safety check for mode switching
            if st.session_state.mode_switched:
                st.session_state.mode_switched = False
                st.session_state.switch_timestamp = None
            
            # Show guidance message if no files are selected
            if not st.session_state.get('selected_bench') or not st.session_state.get('selected_val'):
                st.info("📋 Please select both Benchmark and Target files to begin Comparative Analysis")
                st.stop()
            
            if st.session_state.get('selected_bench') and st.session_state.get('selected_val'):
                # Initialize variables
                b_df = None
                v_df = None
                b_file_ext = None
                v_file_ext = None
                b_dfs = {}
                v_dfs = {}
                selected_bench = st.session_state.get('selected_bench', "None")
                selected_val = st.session_state.get('selected_val', "None")
                selected_assessment = st.session_state.get('selected_assessment', "None")

                # Initialize benchmark variables
                selected_bench = st.session_state.get('selected_bench', "None")
                b_content = st.session_state.get('selected_bench_content')
                
                # Process benchmark file
                if selected_bench != "None" and b_content:
                    b_file_ext = None
                    tmp_file = None
                    try:
                        b_file_ext = os.path.splitext(selected_bench)[-1].lower()
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=b_file_ext)
                        if isinstance(b_content, str):
                            tmp_file.write(b_content.encode('utf-8'))
                        else:
                            tmp_file.write(b_content)
                        tmp_file.flush()
                        tmp_file.close()
                        
                        if b_file_ext == ".ulg":
                            b_dfs, b_topics = load_ulog(tmp_file.name)
                        else:
                            df, _ = load_data(tmp_file.name, b_file_ext, key_suffix="bench")
                            if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                                b_df = df
                                st.session_state.b_df = df
                    except Exception as e:
                        st.error(f"Error processing benchmark file: {str(e)}")
                    finally:
                        if tmp_file:
                            try:
                                os.unlink(tmp_file.name)
                            except:
                                pass

    # Initialize validation variables
    selected_val = st.session_state.get('selected_val', "None")
    v_content = st.session_state.get('selected_val_content')
    
    # Process validation file
    if selected_val != "None" and v_content:
        v_file_ext = None
        tmp_file = None
        try:
            v_file_ext = os.path.splitext(selected_val)[-1].lower()
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=v_file_ext)
            if isinstance(v_content, str):
                tmp_file.write(v_content.encode('utf-8'))
            else:
                tmp_file.write(v_content)
            tmp_file.flush()
            tmp_file.close()
            
            if v_file_ext == ".ulg":
                v_dfs, v_topics = load_ulog(tmp_file.name)
            else:
                df, _ = load_data(tmp_file.name, v_file_ext, key_suffix="val")
                if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                    v_df = df
                    st.session_state.v_df = df
        except Exception as e:
            st.error(f"Error processing target file: {str(e)}")
        finally:
            if tmp_file:
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
            
    # Show topic selection only after both ULG files are selected
    if (selected_bench != "None" and selected_val != "None" and 
        b_file_ext == ".ulg" and v_file_ext == ".ulg"):
        
        assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
        assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
        
        # Get the topic selected from home page
        default_assessment = st.session_state.get('selected_assessment')
        
        # If we have a valid topic from home page, use it directly
        if default_assessment and default_assessment != "None":
            selected_topic = assessment_to_topic.get(str(default_assessment))
            if selected_topic:
                if selected_topic in b_dfs and selected_topic in v_dfs:
                    b_df = b_dfs[selected_topic]
                    v_df = v_dfs[selected_topic]
                    st.session_state.b_df = b_dfs[selected_topic]
                    st.session_state.v_df = v_dfs[selected_topic]
                else:
                    st.warning(f"⚠️ Topic '{selected_topic}' not found in one or both files")
        
        # Show topic selection dropdown with default from home page
        # st.markdown("<h3 style='font-size: 20px;'>📊 Analysis Topic</h3>", unsafe_allow_html=True)
        
        # Handle default topic selection
        default_index = 0  # Default to "None"
        if isinstance(default_assessment, str) and default_assessment in assessment_names:
            default_index = assessment_names.index(default_assessment)
        
        # Update data if topic is changed
        if selected_assessment != "None" and selected_assessment != default_assessment:
            selected_topic = assessment_to_topic.get(str(selected_assessment))
            if selected_topic:
                if selected_topic in b_dfs and selected_topic in v_dfs:
                    b_df = b_dfs[selected_topic]
                    v_df = v_dfs[selected_topic]
                    st.session_state.b_df = b_dfs[selected_topic]
                    st.session_state.v_df = v_dfs[selected_topic]
                    # Update session state with new topic
                    st.session_state.selected_assessment = selected_assessment
                else:
                    st.warning(f"⚠️ Topic '{selected_topic}' not found in one or both files")
    elif selected_bench != "None" and selected_val != "None":
        # For non-ULG files, get data from session state
        b_df = st.session_state.get("b_df", None)
        v_df = st.session_state.get("v_df", None)

    # Show analysis tabs only if both files are loaded and selected
    if st.session_state.get('selected_bench') and st.session_state.get('selected_val'):
        
        def reset_x_comparative_callback(dataframe, axis):
            if dataframe is not None and axis in dataframe.columns:
                if axis == 'timestamp_seconds':
                    st.session_state.x_min_comparative_mmss = seconds_to_mmss(float(dataframe[axis].min()))
                    st.session_state.x_max_comparative_mmss = seconds_to_mmss(float(dataframe[axis].max()))
                else:
                    st.session_state.x_min_comparative = float(dataframe[axis].min())
                    st.session_state.x_max_comparative = float(dataframe[axis].max())
                if 'x_min_comparative_mmss' in st.session_state and axis != 'timestamp_seconds': del st.session_state['x_min_comparative_mmss']
                if 'x_max_comparative_mmss' in st.session_state and axis != 'timestamp_seconds': del st.session_state['x_max_comparative_mmss']

        def reset_y_comparative_callback(dataframe, axis):
            if dataframe is not None and axis in dataframe.columns:
                st.session_state.y_min_comparative = float(dataframe[axis].min())
                st.session_state.y_max_comparative = float(dataframe[axis].max())

        tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
        
        # Data Tab
        with tab2:
            # Create a 20-80 split layout
            data_col, settings_col = st.columns([0.8, 0.2])
            
            with settings_col:
                # Add dataset selector and column management
                # st.markdown("<h3 style='font-size: 20px;'>📋 Data Preview</h3>", unsafe_allow_html=True)
                st.markdown("<h5 style='font-size: 16px;'>🔧 Column Management</h5>", unsafe_allow_html=True)
                dataset_choice = st.selectbox(
                    "Select Dataset to Edit",
                    ["Benchmark", "Target", "Both"],
                    key="dataset_selector"
                )
                
                # Column management based on selection
                if isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                    if dataset_choice == "Benchmark":
                        st.markdown("**Editing Benchmark Dataset**")
                        b_df = add_remove_column(b_df, "Benchmark")
                    elif dataset_choice == "Target":
                        st.markdown("**Editing Target Dataset**")
                        v_df = add_remove_column(v_df, "Target")
                    else:  # Both
                        st.markdown("**Editing Both Datasets**")
                        with st.expander("Common Column Operations"):
                            b_df, v_df = add_remove_common_column(b_df, v_df)
            
            with data_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4 style='font-size: 18px;'>Benchmark Data</h4>", unsafe_allow_html=True)
                    if isinstance(b_df, pd.DataFrame):
                        # Add Index if not present
                        if 'Index' not in b_df.columns:
                            b_df.insert(0, 'Index', range(1, len(b_df) + 1))
                        
                        # Ensure timestamp_seconds is present
                        b_df = ensure_seconds_column(b_df)
                        
                        # Get display columns
                        display_cols = ['Index']
                        if 'timestamp_seconds' in b_df.columns:
                            display_cols.append('timestamp_seconds')
                        
                        # For ULG files with selected assessment
                        if b_file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                            # Add columns from assessment map
                            if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                display_cols.extend([col for col in assessment_cols if col in b_df.columns])
                        else:
                            # Add all numeric columns
                            numeric_cols = [col for col in b_df.columns if pd.api.types.is_numeric_dtype(b_df[col]) 
                                        and col not in display_cols]
                            display_cols.extend(numeric_cols)
                        
                        # Display DataFrame with selected columns
                        st.dataframe(
                            b_df[list(dict.fromkeys(display_cols))].rename(columns=COLUMN_DISPLAY_NAMES),
                            use_container_width=True,
                            height=600
                        )
                    elif isinstance(b_df, dict):
                        # Handle dictionary of DataFrames (ULog case)
                        if selected_assessment and selected_assessment != "None":
                            selected_topic = assessment_to_topic.get(str(selected_assessment))
                            if selected_topic and selected_topic in b_df:
                                df_to_display = b_df[selected_topic]
                                if isinstance(df_to_display, pd.DataFrame):
                                    # Add Index if not present
                                    if 'Index' not in df_to_display.columns:
                                        df_to_display.insert(0, 'Index', range(1, len(df_to_display) + 1))
                                    
                                    # Ensure timestamp_seconds is present
                                    df_to_display = ensure_seconds_column(df_to_display)
                                    
                                    # Get display columns
                                    display_cols = ['Index']
                                    if 'timestamp_seconds' in df_to_display.columns:
                                        display_cols.append('timestamp_seconds')
                                    
                                    # Add columns from assessment map
                                    if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                        assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                        display_cols.extend([col for col in assessment_cols if col in df_to_display.columns])
                                    
                                    # Display DataFrame with selected columns
                                    st.dataframe(
                                        df_to_display[list(dict.fromkeys(display_cols))].rename(columns=COLUMN_DISPLAY_NAMES),
                                        use_container_width=True,
                                        height=600
                                    )
                                else:
                                    st.warning("⚠️ Selected topic data is not in the correct format")
                            else:
                                st.warning("⚠️ Selected topic not found in the data")
                        else:
                            st.info("📋 Please select a topic to view the data")
                    else:
                        st.warning("⚠️ Benchmark data not properly loaded")
                
                with col2:
                    st.markdown("<h4 style='font-size: 18px;'>Target Data</h4>", unsafe_allow_html=True)
                    if isinstance(v_df, pd.DataFrame):
                        # Add Index if not present
                        if 'Index' not in v_df.columns:
                            v_df.insert(0, 'Index', range(1, len(v_df) + 1))
                        
                        # Ensure timestamp_seconds is present
                        v_df = ensure_seconds_column(v_df)
                        
                        # Get display columns
                        display_cols = ['Index']
                        if 'timestamp_seconds' in v_df.columns:
                            display_cols.append('timestamp_seconds')
                        
                        # For ULG files with selected assessment
                        if v_file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                            # Add columns from assessment map
                            if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                display_cols.extend([col for col in assessment_cols if col in v_df.columns])
                        else:
                            # Add all numeric columns
                            numeric_cols = [col for col in v_df.columns if pd.api.types.is_numeric_dtype(v_df[col]) 
                                        and col not in display_cols]
                            display_cols.extend(numeric_cols)
                        
                        # Display DataFrame with selected columns
                        st.dataframe(
                            v_df[list(dict.fromkeys(display_cols))].rename(columns=COLUMN_DISPLAY_NAMES),
                            use_container_width=True,
                            height=600
                        )
                    elif isinstance(v_df, dict):
                        # Handle dictionary of DataFrames (ULog case)
                        if selected_assessment and selected_assessment != "None":
                            selected_topic = assessment_to_topic.get(str(selected_assessment))
                            if selected_topic and selected_topic in v_df:
                                df_to_display = v_df[selected_topic]
                                if isinstance(df_to_display, pd.DataFrame):
                                    # Add Index if not present
                                    if 'Index' not in df_to_display.columns:
                                        df_to_display.insert(0, 'Index', range(1, len(df_to_display) + 1))
                                    
                                    # Ensure timestamp_seconds is present
                                    df_to_display = ensure_seconds_column(df_to_display)
                                    
                                    # Get display columns
                                    display_cols = ['Index']
                                    if 'timestamp_seconds' in df_to_display.columns:
                                        display_cols.append('timestamp_seconds')
                                    
                                    # Add columns from assessment map
                                    if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                        assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                        display_cols.extend([col for col in assessment_cols if col in df_to_display.columns])
                                    
                                    # Display DataFrame with selected columns
                                    st.dataframe(
                                        df_to_display[list(dict.fromkeys(display_cols))].rename(columns=COLUMN_DISPLAY_NAMES),
                                        use_container_width=True,
                                        height=600
                                    )
                                else:
                                    st.warning("⚠️ Selected topic data is not in the correct format")
                            else:
                                st.warning("⚠️ Selected topic not found in the data")
                        else:
                            st.info("📋 Please select a topic to view the data")
                    else:
                        st.warning("⚠️ Target data not properly loaded")
                    
        # Plot Tab
        with tab1:
            # --- Metrics Row (Full Width) ---
            b_df = st.session_state.get("b_df")
            v_df = st.session_state.get("v_df")
            metrics_ready = False
            x_axis = y_axis = z_threshold = x_min = x_max = y_min = y_max = None
            
            x_axis_options = []
            y_axis_options = []
            default_x = None
            default_y = None

            if isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                b_numeric = get_numeric_columns(b_df)
                v_numeric = get_numeric_columns(v_df)
                common_cols = list(set(b_numeric) & set(v_numeric))
                ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in common_cols if col not in ["Index", "timestamp_seconds"]]
                if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                    allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                    allowed_y_axis = [col for col in allowed_y_axis if col in b_df.columns]
                    if not allowed_y_axis:
                        allowed_y_axis = list(b_df.columns)
                    allowed_y_axis = [col for col in allowed_y_axis if pd.api.types.is_numeric_dtype(b_df[col])]
                    ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                else:
                    allowed_y_axis = [col for col in b_df.columns if pd.api.types.is_numeric_dtype(b_df[col])]
                    ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                x_axis_options = ALLOWED_X_AXIS
                y_axis_options = allowed_y_axis
                if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                    default_x = 'timestamp_seconds' if 'timestamp_seconds' in x_axis_options else ('Index' if 'Index' in x_axis_options else x_axis_options[0])
                else:
                    default_x = 'Index' if 'Index' in x_axis_options else x_axis_options[0]
                if b_df is not None and v_df is not None and hasattr(b_df, 'columns') and hasattr(v_df, 'columns'):
                    is_csv_data = any(col in b_df.columns for col in ['Thrust (kgf)', 'cD2detailpeak', 'Thrust'])
                    if is_csv_data:
                        preferred_y_columns = ['Thrust (kgf)', 'cD2detailpeak', 'Thrust']
                        for preferred_col in preferred_y_columns:
                            if preferred_col in y_axis_options:
                                default_y = preferred_col
                                break
                        else:
                            default_y = y_axis_options[1] if y_axis_options else None
                else:
                    default_y = y_axis_options[0] if y_axis_options else None
            metrics_col, param_col = st.columns([0.8, 0.2])
            with param_col:
                st.markdown("""
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span style='font-size: 1.2rem;'>📝</span>
                    <span style='font-size: 1.1rem; font-weight: 600;'>Parameters</span>
                </div>
                """, unsafe_allow_html=True)

                prev_x_axis = st.session_state.get('prev_x_axis_comparative', None)
                prev_y_axis = st.session_state.get('prev_y_axis_comparative', None)

                x_axis_display_cmp = [get_display_name(col) for col in x_axis_options]
                x_axis_selected_display_cmp = st.selectbox("X-Axis", x_axis_display_cmp, key="x_axis_comparative", index=x_axis_display_cmp.index(get_display_name(default_x)) if isinstance(default_x, str) and get_display_name(default_x) in x_axis_display_cmp else 0)
                x_axis = x_axis_options[x_axis_display_cmp.index(x_axis_selected_display_cmp)]
                y_axis_display_cmp = [get_display_name(col) for col in y_axis_options]
                y_axis_selected_display_cmp = st.selectbox("Y-Axis", y_axis_display_cmp, key="y_axis_comparative", index=y_axis_display_cmp.index(get_display_name(default_y)) if isinstance(default_y, str) and get_display_name(default_y) in y_axis_display_cmp else 0)
                y_axis = y_axis_options[y_axis_display_cmp.index(y_axis_selected_display_cmp)]

                if 'last_x_axis_comparative' not in st.session_state or st.session_state['last_x_axis_comparative'] != x_axis:
                    if x_axis == 'timestamp_seconds':
                        if b_df is not None and x_axis in b_df.columns:
                            x_min_val = float(b_df[x_axis].min())
                            x_max_val = float(b_df[x_axis].max())
                        elif v_df is not None and x_axis in v_df.columns:
                            x_min_val = float(v_df[x_axis].min())
                            x_max_val = float(v_df[x_axis].max())
                        else:
                            x_min_val, x_max_val = 0.0, 1.0
                        st.session_state['x_min_comparative_mmss'] = seconds_to_mmss(x_min_val)
                        st.session_state['x_max_comparative_mmss'] = seconds_to_mmss(x_max_val)
                    else:
                        if b_df is not None and x_axis in b_df.columns:
                            x_min_val = float(b_df[x_axis].min())
                            x_max_val = float(b_df[x_axis].max())
                        elif v_df is not None and x_axis in v_df.columns:
                            x_min_val = float(v_df[x_axis].min())
                            x_max_val = float(v_df[x_axis].max())
                        else:
                            x_min_val, x_max_val = 0.0, 1.0
                        st.session_state['x_min_comparative'] = x_min_val
                        st.session_state['x_max_comparative'] = x_max_val
                    st.session_state['last_x_axis_comparative'] = x_axis
                if 'last_y_axis_comparative' not in st.session_state or st.session_state['last_y_axis_comparative'] != y_axis:
                    if b_df is not None and y_axis in b_df.columns:
                        y_min_val = float(b_df[y_axis].min())
                        y_max_val = float(b_df[y_axis].max())
                    elif v_df is not None and y_axis in v_df.columns:
                        y_min_val = float(v_df[y_axis].min())
                        y_max_val = float(v_df[y_axis].max())
                    else:
                        y_min_val, y_max_val = 0.0, 1.0
                    st.session_state['y_min_comparative'] = y_min_val
                    st.session_state['y_max_comparative'] = y_max_val
                    st.session_state['last_y_axis_comparative'] = y_axis

                st.session_state['prev_x_axis_comparative'] = x_axis
                st.session_state['prev_y_axis_comparative'] = y_axis

                z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 2.5, 0.01, key="z-slider-comparative")
                
                # --- Combined Range Calculation ---
                x_min_val, x_max_val, y_min_val, y_max_val = (0.0, 1.0, 0.0, 1.0)
                if x_axis and y_axis:
                    # Calculate combined X-axis range
                    if b_df is not None and v_df is not None and x_axis in b_df.columns and x_axis in v_df.columns:
                        x_min_val, x_max_val = (min(b_df[x_axis].min(), v_df[x_axis].min()), max(b_df[x_axis].max(), v_df[x_axis].max()))
                    elif b_df is not None and x_axis in b_df.columns:
                        x_min_val, x_max_val = (b_df[x_axis].min(), b_df[x_axis].max())
                    elif v_df is not None and x_axis in v_df.columns:
                        x_min_val, x_max_val = (v_df[x_axis].min(), v_df[x_axis].max())

                    # Calculate combined Y-axis range
                    if b_df is not None and v_df is not None and y_axis in b_df.columns and y_axis in v_df.columns:
                        y_min_val, y_max_val = (min(b_df[y_axis].min(), v_df[y_axis].min()), max(b_df[y_axis].max(), v_df[y_axis].max()))
                    elif b_df is not None and y_axis in b_df.columns:
                        y_min_val, y_max_val = (b_df[y_axis].min(), b_df[y_axis].max())
                    elif v_df is not None and y_axis in v_df.columns:
                        y_min_val, y_max_val = (v_df[y_axis].min(), v_df[y_axis].max())

                st.markdown(f"<span style='font-size:1.05rem; color:#444; font-weight:500;'>{'Time' if x_axis == 'timestamp_seconds' else x_axis}</span>", unsafe_allow_html=True)
                x_min_col, x_max_col, x_reset_col = st.columns([6, 6, 2])
                with x_min_col:
                    if x_axis == 'timestamp_seconds':
                        x_min_mmss = seconds_to_mmss(x_min_val)
                        if 'x_min_comparative_mmss' not in st.session_state:
                            st.session_state['x_min_comparative_mmss'] = x_min_mmss
                        x_min_input = st.text_input("Start", value=st.session_state['x_min_comparative_mmss'], key='x_min_comparative_mmss')
                        x_min = mmss_to_seconds(x_min_input)
                    else:
                        if 'x_min_comparative' not in st.session_state:
                            st.session_state['x_min_comparative'] = float(x_min_val)
                        x_min = st.number_input("Start", value=st.session_state['x_min_comparative'], format="%.2f", key="x_min_comparative", step=1.0)
                with x_max_col:
                    if x_axis == 'timestamp_seconds':
                        x_max_mmss = seconds_to_mmss(x_max_val)
                        if 'x_max_comparative_mmss' not in st.session_state:
                            st.session_state['x_max_comparative_mmss'] = x_max_mmss
                        x_max_input = st.text_input("End", value=st.session_state['x_max_comparative_mmss'], key='x_max_comparative_mmss')
                        x_max = mmss_to_seconds(x_max_input)
                    else:
                        if 'x_max_comparative' not in st.session_state:
                            st.session_state['x_max_comparative'] = float(x_max_val)
                        x_max = st.number_input("End", value=st.session_state['x_max_comparative'], format="%.2f", key="x_max_comparative", step=1.0)
                with x_reset_col:
                    st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
                    if st.button("↺", key="reset_x_comparative", help="Reset X-axis range", on_click=reset_x_comparative_callback, args=(b_df, x_axis)):
                        for k in ['x_min_comparative', 'x_max_comparative', 'x_min_comparative_mmss', 'x_max_comparative_mmss']:
                            if k in st.session_state:
                                del st.session_state[k]
                        st.rerun()

                st.markdown(f"<span style='font-size:1.05rem; color:#444; font-weight:500;'>{y_axis}</span>", unsafe_allow_html=True)
                y_min_col, y_max_col, y_reset_col = st.columns([6, 6, 2])
                with y_min_col:
                    if 'y_min_comparative' not in st.session_state:
                        st.session_state['y_min_comparative'] = float(y_min_val)
                    y_min = st.number_input("Start", value=st.session_state['y_min_comparative'], format="%.2f", key="y_min_comparative", step=1.0)
                with y_max_col:
                    if 'y_max_comparative' not in st.session_state:
                        st.session_state['y_max_comparative'] = float(y_max_val)
                    y_max = st.number_input("End", value=st.session_state['y_max_comparative'], format="%.2f", key="y_max_comparative", step=1.0)
                with y_reset_col:
                    st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
                    if st.button("↺", key="reset_y_comparative", help="Reset Y-axis range", on_click=reset_y_comparative_callback, args=(b_df, y_axis)):
                        for k in ['y_min_comparative', 'y_max_comparative']:
                            if k in st.session_state:
                                del st.session_state[k]
                        st.rerun()
            with metrics_col:
                # Calculate metrics based on current parameters
                if (b_df is not None and v_df is not None and x_axis in b_df.columns and y_axis in b_df.columns and 
                    x_axis in v_df.columns and y_axis in v_df.columns and 
                    None not in (x_min, x_max, y_min, y_max)):
                    try:
                        b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) & (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                        v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) & (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
                        
                        if len(b_filtered) > 0 and len(v_filtered) > 0:
                            if x_axis == 'timestamp_seconds':
                                b_filtered, v_filtered, _ = resample_to_common_time(b_filtered, v_filtered)
                            
                            if len(b_filtered) > 0 and len(v_filtered) > 0:
                                merged = pd.DataFrame()
                                merged['benchmark'] = b_filtered[y_axis].reset_index(drop=True)
                                merged['target'] = v_filtered[y_axis].reset_index(drop=True)
                                merged['benchmark_x'] = b_filtered[x_axis].reset_index(drop=True)
                                merged['target_x'] = v_filtered[x_axis].reset_index(drop=True)
                                merged['abs_diff'] = abs(merged['target'] - merged['benchmark'])
                                merged['rel_diff'] = merged['abs_diff'] / (abs(merged['benchmark']) + 1e-10)
                                window = min(50, max(20, len(merged) // 10))
                                merged['rolling_mean'] = merged['abs_diff'].rolling(window=window, center=True).mean()
                                merged['rolling_std'] = merged['abs_diff'].rolling(window=window, center=True).std()
                                rmse = np.sqrt(np.mean((merged['target'] - merged['benchmark']) ** 2))
                                # Use combined range of both datasets for symmetric similarity calculation
                                combined_range = max(merged['benchmark'].max(), merged['target'].max()) - min(merged['benchmark'].min(), merged['target'].min())
                                similarity = 1 - (rmse / combined_range) if combined_range != 0 else (1.0 if rmse == 0 else 0.0)
                                similarity_index = similarity * 100
                                merged["Difference"] = merged['target'] - merged['benchmark']
                                merged["Z_Score"] = (merged["Difference"] - merged["Difference"].mean()) / merged["Difference"].std()
                                merged = merged.reset_index(drop=True)
                                abnormal_mask = abs(merged["Z_Score"]) > z_threshold
                                abnormal_points = merged[abnormal_mask]
                                abnormal_count = int(abnormal_mask.sum())
                                metrics_ready = True
                            else:
                                rmse = 0.0
                                similarity_index = 0.0
                                abnormal_count = 0
                                metrics_ready = False
                        else:
                            rmse = 0.0
                            similarity_index = 0.0
                            abnormal_count = 0
                            metrics_ready = False
                    except Exception as e:
                        rmse = 0.0
                        similarity_index = 0.0
                        abnormal_count = 0
                        metrics_ready = False
                else:
                    rmse = 0.0
                    similarity_index = 0.0
                    abnormal_count = 0
                    metrics_ready = False
                
                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    fig1 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=rmse,
                        title={'text': "RMSE"},
                        number={'valueformat': ',.2f'},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, max(rmse * 2, 1)], 'tickformat': ',.2f'},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, rmse], 'color': "lightgray"},
                                {'range': [rmse, max(rmse * 2, 1)], 'color': "gray"}
                            ]
                        }
                    ))
                    fig1.update_layout(width=200, height=120, margin=dict(t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig1, use_container_width=False)
                with metrics_cols[1]:
                    fig2 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=similarity_index,
                        title={'text': "Similarity Index (%)"},
                        number={'valueformat': '.2f', 'suffix': '%'},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickformat': '.0f'},
                            'bar': {'color': "orange"},
                            'steps': [
                                {'range': [0, 33], 'color': "#d4f0ff"},
                                {'range': [33, 66], 'color': "#ffeaa7"},
                                {'range': [66, 100], 'color': "#c8e6c9"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig2.update_layout(width=200, height=120, margin=dict(t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=False)
                with metrics_cols[2]:
                    fig3 = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=abnormal_count,
                        title={'text': "Abnormal Points"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, max(10, abnormal_count * 2)]},
                            'bar': {'color': "crimson"},
                            'steps': [
                                {'range': [0, 10], 'color': "#c8e6c9"},
                                {'range': [10, 25], 'color': "#ffcc80"},
                                {'range': [25, 100], 'color': "#ef5350"}
                            ]
                        }
                    ))
                    fig3.update_layout(width=200, height=120, margin=dict(t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig3, use_container_width=False)
                # --- Main Content: Plot ---
                plot_container = st.container()
                with plot_container:
                    if not x_axis or not y_axis:
                        st.info("📊 Please select valid X and Y axes to compare")
                        st.stop()
                    if b_df is not None and hasattr(b_df, 'columns') and x_axis in b_df.columns:
                        pass
                    else:
                        st.error(f"Selected X-axis '{x_axis}' not found in data")
                        st.stop()
                    # Plot visualization (move all plot code here)
                    if x_axis and y_axis and isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                        try:
                            b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) & (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                            v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) & (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]
                            if x_axis == 'timestamp_seconds':
                                b_filtered, v_filtered, common_time = resample_to_common_time(b_filtered, v_filtered)
                            merged = pd.DataFrame()
                            merged['benchmark'] = b_filtered[y_axis].reset_index(drop=True)
                            merged['target'] = v_filtered[y_axis].reset_index(drop=True)
                            merged['benchmark_x'] = b_filtered[x_axis].reset_index(drop=True)
                            merged['target_x'] = v_filtered[x_axis].reset_index(drop=True)
                            merged['abs_diff'] = abs(merged['target'] - merged['benchmark'])
                            merged['rel_diff'] = merged['abs_diff'] / (abs(merged['benchmark']) + 1e-10)
                            window = min(50, max(20, len(merged) // 10))
                            merged['rolling_mean'] = merged['abs_diff'].rolling(window=window, center=True).mean()
                            merged['rolling_std'] = merged['abs_diff'].rolling(window=window, center=True).std()
                            rmse = np.sqrt(np.mean((merged['target'] - merged['benchmark']) ** 2))
                            # Use combined range of both datasets for symmetric similarity calculation
                            combined_range = max(merged['benchmark'].max(), merged['target'].max()) - min(merged['benchmark'].min(), merged['target'].min())
                            similarity = 1 - (rmse / combined_range) if combined_range != 0 else (1.0 if rmse == 0 else 0.0)
                            similarity_index = similarity * 100
                            merged["Difference"] = merged['target'] - merged['benchmark']
                            merged["Z_Score"] = (merged["Difference"] - merged["Difference"].mean()) / merged["Difference"].std()
                            merged = merged.reset_index(drop=True)
                            abnormal_mask = abs(merged["Z_Score"]) > z_threshold
                            abnormal_points = merged[abnormal_mask]
                            abnormal_count = int(abnormal_mask.sum())
                            # --- Plot Visualization heading and Plot Mode selector in one row ---
                            heading_col, mode_col = st.columns([0.7, 0.3])
                            with heading_col:
                                st.markdown("### 🧮 Plot Visualization")
                            with mode_col:
                                # Plot mode selection and rerun logic
                                if 'previous_plot_mode' not in st.session_state:
                                    st.session_state['previous_plot_mode'] = 'Superimposed'
                                plot_mode = st.radio(
                                    "Plot Mode",
                                    ["Superimposed", "Separate"],
                                    horizontal=True,
                                    key="comparative_plot_mode"
                                )
                                if plot_mode != st.session_state['previous_plot_mode']:
                                    st.session_state['previous_plot_mode'] = plot_mode
                                    st.rerun()
                            plot_container = st.container()
                            with plot_container:
                                if plot_mode == "Superimposed":
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=b_filtered[x_axis],
                                        y=b_filtered[y_axis],
                                        mode='lines',
                                        name='Benchmark'
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=v_filtered[x_axis],
                                        y=v_filtered[y_axis],
                                        mode='lines',
                                        name='Target',
                                        line=dict(color='green')
                                    ))
                                    if not abnormal_points.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=abnormal_points['benchmark_x'],
                                                y=abnormal_points['benchmark'],
                                                mode='markers',
                                                marker=dict(color='red', size=8),
                                                name='Abnormal Points (Benchmark)'
                                            )
                                        )
                                        fig.add_trace(
                                            go.Scatter(
                                                x=abnormal_points['target_x'],
                                                y=abnormal_points['target'],
                                                mode='markers',
                                                marker=dict(color='orange', size=8),
                                                name='Abnormal Points (Target)'
                                            )
                                        )
                                    if x_axis == 'timestamp_seconds':
                                        tick_vals, tick_texts = get_timestamp_ticks(b_filtered[x_axis])
                                        fig.update_xaxes(
                                            tickvals=tick_vals,
                                            ticktext=tick_texts,
                                            title_text=get_axis_title(x_axis),
                                            type='linear'
                                        )
                                    else:
                                        fig.update_xaxes(title_text=x_axis)
                                    fig.update_layout(
                                        height=450,
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.01,
                                            xanchor="center",
                                            x=0.5
                                        ),
                                        margin=dict(t=15, b=10, l=50, r=20),
                                        plot_bgcolor='white',
                                        yaxis=dict(
                                            showticklabels=True,
                                            title=y_axis
                                        )
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    # Add abnormal points table below the plot
                                    if not abnormal_points.empty:
                                        st.markdown("### ⚠️ Abnormal Points Data")
                                        # Only show the target X-axis value, with the X-axis name as the column header
                                        table_cols = ['target_x', 'benchmark', 'target', 'Difference', 'Z_Score']
                                        table_cols = [col for col in table_cols if col in abnormal_points.columns]
                                        display_df = abnormal_points[table_cols].copy()
                                        if 'target_x' in display_df.columns:
                                            display_df = display_df.rename(columns={'target_x': x_axis})
                                        st.dataframe(
                                            display_df.round(4),
                                            use_container_width=True,
                                            height=250
                                        )
                                elif plot_mode == "Separate":
                                    fig = make_subplots(rows=2, cols=1, 
                                                    shared_xaxes=True, 
                                                    subplot_titles=None,  
                                                    vertical_spacing=0.08,  
                                                    row_heights=[0.5, 0.5])
                                    fig.add_trace(go.Scatter(
                                        x=b_filtered[x_axis],
                                        y=b_filtered[y_axis],
                                        mode='lines',
                                        name='Benchmark',
                                        line=dict(color='blue')
                                    ), row=1, col=1)
                                    fig.add_trace(go.Scatter(
                                        x=v_filtered[x_axis],
                                        y=v_filtered[y_axis],
                                        mode='lines',
                                        name='Target',
                                        line=dict(color='green')
                                    ), row=2, col=1)
                                    if not abnormal_points.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=abnormal_points['benchmark_x'],
                                                y=abnormal_points['benchmark'],
                                                mode='markers',
                                                marker=dict(color='red', size=8),
                                                name='Abnormal Points (Benchmark)'
                                            ), row=1, col=1
                                        )
                                        fig.add_trace(
                                            go.Scatter(
                                                x=abnormal_points['target_x'],
                                                y=abnormal_points['target'],
                                                mode='markers',
                                                marker=dict(color='orange', size=8),
                                                name='Abnormal Points (Target)'
                                            ), row=2, col=1
                                        )
                                    if x_axis == 'timestamp_seconds':
                                        tick_vals, tick_texts = get_timestamp_ticks(b_filtered[x_axis])
                                        fig.update_xaxes(
                                            tickvals=tick_vals,
                                            ticktext=tick_texts,
                                            title_text="",  
                                            type='linear',
                                            row=1, col=1
                                        )
                                        fig.update_xaxes(
                                            tickvals=tick_vals,
                                            ticktext=tick_texts,
                                            title_text=get_axis_title(x_axis),
                                            type='linear',
                                            row=2, col=1
                                        )
                                    else:
                                        fig.update_xaxes(title_text="", row=1, col=1)
                                        fig.update_xaxes(title_text=x_axis, row=2, col=1)
                                    fig.update_layout(
                                        height=450,
                                        showlegend=True,
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.01,
                                            xanchor="center",
                                            x=0.5
                                        ),
                                        margin=dict(t=15, b=10, l=50, r=20),
                                        plot_bgcolor='white',
                                        yaxis=dict(
                                            showticklabels=True,
                                            title=y_axis
                                        )
                                    )
                                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                                    fig.update_yaxes(title_text=y_axis, row=1, col=1)
                                    fig.update_yaxes(title_text=y_axis, row=2, col=1)
                                    st.plotly_chart(fig, use_container_width=True)
                                    # Add abnormal points table below the plot
                                    if not abnormal_points.empty:
                                        st.markdown("### ⚠️ Abnormal Points Data")
                                        # Only show the target X-axis value, with the X-axis name as the column header
                                        table_cols = ['target_x', 'benchmark', 'target', 'Difference', 'Z_Score']
                                        table_cols = [col for col in table_cols if col in abnormal_points.columns]
                                        display_df = abnormal_points[table_cols].copy()
                                        if 'target_x' in display_df.columns:
                                            display_df = display_df.rename(columns={'target_x': x_axis})
                                        st.dataframe(
                                            display_df.round(4),
                                            use_container_width=True,
                                            height=250
                                        )
                        except Exception as e:
                            st.error(f"Error during plotting: {str(e)}")
    else:
        # Show appropriate guidance message
        if selected_bench != "None" and selected_val != "None" and b_file_ext == ".ulg" and v_file_ext == ".ulg":
            st.info("📋 Please select a topic to begin analysis")
        elif b_df is None and v_df is None:
            pass  # Removed warning message
        elif b_df is None:
            pass  # Removed warning message
        else:
            pass  # Removed warning message

if (st.session_state.get('analysis_type') == 'Comparative Analysis' and len(st.session_state.uploaded_files) == 1):
    st.markdown("<div style='margin-top:2em; text-align:center;'>", unsafe_allow_html=True)
    if st.button('Upload More Files', key='upload_more_files_btn', use_container_width=True):
        st.session_state.show_upload_area = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Footer ---
    st.markdown(
        """
        <hr style='margin-top: 2em; margin-bottom: 0.5em; border: none; border-top: 1px solid #e0e0e0;'>
        <div style='text-align: center; color: #888; font-size: 0.98rem; margin-bottom: 0.5em;'>
            If any issues persist, contact us <a href='mailto:business@reude.tech' style='color: #2E86C1; text-decoration: underline;'>business@reude.tech</a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
