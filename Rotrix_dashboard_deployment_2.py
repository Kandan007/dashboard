#Modified script to support both single file analysis and comparative assessment modes


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from io import StringIO
from PIL import Image
import base64
# import open3d as o3d
import tempfile
import os
from pyulog import ULog


st.set_page_config(page_title="ROTRIX Dashboard", layout="wide")

# Helper functions for data handling
def safe_get_range(df, column):
    """Safely get min and max values for a DataFrame column."""
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return 0, 0
    try:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            return 0, 0
        return float(series.min()), float(series.max())
    except:
        return 0, 0

def clean_dataframe(df):
    """Remove columns with all null values or no variation in data."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
        
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Remove columns with all null values
    null_columns = df_cleaned.columns[df_cleaned.isna().all()].tolist()
    if null_columns:
        df_cleaned = df_cleaned.drop(columns=null_columns)
    
    # Remove columns with constant values (no variation)
    constant_columns = df_cleaned.columns[df_cleaned.nunique() == 1].tolist()
    if constant_columns:
        df_cleaned = df_cleaned.drop(columns=constant_columns)
    
    return df_cleaned

def get_plot_ranges(b_df, v_df, x_axis, y_axis):
    """Get plot ranges for both DataFrames."""
    b_x_min, b_x_max = safe_get_range(b_df, x_axis)
    v_x_min, v_x_max = safe_get_range(v_df, x_axis)
    b_y_min, b_y_max = safe_get_range(b_df, y_axis)
    v_y_min, v_y_max = safe_get_range(v_df, y_axis)
    
    return (
        min(b_x_min, v_x_min),
        max(b_x_max, v_x_max),
        min(b_y_min, v_y_min),
        max(b_y_max, v_y_max)
    )

def get_numeric_columns(df):
    """Safely get numeric columns from a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return []
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

def detect_abnormalities(series, threshold=3.0):
    """Detect abnormal points in a series using z-score threshold."""
    if len(series) < 2:  # Need at least 2 points to calculate z-score
        return pd.Series(False, index=series.index), pd.Series(0, index=series.index)
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

# Initialize session state variables if they don't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'b_df' not in st.session_state:
    st.session_state.b_df = None
if 'v_df' not in st.session_state:
    st.session_state.v_df = None

# Function to change page
def change_page(page):
    st.session_state.current_page = page

# Utility functions
def load_csv(file):
    file.seek(0)
    return pd.read_csv(StringIO(file.read().decode("utf-8")))

def load_ulog(file, key_suffix=""):
    ALLOWED_TOPICS = set(t for t, _ in TOPIC_ASSESSMENT_PAIRS)
    ulog = ULog(file)
    extracted_dfs = {msg.name: pd.DataFrame(msg.data) for msg in ulog.data_list}
    filtered_dfs = {topic: df for topic, df in extracted_dfs.items() if topic in ALLOWED_TOPICS}
    topic_names = ["None"] + list(filtered_dfs.keys())
    if not topic_names:
        st.warning("No extractable topics found in ULOG file.")
        return {}, []
    return filtered_dfs, topic_names

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

def resample_to_common_time(df1, df2, freq=1.0):
    """Resample two dataframes to a common time base."""
    if 'timestamp_seconds' not in df1.columns or 'timestamp_seconds' not in df2.columns:
        st.error("timestamp_seconds column missing in one or both dataframes")
        return df1, df2, []
        
    try:
        df1 = df1.copy().sort_values('timestamp_seconds')
        df2 = df2.copy().sort_values('timestamp_seconds')
        
        start = max(df1['timestamp_seconds'].min(), df2['timestamp_seconds'].min())
        end = min(df1['timestamp_seconds'].max(), df2['timestamp_seconds'].max())
        
        if start >= end:
            st.error("No overlapping time range for resampling")
            return df1, df2, []
            
        common_time = np.arange(start, end, freq)
        if len(common_time) == 0:
            st.error("No common time points found after resampling")
            return df1, df2, []
        
        df1_interp = df1.set_index('timestamp_seconds').interpolate(method='linear').reindex(common_time, method='nearest')
        df2_interp = df2.set_index('timestamp_seconds').interpolate(method='linear').reindex(common_time, method='nearest')
        
        df1_interp = df1_interp.reset_index().rename(columns={'index': 'timestamp_seconds'})
        df2_interp = df2_interp.reset_index().rename(columns={'index': 'timestamp_seconds'})
        
        return df1_interp, df2_interp, common_time
    except Exception as e:
        st.error(f"Error during resampling: {str(e)}")
        return df1, df2, []

def load_data(file, filetype, key_suffix):
    """Load data from file and ensure proper timestamp handling."""
    if filetype == ".csv":
        df_csv = load_csv(file)
        df_csv = convert_timestamps_to_seconds(df_csv)
        df_csv = ensure_seconds_column(df_csv)
        return df_csv, None
    elif filetype == ".ulg":
        df_ulog, topic_names = load_ulog(file, key_suffix)
        if isinstance(df_ulog, dict):
            # Handle each dataframe in the dictionary
            for topic in df_ulog:
                df_ulog[topic] = convert_timestamps_to_seconds(df_ulog[topic])
                df_ulog[topic] = ensure_seconds_column(df_ulog[topic])
        return df_ulog, topic_names
    return None, None

def convert_timestamps_to_seconds(df):
    """Convert timestamp columns to seconds."""
    if df is None:
        return df
    if isinstance(df, pd.DataFrame) and len(df.index) > 0:
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        for col in timestamp_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] / 1000000
    return df

def ensure_seconds_column(df):
    """Ensure a 'timestamp_seconds' column exists and is always in seconds from start"""
    if df is None or df.empty:
        return df
    timestamp_col = next((col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()), None)
    if timestamp_col is None:
        return df
    series = df[timestamp_col]
    
    # Get the current topic from the dataframe name or context
    current_topic = None
    for topic, _ in TOPIC_ASSESSMENT_PAIRS:
        if topic in str(df.name) if hasattr(df, 'name') else False:
            current_topic = topic
            break
    
    # Skip special conversion for Control topic
    if current_topic == "px4io_status":  # Control topic
        if series.max() > 1e12:
            df['timestamp_seconds'] = (series - series.min()) / 1e6
        elif series.max() > 1e9:
            df['timestamp_seconds'] = (series - series.min()) / 1e3
        else:
            df['timestamp_seconds'] = series - series.min()
    else:
        # For all other topics, move decimal point 3 places to the left
        df['timestamp_seconds'] = (series - series.min()) / 1000000
    
    # Drop timestamp_sample if it exists
    if 'timestamp_sample' in df.columns:
        df.drop('timestamp_sample', axis=1, inplace=True)
    
    return df

def add_remove_column(target_df, df_name="DataFrame"):
    if target_df is None or target_df.empty:
        st.warning(f"⚠ {df_name} is empty or not loaded.")
        return target_df
    
    st.markdown("##### 🧮 New Column")
    new_col_name = st.text_input("New Column Name", key=f"{df_name}_add")
    custom_formula = st.text_input("Formula (e.g., Voltage * Current)", key=f"{df_name}_formula")

    if st.button(f"Add Column to {df_name}"):
        try:
            if new_col_name and custom_formula:
                target_df[new_col_name] = target_df.eval(custom_formula)
                st.success(f"✅ Added column {new_col_name} to {df_name} using: {custom_formula}")
        except Exception as e:
            st.error(f"❌ Error creating column: {e}")

    st.markdown("##### 🗑 Remove Column")
    columns_to_drop = st.multiselect("Select columns to drop", target_df.columns, key=f"{df_name}_drop")

    if st.button(f"Remove Column from {df_name}"):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"🗑 Removed columns: {', '.join(columns_to_drop)} from {df_name}")
            
    st.markdown("##### ✏ Rename Column")
    rename_col = st.selectbox("Select column to rename", target_df.columns, key=f"{df_name}_rename_col")
    new_name = st.text_input("New column name", key=f"{df_name}_rename_input")

    if st.button(f"Rename Column in {df_name}", key=f"{df_name}_rename_button"):
        if rename_col and new_name:
            target_df.rename(columns={rename_col: new_name}, inplace=True)
            st.success(f"✏ Renamed column {rename_col} to {new_name} in {df_name}")

    return target_df

def add_remove_common_column(b_df, v_df):
    if b_df is None or v_df is None or b_df.empty or v_df.empty:
        st.warning("⚠ Both Benchmark and Target data must be loaded.")
        return b_df, v_df

    if "pending_column" in st.session_state:
        new_col = st.session_state["pending_column"]
        try:
            for df_key in ["b_df", "v_df"]:
                df = st.session_state[df_key]
                if new_col["name"] not in df.columns:
                    df.insert(1, new_col["name"], df.eval(new_col["formula"]))
                else:
                    df[new_col["name"]] = df.eval(new_col["formula"])
                st.session_state[df_key] = df
            st.success(f"✅ Added {new_col['name']} using {new_col['formula']} to both Benchmark and Target.")
        except Exception as e:
            st.error(f"❌ Failed to add column: {e}")
        del st.session_state["pending_column"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("###### 🧮 New Column")
        new_col_name = st.text_input("New Column Name", key="common_add")
        custom_formula = st.text_input("Formula (e.g., Voltage * Current)", key="common_formula")

        if st.button("Add Column"):
            if new_col_name and custom_formula:
                st.session_state["pending_column"] = {
                    "name": new_col_name,
                    "formula": custom_formula
                }
                st.rerun()

    with col2:
        st.markdown("###### 🗑 Remove Column")
        common_cols = list(set(b_df.columns) & set(v_df.columns))
        cols_to_drop = st.multiselect("Select column(s) to drop", common_cols, key="common_drop")

        if st.button("Remove Columns"):
            if cols_to_drop:
                st.session_state.b_df.drop(columns=cols_to_drop, inplace=True)
                st.session_state.v_df.drop(columns=cols_to_drop, inplace=True)
                st.success(f"🗑 Removed columns: {', '.join(cols_to_drop)} from both Benchmark and Target.")
                st.rerun()

    return st.session_state.b_df, st.session_state.v_df

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
    "Thrust": ["thrust[0]", "thrust[1]", "thrust[2]"],
    "Torque": ["xyz[0]", "xyz[1]", "xyz[2]"],
    "Control": ["pwm[0]", "pwm[1]", "pwm[2]", "pwm[3]"],
    "Battery": ["voltage_v", "current_average_a", "discharged_mah"],
}

# 🔹 Logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image(os.path.join(os.path.dirname(__file__), "Rotrix-Logo.png"))
# st.logo(logo_base64, *, size="medium", link=None, icon_image=None)
st.markdown(f"""
    <div style="display: flex; position: fixed; top:50px; left: 50px; z-index:50; justify-content: left; align-items: center; padding: 1px; background-color:white; border-radius:25px;">
        <a href="http://rotrixdemo.reude.tech/" target="_blank">
            <img src="data:image/png;base64,{logo_base64}" width="180" alt="Rotrix Logo">
        </a>
    </div>
""", unsafe_allow_html=True)

# Home Page
if st.session_state.current_page == 'home':
    # Logo is already handled separately

    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #2E86C1; display: flex; align-items: center; justify-content: center; gap: 10px; margin: 0;'>
            <span>🚀</span> ROTRIX Analysis Dashboard
        </h1>
        <p style='color: #666; margin: 10px 0 0 0;'>Advanced data analysis and visualization platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Data Source Selection
    st.markdown("""
    <div style='padding: 0 0 10px 0;'>
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span style='font-size: 1.2em;'>📊</span>
            <h2 style='color: #2E86C1; margin: 0; font-size: 1.2em;'>Data Source Selection</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    data_source = st.radio(
        "Where would you like to get the data from?",
        ["A", "B", "Other"],
        help="Choose your preferred data source for analysis",
        horizontal=True
    )
    st.session_state.data_source = data_source

    if data_source in ["A", "B"]:
        st.info("🚧 This feature is currently under development. Please select 'Other' to proceed.")
        st.stop()

    # Analysis Type Selection
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <div style='display: flex; align-items: center; gap: 8px;'>
            <span style='font-size: 1.2em;'>🔍</span>
            <h2 style='color: #2E86C1; margin: 0; font-size: 1.2em;'>Analysis Type</h2>
        </div>
        <p style='color: #666; margin: 5px 0 0 0;'>Select your preferred analysis method</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e1e4e8;'>
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
                <span>📈</span>
                <h3 style='color: #2E86C1; margin: 0; font-size: 1em;'>Single File Analysis</h3>
            </div>
            <p style='color: #666; margin: 0 0 10px 0; font-size: 0.95em;'>Analyze individual data files</p>
            <ul style='color: #666; padding-left: 20px; margin: 0; font-size: 0.95em;'>
                <li>Data visualization</li>
                <li>Statistical analysis</li>
                <li>Anomaly detection</li>
                <li>Performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #e1e4e8;'>
            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 10px;'>
                <span>🔄</span>
                <h3 style='color: #2E86C1; margin: 0; font-size: 1em;'>Comparative Analysis</h3>
            </div>
            <p style='color: #666; margin: 0 0 10px 0; font-size: 0.95em;'>Compare two data files</p>
            <ul style='color: #666; padding-left: 20px; margin: 0; font-size: 0.95em;'>
                <li>Side-by-side comparison</li>
                <li>Difference analysis</li>
                <li>Performance benchmarking</li>
                <li>Trend visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
    analysis_type = st.radio(
        "Select your analysis mode:",
        ["Single File Analysis", "Comparative Analysis"],
        help="Choose the type of analysis you want to perform"
    )
    st.session_state.analysis_type = analysis_type

    # Proceed Button
    if st.button("Proceed to Analysis", 
                 help="Click to continue with your selected analysis type",
                 type="primary",
                 use_container_width=True):
        if analysis_type == "Single File Analysis":
            change_page('single_analysis')
        else:
            change_page('comparative_analysis')
        st.rerun()

# Single File Analysis Page
elif st.session_state.current_page == 'single_analysis':
    # Add vertical space and back button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        change_page('home')
        st.rerun()
        
    st.markdown("### 🔍 Single File Analysis")
    
    # File upload section
    uploaded_files = st.file_uploader("📂 Upload File", type=["csv", "ulg"], accept_multiple_files=True)
    
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        selected_file = st.selectbox("Select File", ["None"] + file_names)
        
        if selected_file != "None":
            file = uploaded_files[file_names.index(selected_file)]
            file_ext = os.path.splitext(file.name)[-1].lower()
            
            try:
                df = None
                if file_ext == ".ulg":
                    dfs, topics = load_ulog(file)
                    if topics:
                        assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
                        assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
                        selected_assessment = st.selectbox("Select Topic", options=assessment_names)
                        
                        if selected_assessment and selected_assessment != "None":
                            selected_topic = assessment_to_topic.get(str(selected_assessment))
                            if selected_topic and selected_topic in dfs:
                                df = dfs[selected_topic]
                                df = ensure_seconds_column(df)
                else:
                    df, _ = load_data(file, file_ext, "")
                    df = ensure_seconds_column(df)
                
                if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                    # Add Index column if it doesn't exist
                    if 'Index' not in df.columns:
                        df.insert(0, "Index", range(1, len(df) + 1))
                    
                    # Analysis tabs
                    tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
                    
                    with tab1:
                        st.markdown("### 📈 Plot Visualization")
                        col1, col2 = st.columns([0.2, 0.8])
                        
                        with col1:
                            st.markdown("#### 📈 Parameters")
                            # Get selected assessment/topic
                            if file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                                allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                                allowed_y_axis = [col for col in allowed_y_axis if col in df.columns]
                                if not allowed_y_axis:
                                    allowed_y_axis = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                            else:
                                allowed_y_axis = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                            
                            ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                            
                            # Set default x_axis based on file type
                            if file_ext == ".ulg":
                                default_x = 'timestamp_seconds' if 'timestamp_seconds' in ALLOWED_X_AXIS else 'Index'
                            else:
                                default_x = 'Index' if 'Index' in ALLOWED_X_AXIS else ALLOWED_X_AXIS[0]
                            
                            x_axis = st.selectbox("X-Axis", ALLOWED_X_AXIS, key="x_axis_single", index=ALLOWED_X_AXIS.index(default_x))
                            
                            # Set default y_axis based on file type and available columns
                            if file_ext == ".csv" and 'cD2detailpeak' in allowed_y_axis:
                                default_y = 'cD2detailpeak'
                            else:
                                default_y = allowed_y_axis[0] if allowed_y_axis else None
                            
                            y_axis = st.selectbox("Y-Axis", allowed_y_axis, key="y_axis_single", 
                                                index=allowed_y_axis.index(default_y) if default_y in allowed_y_axis else 0)
                            
                            if x_axis and y_axis:
                                z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider-single")
                                x_min = st.number_input("X min", value=float(df[x_axis].min()))
                                x_max = st.number_input("X max", value=float(df[x_axis].max()))
                                y_min = st.number_input("Y min", value=float(df[y_axis].min()))
                                y_max = st.number_input("Y max", value=float(df[y_axis].max()))

                        with col2:
                            if x_axis and y_axis:
                                try:
                                    # Filter data
                                    filtered_df = df[(df[x_axis] >= x_min) & (df[x_axis] <= x_max) &
                                                   (df[y_axis] >= y_min) & (df[y_axis] <= y_max)]
                                    if len(filtered_df.index) > 0:
                                        # Calculate statistics and detect abnormalities
                                        stats = filtered_df[y_axis].describe()
                                        abnormal_mask, z_scores = detect_abnormalities(filtered_df[y_axis], z_threshold)
                                        filtered_df["Z_Score"] = z_scores
                                        filtered_df["Abnormal"] = abnormal_mask
                                        abnormal_points = filtered_df[filtered_df["Abnormal"]]
                                        
                                        # Display metrics
                                        st.markdown("### 🎯 Metrics")
                                        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 1])
                                        
                                        with metric_col1:
                                            fig1 = go.Figure(go.Indicator(
                                                mode="gauge+number",
                                                value=float(stats['mean']),
                                                title={'text': "Mean Value"},
                                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                                gauge={
                                                    'axis': {'range': [float(stats['min']), float(stats['max'])],
                                                            'tickformat': '.2f'},
                                                    'bar': {'color': "darkblue"},
                                                    'steps': [
                                                        {'range': [float(stats['min']), float(stats['25%'])], 'color': "lightgray"},
                                                        {'range': [float(stats['25%']), float(stats['75%'])], 'color': "gray"},
                                                        {'range': [float(stats['75%']), float(stats['max'])], 'color': "darkgray"}
                                                    ]
                                                }
                                            ))
                                            fig1.update_layout(width=200, height=120, margin=dict(t=50, b=10))
                                            st.plotly_chart(fig1)
                                        
                                        with metric_col2:
                                            fig2 = go.Figure(go.Indicator(
                                                mode="gauge+number",
                                                value=float(stats['std']),
                                                title={'text': "Standard Deviation"},
                                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
                                                gauge={
                                                    'axis': {'range': [0, float(stats['std'] * 2)],
                                                            'tickformat': '.2f'},
                                                    'bar': {'color': "orange"},
                                                    'steps': [
                                                        {'range': [0, float(stats['std'])], 'color': "#d4f0ff"},
                                                        {'range': [float(stats['std']), float(stats['std'] * 2)], 'color': "#ffeaa7"}
                                                    ]
                                                }
                                            ))
                                            fig2.update_layout(width=200, height=120, margin=dict(t=50, b=10))
                                            st.plotly_chart(fig2)
                                        
                                        with metric_col3:
                                            abnormal_count = int(abnormal_mask.sum())
                                            fig3 = go.Figure(go.Indicator(
                                                mode="gauge+number",
                                                value=abnormal_count,
                                                title={'text': "Abnormal Points"},
                                                domain={'x': [0.15, 0.85], 'y': [0, 1]},
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
                                            fig3.update_layout(width=200, height=120, margin=dict(t=50, b=10))
                                            st.plotly_chart(fig3)
                                        
                                        # Create plot
                                        st.markdown("### 🧮 Plot Visualization")
                                        fig = go.Figure()
                                        
                                        # Add main line plot
                                        fig.add_trace(go.Scatter(
                                            x=filtered_df[x_axis],
                                            y=filtered_df[y_axis],
                                            mode='lines',
                                            name='Data'
                                        ))
                                        
                                        # Add abnormal points
                                        if not abnormal_points.empty:
                                            fig.add_trace(go.Scatter(
                                                x=abnormal_points[x_axis],
                                                y=abnormal_points[y_axis],
                                                mode='markers',
                                                marker=dict(color='red', size=8),
                                                name='Abnormal Points'
                                            ))
                                        
                                        # Add mean line
                                        mean_value = filtered_df[y_axis].mean()
                                        fig.add_hline(y=mean_value, line_dash="dash", line_color="green",
                                                    annotation_text=f"Mean: {mean_value:.2f}")
                                        
                                        # Get timestamp ticks if needed
                                        if x_axis == 'timestamp_seconds':
                                            tick_vals, tick_texts = get_timestamp_ticks(filtered_df[x_axis])
                                            fig.update_xaxes(
                                                tickvals=tick_vals,
                                                ticktext=tick_texts,
                                                title_text=get_axis_title(x_axis),
                                                type='linear'
                                            )
                                        else:
                                            fig.update_xaxes(title_text=x_axis)
                                        
                                        fig.update_layout(
                                            height=900,
                                            showlegend=True,
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.05,
                                                xanchor="center",
                                                x=0.5
                                            ),
                                            margin=dict(t=100),
                                            yaxis=dict(
                                                showticklabels=True,
                                                title=y_axis
                                            )
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No data points found in the selected range.")
                                except Exception as e:
                                    st.error(f"Error creating plot: {str(e)}")
                    
                    with tab2:
                        st.markdown("### 📋 Data Preview")
                        
                        # Create a 20-80 split layout
                        settings_col, data_col = st.columns([0.2, 0.8])
                        
                        with settings_col:
                            st.markdown("#### ⚙️ Data Settings")
                            
                            # Add data analysis settings
                            st.markdown("##### 📊 Display Options")
                            show_numeric_only = st.checkbox("Show Numeric Columns Only", value=True)
                            show_timestamps = st.checkbox("Show Timestamps", value=True)
                            
                            # Add column management
                            st.markdown("##### 🔧 Column Management")
                            if isinstance(df, pd.DataFrame):
                                df = add_remove_column(df, "Dataset")
                        
                        with data_col:
                            if isinstance(df, pd.DataFrame) and len(df.index) > 0:
                                # For ULG files after topic selection, show only selected axes
                                if file_ext == ".ulg" and selected_assessment and selected_assessment != "None" and x_axis and y_axis:
                                    # Only add Index column if it doesn't exist
                                    if 'Index' not in df.columns:
                                        df.insert(0, "Index", range(1, len(df) + 1))
                                    
                                    # Create a list of columns to display in order
                                    display_cols = []
                                    # Always start with Index
                                    if 'Index' in df.columns:
                                        display_cols.append('Index')
                                    # Add timestamp_seconds next if available
                                    if show_timestamps and 'timestamp_seconds' in df.columns:
                                        display_cols.append('timestamp_seconds')
                                    # Add selected axes if not already included
                                    if x_axis not in display_cols:
                                        display_cols.append(x_axis)
                                    if y_axis not in display_cols:
                                        display_cols.append(y_axis)
                                    
                                    # Add any numeric columns from the assessment map if available
                                    if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                        for col in ASSESSMENT_Y_AXIS_MAP[selected_assessment]:
                                            if col in df.columns and col not in display_cols and pd.api.types.is_numeric_dtype(df[col]):
                                                display_cols.append(col)
                                    
                                    # Display the filtered DataFrame
                                    st.dataframe(
                                        df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                                        use_container_width=True,
                                        height=500  # Increased height for better visibility
                                    )
                                else:
                                    # For CSV files or when no topic/axes selected
                                    display_cols = ['Index'] if show_numeric_only else list(df.columns)
                                    if show_timestamps and 'timestamp_seconds' in df.columns:
                                        display_cols.append('timestamp_seconds')
                                    
                                    if show_numeric_only:
                                        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) 
                                                      and col not in display_cols]
                                        display_cols.extend(numeric_cols)
                                    
                                    st.dataframe(
                                        df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                                        use_container_width=True,
                                        height=500
                                    )
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                df = None
    else:
        st.info("Please upload a valid data file to begin analysis.")

# Comparative Analysis Page
elif st.session_state.current_page == 'comparative_analysis':
    # Add vertical space and back button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        change_page('home')
        st.rerun()
        
    st.markdown("### 🚀 Comparative Analysis")
    
    # File upload section
    st.markdown("<h4 style='font-size:20px; color:#4B8BBE;'>🔼 Upload Benchmark & Target Files</h4>", unsafe_allow_html=True)

    # Simulate a topbar with two upload sections
    top_col1, top_col2, top_col3, top_col4 = st.columns(4)

    with top_col1:
        benchmark_files = st.file_uploader("📂 Upload Benchmark File", type=["csv", "ulg"], accept_multiple_files=True)
        benchmark_names = [f.name for f in benchmark_files] if benchmark_files else []
        if not benchmark_files:
            st.info(" Please upload a Benchmark file")
    
    with top_col2:
        validation_files = st.file_uploader("📂 Upload Target File", type=["csv", "ulg"], accept_multiple_files=True)
        validation_names = [f.name for f in validation_files] if validation_files else []
        if not validation_files:
            st.info(" Please upload a Target file")
    
    # Initialize variables
    b_df = None
    v_df = None
    b_file_ext = None
    v_file_ext = None
    b_dfs = {}
    v_dfs = {}
    selected_bench = "None"
    selected_val = "None"

    with top_col3:
        if benchmark_files:
            selected_bench = st.selectbox("Select Benchmark File", ["None"] + benchmark_names)
            if selected_bench == "None":
                st.info("📋 Select a Benchmark file to proceed")
            elif selected_bench != "None":
                b_file = benchmark_files[benchmark_names.index(selected_bench)]
                b_file_ext = os.path.splitext(b_file.name)[-1].lower()
                if b_file_ext == ".ulg":
                    b_dfs, b_topics = load_ulog(b_file)
                    if "common_topic" not in st.session_state:
                        st.session_state.common_topic = b_topics[0] if b_topics else "None"
                else:
                    df, _ = load_data(b_file, b_file_ext, key_suffix="bench")
                    if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                        b_df = df
                        st.session_state.b_df = df
        else:
            st.selectbox("Select Benchmark File", ["Please upload a file first"], disabled=True)
    
    with top_col4:
        if validation_files:
            selected_val = st.selectbox("Select Target File", ["None"] + validation_names)
            if selected_val == "None":
                st.info("📋 Select a Target file to proceed")
            elif selected_val != "None":
                v_file = validation_files[validation_names.index(selected_val)]
                v_file_ext = os.path.splitext(v_file.name)[-1].lower()
                if v_file_ext == ".ulg":
                    v_dfs, v_topics = load_ulog(v_file)
                else:
                    df, _ = load_data(v_file, v_file_ext, key_suffix="val")
                    if df is not None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
                        v_df = df
                        st.session_state.v_df = df
        else:
            st.selectbox("Select Target File", ["Please upload a file first"], disabled=True)
            
    # Show topic selection only after both ULG files are selected
    if (selected_bench != "None" and selected_val != "None" and 
        b_file_ext == ".ulg" and v_file_ext == ".ulg"):
        st.markdown("### 📊 Select Analysis Topic")
        assessment_names = ["None"] + [a for _, a in TOPIC_ASSESSMENT_PAIRS]
        assessment_to_topic = {a: t for t, a in TOPIC_ASSESSMENT_PAIRS}
        selected_assessment = st.selectbox("Select Topic", options=assessment_names, key="common_topic")
        
        if selected_assessment != "None":
            selected_topic = assessment_to_topic.get(str(selected_assessment))
            if selected_topic:
                if selected_topic in b_dfs and selected_topic in v_dfs:
                    b_df = b_dfs[selected_topic]
                    v_df = v_dfs[selected_topic]
                    st.session_state.b_df = b_dfs[selected_topic]
                    st.session_state.v_df = v_dfs[selected_topic]
                else:
                    st.warning(f"⚠️ Topic '{selected_topic}' not found in one or both files")
    elif selected_bench != "None" and selected_val != "None":
        # For non-ULG files, get data from session state
        b_df = st.session_state.get("b_df", None)
        v_df = st.session_state.get("v_df", None)

    # Show analysis tabs only if both files are loaded and selected
    if b_df is not None and v_df is not None:
        tab1, tab2 = st.tabs(["📊 Plot", "📋 Data"])
        
        # Data Tab
        with tab2:
            st.markdown("### 📋 Data Preview")
            
            # Create a 20-80 split layout
            settings_col, data_col = st.columns([0.2, 0.8])
            
            with settings_col:
                st.markdown("#### ⚙️ Data Settings")
                
                # Add data analysis settings
                st.markdown("##### 📊 Display Options")
                show_numeric_only = st.checkbox("Show Numeric Columns Only", value=True)
                show_timestamps = st.checkbox("Show Timestamps", value=True)
                
                # Add dataset selector and column management
                st.markdown("##### 🔧 Column Management")
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
                    st.markdown("#### Benchmark Data")
                    if isinstance(b_df, pd.DataFrame):
                        # Add Index if not present
                        if 'Index' not in b_df.columns:
                            b_df.insert(0, 'Index', range(1, len(b_df) + 1))
                        
                        # Ensure timestamp_seconds is present
                        b_df = ensure_seconds_column(b_df)
                        
                        # Get display columns based on settings and file type
                        display_cols = ['Index'] if show_numeric_only else list(b_df.columns)
                        if show_timestamps and 'timestamp_seconds' in b_df.columns:
                            display_cols.append('timestamp_seconds')
                        
                        # For ULG files with selected assessment
                        if b_file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                            # Add columns from assessment map
                            if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                display_cols.extend([col for col in assessment_cols if col in b_df.columns])
                        elif show_numeric_only:
                            # For CSV files or when no assessment selected, show all numeric columns
                            numeric_cols = [col for col in b_df.columns if pd.api.types.is_numeric_dtype(b_df[col]) 
                                          and col not in display_cols]
                            display_cols.extend(numeric_cols)
                        
                        # Display DataFrame with selected columns
                        st.dataframe(
                            b_df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                            use_container_width=True,
                            height=500
                        )
                    else:
                        st.warning("⚠️ Benchmark data not properly loaded")
                
                with col2:
                    st.markdown("#### Target Data")
                    if isinstance(v_df, pd.DataFrame):
                        # Add Index if not present
                        if 'Index' not in v_df.columns:
                            v_df.insert(0, 'Index', range(1, len(v_df) + 1))
                        
                        # Ensure timestamp_seconds is present
                        v_df = ensure_seconds_column(v_df)
                        
                        # Get display columns based on settings and file type
                        display_cols = ['Index'] if show_numeric_only else list(v_df.columns)
                        if show_timestamps and 'timestamp_seconds' in v_df.columns:
                            display_cols.append('timestamp_seconds')
                        
                        # For ULG files with selected assessment
                        if v_file_ext == ".ulg" and selected_assessment and selected_assessment != "None":
                            # Add columns from assessment map
                            if selected_assessment in ASSESSMENT_Y_AXIS_MAP:
                                assessment_cols = ASSESSMENT_Y_AXIS_MAP[selected_assessment]
                                display_cols.extend([col for col in assessment_cols if col in v_df.columns])
                        elif show_numeric_only:
                            # For CSV files or when no assessment selected, show all numeric columns
                            numeric_cols = [col for col in v_df.columns if pd.api.types.is_numeric_dtype(v_df[col]) 
                                          and col not in display_cols]
                            display_cols.extend(numeric_cols)
                        
                        # Display DataFrame with selected columns
                        st.dataframe(
                            v_df[list(dict.fromkeys(display_cols))],  # Remove duplicates while preserving order
                            use_container_width=True,
                            height=500
                        )
                    else:
                        st.warning("⚠️ Target data not properly loaded")
                    
        # Plot Tab
        with tab1:
            st.markdown("### 🎯 Comparative Analysis")
            col1, col2 = st.columns([0.2, 0.8])
            
            with col1:
                st.markdown("#### 📈 Parameters")
                # Get common columns
                b_df = st.session_state.get("b_df")
                v_df = st.session_state.get("v_df")
                
                if isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                    b_numeric = get_numeric_columns(b_df)
                    v_numeric = get_numeric_columns(v_df)
                    common_cols = list(set(b_numeric) & set(v_numeric))
                    ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in common_cols if col not in ["Index", "timestamp_seconds"]]
                    
                    # Define allowed columns for axes
                    if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                        allowed_y_axis = ASSESSMENT_Y_AXIS_MAP.get(selected_assessment, [])
                        # Filter columns that actually exist in the dataframe
                        allowed_y_axis = [col for col in allowed_y_axis if col in b_df.columns]
                        if not allowed_y_axis:
                            allowed_y_axis = list(b_df.columns)
                        # Filter out non-numeric columns for better visualization
                        allowed_y_axis = [col for col in allowed_y_axis if pd.api.types.is_numeric_dtype(b_df[col])]
                        ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                    else:
                        # For non-topic files, only show numeric columns
                        allowed_y_axis = [col for col in b_df.columns if pd.api.types.is_numeric_dtype(b_df[col])]
                        ALLOWED_X_AXIS = ["Index", "timestamp_seconds"] + [col for col in allowed_y_axis if col not in ["Index", "timestamp_seconds"]]
                    x_axis_options = ALLOWED_X_AXIS
                    y_axis_options = allowed_y_axis
                    # Set default x_axis to 'Index' unless a topic is selected
                    if 'selected_assessment' in locals() and isinstance(selected_assessment, str) and selected_assessment != "None":
                        default_x = 'timestamp_seconds' if 'timestamp_seconds' in x_axis_options else ('Index' if 'Index' in x_axis_options else x_axis_options[0])
                    else:
                        default_x = 'Index' if 'Index' in x_axis_options else x_axis_options[0]
                    # If CSV and cD2detailpeak exists, use as default y if x is Index
                    if b_df is not None and v_df is not None and hasattr(b_df, 'columns') and hasattr(v_df, 'columns'):
                        if 'cD2detailpeak' in b_df.columns and 'cD2detailpeak' in v_df.columns:
                            default_y = 'cD2detailpeak' if default_x == 'Index' else (y_axis_options[0] if y_axis_options else None)
                        else:
                            default_y = y_axis_options[0] if y_axis_options else None
                    else:
                        default_y = y_axis_options[0] if y_axis_options else None
                    x_axis = st.selectbox("X-Axis", x_axis_options, key="x_axis_select", index=x_axis_options.index(default_x))
                    # If user selects Index for both, force y to cD2detailpeak if available
                    if x_axis == 'Index' and 'cD2detailpeak' in y_axis_options:
                        y_axis = st.selectbox("Y-Axis", y_axis_options, key="y_axis_select", index=y_axis_options.index('cD2detailpeak'))
                    else:
                        y_axis = st.selectbox("Y-Axis", y_axis_options, key="y_axis_select", index=y_axis_options.index(default_y) if default_y in y_axis_options else 0)
                    
                    if not x_axis or not y_axis:
                        st.info("📌 Please select valid X and Y axes to compare.")
                    else:
                        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1, key="z-slider-comparative")
                        # Check if x_axis exists in DataFrame before accessing
                        if x_axis in b_df.columns:
                            x_min = st.number_input("X min", value=float(b_df[x_axis].min()), key="x_min_param")
                            x_max = st.number_input("X max", value=float(b_df[x_axis].max()), key="x_max_param")
                        else:
                            st.error(f"Selected X-axis '{x_axis}' not found in data")
                            st.stop()
                        
                        if y_axis in b_df.columns:
                            y_min = st.number_input("Y min", value=float(b_df[y_axis].min()), key="y_min_param")
                            y_max = st.number_input("Y max", value=float(b_df[y_axis].max()), key="y_max_param")
                        else:
                            st.error(f"Selected Y-axis '{y_axis}' not found in data")
                            st.stop()
                        plot_mode = st.radio("Plot Mode", ["Superimposed", "Separate"], horizontal=True, key="comparative_plot_mode")
                else:
                    st.info("Please upload both Benchmark and Target files to begin analysis.")
            
            with col2:
                # Get the DataFrames and ensure they are valid
                b_df = st.session_state.get("b_df")
                v_df = st.session_state.get("v_df")
                
                if x_axis and y_axis and isinstance(b_df, pd.DataFrame) and isinstance(v_df, pd.DataFrame):
                    try:
                        # Filter data
                        b_filtered = b_df[(b_df[x_axis] >= x_min) & (b_df[x_axis] <= x_max) &
                                        (b_df[y_axis] >= y_min) & (b_df[y_axis] <= y_max)]
                        v_filtered = v_df[(v_df[x_axis] >= x_min) & (v_df[x_axis] <= x_max) &
                                        (v_df[y_axis] >= y_min) & (v_df[y_axis] <= y_max)]

                        if x_axis == 'timestamp_seconds':
                            b_filtered, v_filtered, common_time = resample_to_common_time(b_filtered, v_filtered)

                        if plot_mode == "Superimposed":
                            fig = go.Figure()
                            # Add main line plot
                            fig.add_trace(go.Scatter(
                                x=b_filtered[x_axis],
                                y=b_filtered[y_axis],
                                mode='lines',
                                name='Benchmark'
                            ))
                            # Add validation data (Target) in green
                            fig.add_trace(go.Scatter(
                                x=v_filtered[x_axis],
                                y=v_filtered[y_axis],
                                mode='lines',
                                name='Target',
                                line=dict(color='green')
                            ))
                            # Add abnormal points to target plot
                            if len(b_filtered) > 0:
                                # Calculate difference and z-score for abnormality detection
                                diff = v_filtered[y_axis] - b_filtered[y_axis]
                                z_scores = (diff - diff.mean()) / diff.std()
                                abnormal_mask = abs(z_scores) > z_threshold
                                abnormal_points = b_filtered[abnormal_mask]
                                if not abnormal_points.empty:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=abnormal_points[x_axis], 
                                            y=abnormal_points[y_axis], 
                                            mode='markers', 
                                            marker=dict(color='red', size=8), 
                                            name='Abnormal Points'
                                        )
                                    )
                            
                            # Get timestamp ticks if needed
                            if x_axis == 'timestamp_seconds':
                                tick_vals, tick_texts = get_timestamp_ticks(b_filtered[x_axis])
                            else:
                                tick_vals, tick_texts = None, None
                                
                            fig.update_layout(
                                height=900,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.05,
                                    xanchor="center",
                                    x=0.5
                                ),
                                margin=dict(t=100),
                                xaxis=dict(
                                    showticklabels=True,
                                    title=get_axis_title(x_axis),
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    type='linear'
                                ),
                                yaxis=dict(
                                    showticklabels=True,
                                    title=y_axis
                                )
                            )
                            # Robustly set x-axis ticks/labels for timestamp_seconds
                            if x_axis == 'timestamp_seconds' and tick_vals is not None and tick_texts is not None:
                                fig.update_xaxes(
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    title_text=get_axis_title(x_axis),
                                    type='linear'
                                )
                            st.plotly_chart(fig, use_container_width=True)
                        else:  # Separate plots
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                              subplot_titles=("Benchmark", "Target"),
                                              vertical_spacing=0.2)
                            
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
                            
                            # Add abnormal points
                            if len(b_filtered) > 0:
                                # Calculate difference and z-score for abnormality detection
                                diff = v_filtered[y_axis] - b_filtered[y_axis]
                                z_scores = (diff - diff.mean()) / diff.std()
                                abnormal_mask = abs(z_scores) > z_threshold
                                abnormal_points = b_filtered[abnormal_mask]
                                if not abnormal_points.empty:
                                    fig.add_trace(
                                        go.Scatter(
                                            x=abnormal_points[x_axis],
                                            y=abnormal_points[y_axis],
                                            mode='markers',
                                            marker=dict(color='red', size=8),
                                            name='Abnormal Points'
                                        ), row=2, col=1
                                    )
                            
                            # Get timestamp ticks if needed
                            if x_axis == 'timestamp_seconds':
                                tick_vals, tick_texts = get_timestamp_ticks(b_filtered[x_axis])
                                fig.update_xaxes(
                                    tickvals=tick_vals,
                                    ticktext=tick_texts,
                                    title_text=get_axis_title(x_axis),
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
                                fig.update_xaxes(title_text=x_axis, row=1, col=1)
                                fig.update_xaxes(title_text=x_axis, row=2, col=1)
                            
                            fig.update_layout(
                                height=900,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.05,
                                    xanchor="center",
                                    x=0.5
                                ),
                                margin=dict(t=100)
                            )
                            fig.update_yaxes(title_text=y_axis, row=1, col=1)
                            fig.update_yaxes(title_text=y_axis, row=2, col=1)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error during plotting: {str(e)}")
    else:
        # Show appropriate guidance message
        if selected_bench != "None" and selected_val != "None" and b_file_ext == ".ulg" and v_file_ext == ".ulg":
            st.info("📋 Select a topic")
        elif b_df is None and v_df is None:
            st.warning("⚠️ Please upload and select both Benchmark and Target files to begin analysis")
        elif b_df is None:
            st.warning("⚠️ Please upload and select a Benchmark file to continue")
        else:
            st.warning("⚠️ Please upload and select a Target file to continue")