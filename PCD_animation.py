#Modified script to support both single file analysis and comparative assessment modes

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import StringIO, BytesIO
from PIL import Image
import base64
import open3d as o3d
import tempfile
import time
import os
import requests
import importlib.util
import json
from datetime import datetime
import zipfile



st.set_page_config(page_title="ROTRIX Dashboard", layout="wide")

class UploadedGitHubFile:
    def __init__(self, content, name, filetype): 
        self.file = BytesIO(content)
        self.name = name
        self.type = filetype
        self.size = len(content)
    def read(self, *args, **kwargs):
        return self.file.read(*args, **kwargs)
    def seek(self, *args, **kwargs):
        return self.file.seek(*args, **kwargs)

def process_url(url):
    if "github.com" in url:
        try:
            # Handle folder URL (e.g., https://github.com/username/repo/tree/main/folder)
            if "tree" in url:
                parts = url.split("tree/")
                if len(parts) < 2:
                    st.error("Invalid GitHub folder URL format. Please include 'tree/' followed by the folder path.")
                    return None
                base_url = parts[0].rstrip('/')
                path = parts[1].lstrip('/')
                url_parts = base_url.split("/")
                if len(url_parts) < 5 or url_parts[2] != "github.com":
                    st.error("Unable to parse repository from URL.")
                    return None
                repo = f"{url_parts[3]}/{url_parts[4]}"
                folder_path = path.split('/', 1)[-1] if '/' in path else path
                api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}"
                response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
                if response.status_code == 200:
                    files = [item for item in response.json() if item['type'] == 'file' and item['name'].endswith('.pcd')]
                    if not files:
                        st.warning("No .pcd files found in the folder.")
                        return None
                    file_data = {}
                    for file in files:
                        file_response = requests.get(file['download_url'])
                        if file_response.status_code == 200:
                            file_ext = os.path.splitext(file['name'])[-1].lower()
                            file_data[file['name']] = (file_response.content, file_ext)
                    return file_data if file_data else None
                else:
                    st.error(f"Failed to fetch folder contents. Status code: {response.status_code}, Message: {response.text}, API URL: {api_url}")
                    return None
            # Handle raw file URL (e.g., https://raw.githubusercontent.com/username/repo/main/file.pcd)
            elif "raw.githubusercontent.com" in url:
                file_name = url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext == ".pcd":
                    response = requests.get(url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {url}")
                        return None
            # Handle blob URL (e.g., https://github.com/username/repo/blob/main/file.pcd)
            elif "/blob/" in url:
                raw_url = url.replace("/blob/", "/raw/")
                file_name = raw_url.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext == ".pcd":
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            # Handle direct file URL (e.g., https://github.com/username/repo/filename.pcd)
            elif len(url.split("/")) > 4 and url.split("/")[4] not in ["tree", "blob", "raw"]:
                base_parts = url.split("/")
                repo = f"{base_parts[3]}/{base_parts[4]}"
                file_path = "/".join(base_parts[5:])
                raw_url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
                file_name = file_path.split("/")[-1]
                file_ext = os.path.splitext(file_name)[-1].lower()
                if file_ext == ".pcd":
                    response = requests.get(raw_url)
                    if response.status_code == 200:
                        return {file_name: (response.content, file_ext)}
                    else:
                        st.error(f"Failed to download file. Status code: {response.status_code}, URL: {raw_url}")
                        return None
            else:
                st.warning("Unsupported GitHub URL format. Please use a folder URL with 'tree/' or a raw/blob file URL.")
                return None
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")
            return None
    return None

# Function to parse PCD file
def parse_custom_pcd(filepath):
    """Parse PCD file and return DataFrame with X, Y, Temp columns"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Filter out comments and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        # Parse data into array
        data_array = np.array([list(map(float, line.split())) for line in data_lines])
        
        # Return DataFrame with first 3 columns as X, Y, Temp
        if data_array.shape[1] >= 3:
            return pd.DataFrame(data_array[:, :3], columns=['X', 'Y', 'Temp'])
        else:
            # If less than 3 columns, pad with zeros
            padded_array = np.zeros((data_array.shape[0], 3))
            padded_array[:, :data_array.shape[1]] = data_array
            return pd.DataFrame(padded_array, columns=['X', 'Y', 'Temp'])
    except Exception as e:
        st.error(f"Error parsing PCD file: {str(e)}")
        return None

def plot_pcd_3d(df, title="PCD Point Cloud Visualization", 
                point_size=2.0, color_scheme='Viridis', opacity=0.8, rotation_angle=0):
    """Create 3D scatter plot for PCD data with frame-by-frame parameters"""
    if df is None or df.empty:
        st.warning("No PCD data to plot")
        return None
    
    # Ensure we have the required columns
    required_cols = ['X', 'Y', 'Temp']
    if not all(col in df.columns for col in required_cols):
        st.error("PCD data must contain X, Y, and Temp columns")
        return None
    
    # Apply rotation if specified
    if rotation_angle != 0:
        angle_radians = np.radians(rotation_angle)
        cos_t = np.cos(angle_radians)
        sin_t = np.sin(angle_radians)
        
        # Apply rotation transformation to X and Y coordinates
        x_rot = df['X'] * cos_t - df['Y'] * sin_t
        y_rot = df['X'] * sin_t + df['Y'] * cos_t
        
        x_plot = x_rot
        y_plot = y_rot
    else:
        x_plot = df['X']
        y_plot = df['Y']
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add scatter plot with temperature as color
    fig.add_trace(go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=df['Temp'],
        mode='markers',
        marker=dict(
            size=point_size,
            color=df['Temp'],
            colorscale=color_scheme,
            opacity=opacity,
            colorbar=dict(title="Temperature")
        ),
        name='Point Cloud'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Temperature',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        showlegend=False
    )
    
    return fig

def plot_pcd_2d(df, x_col='X', y_col='Y', color_col='Temp', title="PCD 2D Visualization", 
                point_size=1.0, color_scheme='Viridis', opacity=0.7, rotation_angle=0.0):
    if df is None or df.empty:
        st.warning("No PCD data to plot")
        return None
    
    # Ensure we have the required columns
    if not all(col in df.columns for col in [x_col, y_col, color_col]):
        st.error(f"PCD data must contain {x_col}, {y_col}, and {color_col} columns")
        return None
    
    if rotation_angle != 0:
        angle_radians = np.radians(rotation_angle)
        cos_t = np.cos(angle_radians)
        sin_t = np.sin(angle_radians)
        
        # Apply rotation transformation
        x_rot = df[x_col] * cos_t - df[y_col] * sin_t
        y_rot = df[x_col] * sin_t + df[y_col] * cos_t
        
        x_plot = x_rot
        y_plot = y_rot
    else:
        x_plot = df[x_col]
        y_plot = df[y_col]
    
    # Create 2D scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        mode='markers',
        marker=dict(
            size=point_size,
            color=df[color_col],
            colorscale=color_scheme,
            opacity=opacity,
            colorbar=dict(title=color_col)
        ),
        name='Point Cloud'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=800,
        height=650,
        showlegend=False
    )
    
    return fig

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

def detect_abnormalities(series, threshold=3.0):
    """Detect abnormal points in a series using z-score threshold."""
    if len(series) < 2:  # Need at least 2 points to calculate z-score
        return pd.Series(False, index=series.index), pd.Series(0, index=series.index)
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold, z_scores

def get_numeric_columns(df):
    """Safely get numeric columns from a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        return []
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

# New utility functions for enhanced features
def export_data_to_csv(df, filename="pcd_data"):
    """Export DataFrame to CSV format"""
    csv = df.to_csv(index=False)
    return csv

def export_data_to_json(df, filename="pcd_data"):
    """Export DataFrame to JSON format"""
    json_str = df.to_json(orient='records', indent=2)
    return json_str

def create_zip_archive(files_dict, filename="pcd_analysis_export"):
    """Create a ZIP archive containing multiple files"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_name, file_content in files_dict.items():
            zip_file.writestr(file_name, file_content)
    return zip_buffer.getvalue()

def perform_statistical_analysis(df):
    """Perform comprehensive statistical analysis on PCD data"""
    if df is None or df.empty:
        return None
    
    stats = {}
    numeric_cols = get_numeric_columns(df)
    
    for col in numeric_cols:
        stats[col] = {
            'count': len(df[col]),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
            'skewness': float(df[col].skew()),
            'kurtosis': float(df[col].kurtosis())
        }
    
    return stats

def perform_clustering_analysis(df, method='kmeans', n_clusters=3, eps=0.5, min_samples=5):
    """Perform clustering analysis on PCD data"""
    if df is None or df.empty:
        return None, None
    
    # Select numeric columns for clustering
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering")
        return None, None
    
    # Prepare data for clustering
    X = df[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        st.error("Unsupported clustering method")
        return None, None
    
    # Perform clustering
    cluster_labels = clusterer.fit_predict(X_scaled)
    
    # Add cluster labels to DataFrame
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Noise points in DBSCAN
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        cluster_stats[cluster_name] = {
            'count': len(cluster_data),
            'percentage': len(cluster_data) / len(df_clustered) * 100
        }
        
        # Add statistics for each numeric column
        for col in numeric_cols:
            cluster_stats[cluster_name][f'{col}_mean'] = float(cluster_data[col].mean())
            cluster_stats[cluster_name][f'{col}_std'] = float(cluster_data[col].std())
    
    return df_clustered, cluster_stats

def perform_pca_analysis(df, n_components=2):
    """Perform Principal Component Analysis on PCD data"""
    if df is None or df.empty:
        return None, None
    
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for PCA")
        return None, None
    
    # Prepare data
    X = df[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, len(numeric_cols)))
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA results
    pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    
    # Add original columns for reference
    for col in numeric_cols:
        df_pca[col] = df[col]
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    pca_stats = {
        'explained_variance': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'components': pca.components_.tolist(),
        'feature_names': numeric_cols
    }
    
    return df_pca, pca_stats

def create_heatmap_correlation(df):
    """Create correlation heatmap for numeric columns"""
    if df is None or df.empty:
        return None
    
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        width=600,
        height=500,
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    
    return fig

def create_distribution_plots(df, columns=None):
    """Create distribution plots for selected columns"""
    if df is None or df.empty:
        return None
    
    if columns is None:
        columns = get_numeric_columns(df)
    
    if not columns:
        return None
    
    n_cols = len(columns)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=columns,
        specs=[[{"type": "histogram"} for _ in range(n_cols)]]
    )
    
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Histogram(x=df[col], name=col, nbinsx=30),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title="Distribution Plots",
        height=400,
        showlegend=False
    )
    
    return fig

def filter_data_by_conditions(df, filters):
    """Filter DataFrame based on multiple conditions"""
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    for column, condition in filters.items():
        if column in df.columns:
            if condition['type'] == 'range':
                min_val = condition.get('min', float('-inf'))
                max_val = condition.get('max', float('inf'))
                filtered_df = filtered_df[
                    (filtered_df[column] >= min_val) & 
                    (filtered_df[column] <= max_val)
                ]
            elif condition['type'] == 'percentile':
                lower_percentile = condition.get('lower', 0)
                upper_percentile = condition.get('upper', 100)
                lower_val = df[column].quantile(lower_percentile / 100)
                upper_val = df[column].quantile(upper_percentile / 100)
                filtered_df = filtered_df[
                    (filtered_df[column] >= lower_val) & 
                    (filtered_df[column] <= upper_val)
                ]
    
    return filtered_df

def create_monitoring_dashboard(df, file_name):
    """Create a real-time monitoring dashboard for PCD data"""
    if df is None or df.empty:
        return None
    
    # Create metrics
    total_points = len(df)
    numeric_cols = get_numeric_columns(df)
    
    # Calculate alerts
    alerts = []
    if 'Temp' in df.columns:
        temp_mean = df['Temp'].mean()
        temp_std = df['Temp'].std()
        temp_outliers = df[df['Temp'] > temp_mean + 2*temp_std]
        if len(temp_outliers) > 0:
            alerts.append(f"⚠️ {len(temp_outliers)} temperature outliers detected")
    
    # Create monitoring layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Points", f"{total_points:,}")
    
    with col2:
        if 'Temp' in df.columns:
            st.metric("🌡️ Avg Temperature", f"{df['Temp'].mean():.2f}°C")
    
    with col3:
        if 'X' in df.columns and 'Y' in df.columns:
            coverage_area = (df['X'].max() - df['X'].min()) * (df['Y'].max() - df['Y'].min())
            st.metric("📐 Coverage Area", f"{coverage_area:.2f}")
    
    with col4:
        st.metric("📈 Data Quality", f"{len(numeric_cols)}/3 columns")
    
    # Display alerts
    if alerts:
        st.markdown("### 🚨 Alerts")
        for alert in alerts:
            st.warning(alert)
    
    # Real-time statistics
    st.markdown("### 📈 Live Statistics")
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        if 'Temp' in df.columns:
            st.markdown("**Temperature Statistics**")
            temp_stats = {
                "Min": f"{df['Temp'].min():.2f}°C",
                "Max": f"{df['Temp'].max():.2f}°C",
                "Mean": f"{df['Temp'].mean():.2f}°C",
                "Std": f"{df['Temp'].std():.2f}°C"
            }
            for key, value in temp_stats.items():
                st.write(f"{key}: {value}")
    
    with stats_col2:
        if 'X' in df.columns and 'Y' in df.columns:
            st.markdown("**Spatial Statistics**")
            spatial_stats = {
                "X Range": f"{df['X'].min():.2f} to {df['X'].max():.2f}",
                "Y Range": f"{df['Y'].min():.2f} to {df['Y'].max():.2f}",
                "Density": f"{total_points/((df['X'].max()-df['X'].min())*(df['Y'].max()-df['Y'].min())):.2f} pts/unit²"
            }
            for key, value in spatial_stats.items():
                st.write(f"{key}: {value}")

def load_pcd(file):
    """Load PCD file and return DataFrame"""
    try:
        # Create a temporary file to store the content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcd') as tmp_file:
            # If file is a string (path), read directly
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    tmp_file.write(f.read())
            else:
                # If file is a file object, write its content
                file.seek(0)
                tmp_file.write(file.read())
            tmp_file.flush()
            
            # Parse the PCD file
            df = parse_custom_pcd(tmp_file.name)
            return df
    except Exception as e:
        st.error(f"Error loading PCD file: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_file.name)
        except:
            pass

def load_data(file, filetype, key_suffix):
    """Load data from file - PCD only."""
    try:
        if filetype == ".pcd":
            df_pcd = load_pcd(file)
            if df_pcd is not None and not df_pcd.empty:
                # Add Index column for PCD files
                df_pcd.insert(0, 'Index', range(1, len(df_pcd) + 1))
            return df_pcd, None
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None



def add_remove_column(target_df, df_name="DataFrame"):
    if target_df is None or target_df.empty:
        st.warning(f"⚠ {df_name} is empty or not loaded.")
        return target_df
    
    # New Column Section with columns
    st.markdown("<p style='font-size: 12px; margin: 0;'>🧮 New Column</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        new_col_name = st.text_input("Column Name", key=f"{df_name}_add", label_visibility="collapsed", placeholder="Column Name")
    with col2:
        custom_formula = st.text_input("Formula", key=f"{df_name}_formula", label_visibility="collapsed", placeholder="Formula (e.g., x*y)")

    if st.button("Add Column", key=f"add_btn_{df_name}", use_container_width=True):
        try:
            if new_col_name and custom_formula:
                target_df[new_col_name] = target_df.eval(custom_formula)
                st.success(f"Added: {new_col_name}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Remove Column Section
    st.markdown("<p style='font-size: 12px; margin: 0;'>🗑 Remove Column</p>", unsafe_allow_html=True)
    columns_to_drop = st.multiselect("Select columns", target_df.columns, key=f"{df_name}_drop", label_visibility="collapsed")
    if st.button("Remove Selected", key=f"remove_btn_{df_name}", use_container_width=True):
        if columns_to_drop:
            target_df.drop(columns=columns_to_drop, inplace=True)
            st.success(f"Removed {len(columns_to_drop)} column(s)")
            
    # Rename Column Section
    st.markdown("<p style='font-size: 12px; margin: 0;'>✏ Rename Column</p>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        rename_col = st.selectbox("Select column", target_df.columns, key=f"{df_name}_rename_col", label_visibility="collapsed")
    with col4:
        new_name = st.text_input("New name", key=f"{df_name}_rename_input", label_visibility="collapsed", placeholder="New name")

    if st.button("Rename Column", key=f"rename_btn_{df_name}", use_container_width=True):
        if rename_col and new_name:
            target_df.rename(columns={rename_col: new_name}, inplace=True)
            st.success(f"Renamed: {rename_col} → {new_name}")

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



# 🔹 Logo
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- Session state initializations (safe for reruns) ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'files_submitted' not in st.session_state:
    st.session_state.files_submitted = False
if 'show_upload_area' not in st.session_state:
    st.session_state.show_upload_area = True
if 'file_rename_mode' not in st.session_state:
    st.session_state.file_rename_mode = {}
if 'file_share_mode' not in st.session_state:
    st.session_state.file_share_mode = {}
if 'single_file_selection' not in st.session_state:
    st.session_state.single_file_selection = "None"
if 'benchmark_file_selection' not in st.session_state:
    st.session_state.benchmark_file_selection = "None"
if 'target_file_selection' not in st.session_state:
    st.session_state.target_file_selection = "None"
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

# --- Home Page: 3-tab layout for PCD files only ---
if st.session_state.current_page == 'home':
    st.markdown("""
    <style>
    .fixed-header {
        position: fixed;
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
        max-width: 350px;
        border: 1px solid #e0e0e0;
    }
    .fixed-header h1 {
        color: #2E86C1;
        margin: 0 0 2px 0;
        font-size: 1.35rem;
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
    .main .block-container {
        padding-top: 40px !important;
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
    .file-type-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .file-type-badge.pcd {
        background: #e3f2fd;
        color: #1565c0;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class=\"fixed-header\">\n        <h1><span class=\"rocket-icon\">🚀</span> PCD Data Assessment</h1>\n    </div>\n    """, unsafe_allow_html=True)

    show_upload = st.session_state.show_upload_area or not st.session_state.files_submitted
    if show_upload:
        st.markdown("<h3 style='text-align: center; color: #2E86C1; margin-bottom: 30px;'>📁 File Upload & Management</h3>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["💾 Local & GitHub", "🚀 ROTRIX Account", "📁 Shared Files"])
        # --- Tab 1: Local & GitHub ---
        with tab1:
            github_col, upload_col = st.columns([1, 1])
            with github_col:
                st.markdown("""
                <div style='padding: 12px 16px; background: #f0f8ff; border-radius: 8px; border: 1.5px solid #b3d8fd; margin-bottom: 8px;'>
                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 2px;'>
                        <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='22' style='margin-right: 4px;'/>
                        <span style='font-size: 1.08rem; font-weight: 700; color: #24292f;'>GitHub</span>
                        <span style='font-size: 0.98rem; color: #2980b9; margin-left: 6px;'>(<a style='color:#2980b9; text-decoration:underline; cursor:pointer;' href='#'>.pcd</a>)</span>
                    </div>
                    <div style='font-size: 0.98rem; color: #444;'>
                        Paste a <b>GitHub <span style='font-weight:700;'>raw/blob/folder URL</span></b> to fetch .pcd files.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.session_state.get("clear_github_url_input", False):
                    st.session_state.github_url_input = ""
                    st.session_state.clear_github_url_input = False
                github_col, fetch_col = st.columns([5, 1])
                with fetch_col:
                    fetch_github = st.button("Fetch", key="fetch_github_btn", use_container_width=True)
                with github_col:
                    github_url = st.text_input("GitHub URL (raw, blob, or folder)", key="github_url_input", label_visibility="collapsed", placeholder="e.g. https://github.com/user/repo/blob/main/data.pcd")
                if fetch_github and github_url:
                    result = process_url(github_url)
                    if result:
                        existing_names = [f.name for f in st.session_state.uploaded_files]
                        for file_name, (file_content, file_ext) in result.items():
                            if file_name not in existing_names:
                                filetype = "application/octet-stream"
                                file_like = UploadedGitHubFile(file_content, file_name, filetype)
                                st.session_state.uploaded_files.append(file_like)
                        st.session_state.clear_github_url_input = True  # Set flag to clear input on next rerun
                        st.rerun()
                    else:
                        st.warning("No valid .pcd files found at the provided URL.")
            with upload_col:
                uploaded_files = st.file_uploader(
                    "Choose PCD files to upload", 
                    type=["pcd"], 
                    key="pcd_uploader", 
                    label_visibility="collapsed", 
                    accept_multiple_files=True,
                    help="Drag and drop .pcd files here or click to browse"
                )
            if uploaded_files:
                new_files_added = False
                existing_names = [f.name for f in st.session_state.uploaded_files]
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in existing_names:
                        st.session_state.uploaded_files.append(uploaded_file)
                        new_files_added = True
                if new_files_added:
                    st.rerun()
            if st.session_state.uploaded_files:
                st.markdown("<h4 style='margin-top: 0px; color: #495057;'>📋 Uploaded Files</h4>", unsafe_allow_html=True)
                for i, file in enumerate(st.session_state.uploaded_files):
                    file_cols = st.columns([9, 1, 1, 1])
                    with file_cols[0]:
                        st.markdown(f"<div style='font-weight: 600; color: #495057;'>📄 {file.name} <span style='font-size: 12px; color: #6c757d; margin-left: 10px;'>Size: {file.size / (1024*1024):.2f} MB</span></div>", unsafe_allow_html=True)
                    with file_cols[1]:
                        if st.button("✏️", key=f"rename_btn_{i}", use_container_width=True, help="Rename"):
                            st.session_state.file_rename_mode[i] = not st.session_state.file_rename_mode.get(i, False)
                            st.rerun()
                    with file_cols[2]:
                        if st.button("➦", key=f"share_btn_{i}", use_container_width=True, help="Share"):
                            st.session_state.file_share_mode[i] = not st.session_state.file_share_mode.get(i, False)
                            st.rerun()
                    with file_cols[3]:
                        if st.button("🗑️", key=f"remove_{i}", use_container_width=True, help="Remove file"):
                            st.session_state.uploaded_files.pop(i)
                            st.rerun()
                    # Rename UI
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
                    # Share UI (placeholder)
                    if st.session_state.file_share_mode.get(i, False):
                        with st.container():
                            st.markdown("**➦ Share File**")
                            share_option = st.selectbox(
                                "Select sharing option:",
                                ["Public Link", "ROTRIX Team", "Email"],
                                key=f"share_option_{i}"
                            )
                            email_address = None
                            if share_option == "Email":
                                email_address = st.text_input(
                                    "Recipient Email:",
                                    key=f"email_input_{i}",
                                    placeholder="Enter recipient email"
                                )
                                col_share1, col_share2 = st.columns([1, 1])
                                with col_share1:
                                    if st.button("✅ Share", key=f"confirm_share_{i}", use_container_width=True):
                                        if email_address:
                                            st.success(f"File '{file.name}' shared via Email to: {email_address}")
                                        else:
                                            st.warning("Please enter a recipient email address.")
                                            st.stop()
                                        st.session_state.file_share_mode[i] = False
                                        st.rerun()
                                with col_share2:
                                    if st.button("❌ Cancel", key=f"cancel_share_{i}", use_container_width=True):
                                        st.session_state.file_share_mode[i] = False
                                        st.rerun()
                            elif share_option in ["Public Link", "ROTRIX Team"]:
                                st.info(f"🔧 {share_option} sharing is a Work in Progress.")
                if st.button("✅ Submit Files for Analysis", type="primary", use_container_width=True):
                    st.session_state.files_submitted = True
                    st.session_state.show_upload_area = False
                    st.rerun()
            else:
                st.info("Upload your .pcd files to begin analysis.")
        # --- Tab 2: ROTRIX Account ---
        with tab2:
            # Show Rotrix logo at the top of the section instead of the emoji
            logo_base64 = get_base64_image("Rotrix-Logo.png")
            st.markdown(f'''
            <div class="upload-zone">
                <img src="data:image/png;base64,{logo_base64}" width="215" style="margin-bottom: 20px;" />
                <h4 style="color: #6c757d; margin-bottom: 10px;">ROTRIX Account Integration</h4>
                <p style="color: #adb5bd; margin-bottom: 20px;">Connect to your ROTRIX account to access your .pcd files</p>
                <div style="display: flex; gap: 10px; justify-content: center;">
                    <button style="background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.background='#0056b3'" onmouseout="this.style.background='#007bff'">
                        🔗 Connect ROTRIX Account
                    </button>
                    <button style="background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.background='#545b62'" onmouseout="this.style.background='#6c757d'">
                        📋 View Account Info
                    </button>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Mock ROTRIX files (for demonstration)
            st.markdown("<h5 style='margin-top: 30px; color: #6c757d;'>📁 Recent ROTRIX Files (PCD only)</h5>", unsafe_allow_html=True)
            
            # Create mock ROTRIX files for demonstration
            mock_rotrix_files = [
                {"name": "flight_data_001.pcd", "size": 2048, "date": "2024-01-15", "type": "pcd"},
                {"name": "battery_analysis.pcd", "size": 512, "date": "2024-01-14", "type": "pcd"},
                {"name": "performance_test.pcd", "size": 4096, "date": "2024-01-13", "type": "pcd"}
            ]
            
            for file_info in mock_rotrix_files:
                file_ext = file_info["type"]
                file_type_badge = f"<span class='file-type-badge {file_ext}'>{file_ext}</span>"
                
                st.markdown(f"""
                <div class="file-preview-card" style="opacity: 0.7;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="flex: 1;">
                            <div style="font-weight: 600; color: #495057; margin-bottom: 4px;">
                                📄 {file_info["name"]}
                            </div>
                            <div style="font-size: 12px; color: #6c757d; margin-bottom: 8px;">
                                Size: {file_info["size"]} KB | Date: {file_info["date"]} {file_type_badge}
                            </div>
                            <div style="font-size: 11px; color: #adb5bd;">
                                🔒 ROTRIX Account • Requires authentication
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.info("🔒 This feature requires ROTRIX account authentication")
        
        # --- Tab 3: Shared Files ---
        with tab3:
            # Placeholder for shared files
            st.markdown("""
            <div class="upload-zone">
                <div style="font-size: 48px; margin-bottom: 20px;">📁</div>
                <h4 style="color: #6c757d; margin-bottom: 10px;">Shared Files</h4>
                <p style="color: #adb5bd; margin-bottom: 20px;">Access .pcd files shared with you by your team</p>
                <div style="display: flex; gap: 10px; justify-content: center;">
                    <button style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.background='#218838'" onmouseout="this.style.background='#28a745'">
                        🔍 Browse Shared Files
                    </button>
                    <button style="background: #17a2b8; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.background='#138496'" onmouseout="this.style.background='#17a2b8'">
                        👥 Manage Permissions
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Mock shared files (for demonstration)
            st.markdown("<h5 style='margin-top: 30px; color: #6c757d;'>📁 Recently Shared (PCD only)</h5>", unsafe_allow_html=True)
            
            # Create mock shared files for demonstration
            mock_shared_files = [
                {"name": "team_analysis.pcd", "size": 1536, "shared_by": "John Doe", "date": "2024-01-15", "type": "pcd"},
                {"name": "comparison_data.pcd", "size": 768, "shared_by": "Jane Smith", "date": "2024-01-14", "type": "pcd"},
                {"name": "test_results.pcd", "size": 3072, "shared_by": "Mike Johnson", "date": "2024-01-13", "type": "pcd"}
            ]
            
            for file_info in mock_shared_files:
                file_ext = file_info["type"]
                file_type_badge = f"<span class='file-type-badge {file_ext}'>{file_ext}</span>"
                
                st.markdown(f"""
                <div class="file-preview-card" style="opacity: 0.7;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="flex: 1;">
                            <div style="font-weight: 600; color: #495057; margin-bottom: 4px;">
                                📄 {file_info["name"]}
                            </div>
                            <div style="font-size: 12px; color: #6c757d; margin-bottom: 8px;">
                                Size: {file_info["size"]} KB | Shared by: {file_info["shared_by"]} {file_type_badge}
                            </div>
                            <div style="font-size: 11px; color: #adb5bd;">
                                📅 Shared on {file_info["date"]} • Requires permissions
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Analysis Type Selection after files are submitted
    if st.session_state.files_submitted and not st.session_state.show_upload_area:
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        col1, col2 = st.columns([8,1])
        with col1:
            analysis_choice = st.radio(
                "Choose Analysis Type",
                ["Single File Analysis", "Video Generation"],
                index=0 if st.session_state.analysis_type != "Single File Analysis" else 1,
                key="analysis_type_radio",
                horizontal=True
            )
            if analysis_choice == "Video Generation":
                st.session_state.analysis_type = "Frame-by-Frame Video"
            else:
                st.session_state.analysis_type = "Single Analysis"
        with col2:
            if st.button("+ Upload", key="plus_upload_btn", use_container_width=True):
                st.session_state.files_submitted = False
                st.session_state.show_upload_area = True
                st.session_state.analysis_type = None
                st.rerun()

    # --- Analysis logic after type selection ---
    if st.session_state.analysis_type == "Single Analysis":
        file_options = [f.name for f in st.session_state.uploaded_files]
        selected_file = st.selectbox("Select File for Analysis", file_options, key="single_file_selector")
        if not file_options:
            st.info("📋 Please upload files to begin analysis.")
        else:
            with st.container():
                # st.markdown("#### Single File Analysis")
                tab_plot, tab_data = st.tabs(["Plot & Parameters", "Data"])
                with tab_plot:
                    col_plot, col_params = st.columns([3, 1])
                    with col_params:
                        st.markdown("#### Parameters")
                        rotation_angle = st.number_input("Rotation Angle (degrees)", min_value=0, max_value=360, value=0, step=1, key="rotation_angle")
                        zscore_threshold = st.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="zscore_threshold")
                        # Axis limits controls
                        file_obj_for_limits = next((f for f in st.session_state.uploaded_files if f.name == selected_file), None)
                        df_limits, _ = load_data(file_obj_for_limits, ".pcd", "limits") if file_obj_for_limits else (None, None)
                        if df_limits is not None and not df_limits.empty:
                            x_min_data, x_max_data = float(df_limits['X'].min()), float(df_limits['X'].max())
                            y_min_data, y_max_data = float(df_limits['Y'].min()), float(df_limits['Y'].max())
                        else:
                            x_min_data, x_max_data, y_min_data, y_max_data = 0.0, 1.0, 0.0, 1.0
                        # Initialize session state for axis limits (before creating widgets)
                        axis_key = f"axis_limits_{selected_file}"
                        if axis_key not in st.session_state:
                            st.session_state[axis_key] = {
                                'x_min': x_min_data,
                                'x_max': x_max_data,
                                'y_min': y_min_data,
                                'y_max': y_max_data
                            }
                        
                        st.write('---')
                        
                        def reset_x_axis_callback():
                            st.session_state[axis_key]['x_min'] = x_min_data
                            st.session_state[axis_key]['x_max'] = x_max_data
                            st.rerun()
                            
                        def reset_y_axis_callback():
                            st.session_state[axis_key]['y_min'] = y_min_data
                            st.session_state[axis_key]['y_max'] = y_max_data
                            st.rerun()
                        
                        st.markdown('**X Axis Limits**')
                        x_min_col, x_max_col, x_reset_col = st.columns([6, 6, 2])
                        with x_min_col:
                            x_min = st.number_input(
                                'X min', 
                                value=st.session_state[axis_key]['x_min'], 
                                key=f'x_min_{selected_file}', 
                                min_value=float(x_min_data), 
                                max_value=float(x_max_data)
                            )
                        with x_max_col:
                            x_max = st.number_input(
                                'X max', 
                                value=st.session_state[axis_key]['x_max'], 
                                key=f'x_max_{selected_file}', 
                                min_value=float(x_min_data), 
                                max_value=float(x_max_data)
                            )
                        with x_reset_col:
                            st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
                            st.button("↺", key=f"reset_x_axis_{selected_file}", help="Reset X-axis range", on_click=reset_x_axis_callback)
                        
                        st.markdown('**Y Axis Limits**')
                        y_min_col, y_max_col, y_reset_col = st.columns([6, 6, 2])
                        with y_min_col:
                            y_min = st.number_input(
                                'Y min', 
                                value=st.session_state[axis_key]['y_min'], 
                                key=f'y_min_{selected_file}', 
                                min_value=float(y_min_data), 
                                max_value=float(y_max_data)
                            )
                        with y_max_col:
                            y_max = st.number_input(
                                'Y max', 
                                value=st.session_state[axis_key]['y_max'], 
                                key=f'y_max_{selected_file}', 
                                min_value=float(y_min_data), 
                                max_value=float(y_max_data)
                            )
                        with y_reset_col:
                            st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
                            st.button("↺", key=f"reset_y_axis_{selected_file}", help="Reset Y-axis range", on_click=reset_y_axis_callback)
                    with col_plot:
                        file_obj = next((f for f in st.session_state.uploaded_files if f.name == selected_file), None)
                        if file_obj:
                            df, _ = load_data(file_obj, ".pcd", "single")
                            if df is not None and not df.empty:
                                # Filter by axis limits
                                axis_key = f"axis_limits_{selected_file}"
                                x_min = st.session_state[axis_key]['x_min']
                                x_max = st.session_state[axis_key]['x_max']
                                y_min = st.session_state[axis_key]['y_min']
                                y_max = st.session_state[axis_key]['y_max']
                                df = df[(df['X'] >= x_min) & (df['X'] <= x_max) & (df['Y'] >= y_min) & (df['Y'] <= y_max)]
                                abnormal, z_scores = detect_abnormalities(df['Temp'], threshold=zscore_threshold) if 'Temp' in df.columns else (None, None)
                                # st.markdown("### 2D Scatter Plot")
                                fig2d = go.Figure()
                                fig2d.add_trace(go.Scatter(
                                    x=df['X'],
                                    y=df['Y'],
                                    mode='markers',
                                    marker=dict(
                                        size=6,
                                        color=df['Temp'] if 'Temp' in df.columns else None,
                                        colorscale='Viridis',
                                        opacity=0.7,
                                        colorbar=dict(title='Temp') if 'Temp' in df.columns else None
                                    ),
                                    name='All Points'
                                ))
                                if abnormal is not None and abnormal.any():
                                    fig2d.add_trace(go.Scatter(
                                        x=df.loc[abnormal, 'X'],
                                        y=df.loc[abnormal, 'Y'],
                                        mode='markers',
                                        marker=dict(size=5, color='red', symbol='circle'),
                                        name='Abnormal Points'
                                    ))
                                fig2d.update_layout(
                                    title="PCD 2D Visualization (Abnormal points in red)",
                                    xaxis_title='X',
                                    yaxis_title='Y',
                                    width=800,
                                    height=650,
                                    showlegend=True
                                )
                                st.plotly_chart(fig2d, use_container_width=True)
                                if abnormal is not None and abnormal.any():
                                    st.markdown("### Abnormal Points Table")
                                    st.dataframe(df[abnormal], use_container_width=True)
                            else:
                                st.warning("Could not load or parse the selected PCD file.")
                with tab_data:
                    selected_file = st.session_state.get("single_file_selector", file_options[0])
                    file_obj = next((f for f in st.session_state.uploaded_files if f.name == selected_file), None)
                    if file_obj:
                        df, _ = load_data(file_obj, ".pcd", "single_data_tab")
                        if df is not None and not df.empty:
                            st.markdown("### Data Table")
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("Could not load or parse the selected PCD file.")
                
                # New tabs for advanced features
                tab_analytics, tab_export, tab_filter, tab_monitor = st.tabs(["📊 Analytics", "💾 Export", "🔍 Filter", "📈 Monitor"])
                
                with tab_analytics:
                    file_obj = next((f for f in st.session_state.uploaded_files if f.name == selected_file), None)
                    if file_obj:
                        df, _ = load_data(file_obj, ".pcd", "analytics")
                        if df is not None and not df.empty:
                            # Statistical Analysis
                            st.markdown("### 📈 Statistical Analysis")
                            stats = perform_statistical_analysis(df)
                            if stats:
                                stats_df = pd.DataFrame(stats).T
                                st.dataframe(stats_df, use_container_width=True)
                            
                            # Correlation Analysis
                            st.markdown("### 🔗 Correlation Analysis")
                            corr_fig = create_heatmap_correlation(df)
                            if corr_fig:
                                st.plotly_chart(corr_fig, use_container_width=True)
                            
                            # Distribution Plots
                            st.markdown("### 📊 Distribution Plots")
                            dist_fig = create_distribution_plots(df)
                            if dist_fig:
                                st.plotly_chart(dist_fig, use_container_width=True)
                            
                            # Clustering Analysis
                            st.markdown("### 🎯 Clustering Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                clustering_method = st.selectbox("Clustering Method", ["kmeans", "dbscan"], key="clustering_method")
                                if clustering_method == "kmeans":
                                    n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="n_clusters")
                                else:
                                    eps = st.slider("Epsilon", 0.1, 2.0, 0.5, step=0.1, key="eps")
                                    min_samples = st.slider("Min Samples", 2, 20, 5, key="min_samples")
                            
                            with col2:
                                if st.button("Perform Clustering", key="perform_clustering"):
                                    if clustering_method == "kmeans":
                                        df_clustered, cluster_stats = perform_clustering_analysis(df, method='kmeans', n_clusters=n_clusters)
                                    else:
                                        df_clustered, cluster_stats = perform_clustering_analysis(df, method='dbscan', eps=eps, min_samples=min_samples)
                                    
                                    if df_clustered is not None:
                                        st.success("Clustering completed!")
                                        st.dataframe(df_clustered, use_container_width=True)
                                        
                                        if cluster_stats:
                                            st.markdown("#### Cluster Statistics")
                                            cluster_stats_df = pd.DataFrame(cluster_stats).T
                                            st.dataframe(cluster_stats_df, use_container_width=True)
                            
                            # PCA Analysis
                            st.markdown("### 🔍 Principal Component Analysis")
                            n_components = st.slider("Number of Components", 2, min(5, len(get_numeric_columns(df))), 2, key="pca_components")
                            if st.button("Perform PCA", key="perform_pca"):
                                df_pca, pca_stats = perform_pca_analysis(df, n_components=n_components)
                                if df_pca is not None:
                                    st.success("PCA completed!")
                                    st.dataframe(df_pca, use_container_width=True)
                                    
                                    if pca_stats:
                                        st.markdown("#### PCA Statistics")
                                        st.write(f"Explained Variance: {[f'{v:.3f}' for v in pca_stats['explained_variance']]}")
                                        st.write(f"Cumulative Variance: {[f'{v:.3f}' for v in pca_stats['cumulative_variance']]}")
                        else:
                            st.warning("Could not load or parse the selected PCD file.")
                
                with tab_export:
                    file_obj = next((f for f in st.session_state.uploaded_files if f.name == selected_file), None)
                    if file_obj:
                        df, _ = load_data(file_obj, ".pcd", "export")
                        if df is not None and not df.empty:
                            st.markdown("### 💾 Data Export Options")
                            
                            # Export format selection
                            export_format = st.selectbox("Export Format", ["CSV", "JSON", "ZIP Archive"], key="export_format")
                            
                            if export_format == "CSV":
                                csv_data = export_data_to_csv(df, f"{selected_file}_data")
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"{selected_file}_data.csv",
                                    mime="text/csv"
                                )
                            
                            elif export_format == "JSON":
                                json_data = export_data_to_json(df, f"{selected_file}_data")
                                st.download_button(
                                    label="📥 Download JSON",
                                    data=json_data,
                                    file_name=f"{selected_file}_data.json",
                                    mime="application/json"
                                )
                            
                            elif export_format == "ZIP Archive":
                                # Create multiple files for ZIP
                                csv_data = export_data_to_csv(df, f"{selected_file}_data")
                                json_data = export_data_to_json(df, f"{selected_file}_data")
                                stats = perform_statistical_analysis(df)
                                stats_json = json.dumps(stats, indent=2)
                                
                                files_dict = {
                                    f"{selected_file}_data.csv": csv_data,
                                    f"{selected_file}_data.json": json_data,
                                    f"{selected_file}_statistics.json": stats_json,
                                    "export_info.txt": f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nFile: {selected_file}\nTotal Points: {len(df)}"
                                }
                                
                                zip_data = create_zip_archive(files_dict, f"{selected_file}_analysis")
                                st.download_button(
                                    label="📥 Download ZIP Archive",
                                    data=zip_data,
                                    file_name=f"{selected_file}_analysis.zip",
                                    mime="application/zip"
                                )
                            
                            # Preview of data to be exported
                            st.markdown("### 📋 Export Preview")
                            st.dataframe(df.head(100), use_container_width=True)
                            st.info(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
                        else:
                            st.warning("Could not load or parse the selected PCD file.")
                
                with tab_filter:
                    file_obj = next((f for f in st.session_state.uploaded_files if f.name == selected_file), None)
                    if file_obj:
                        df, _ = load_data(file_obj, ".pcd", "filter")
                        if df is not None and not df.empty:
                            st.markdown("### 🔍 Advanced Filtering")
                            
                            # Filter type selection
                            filter_type = st.selectbox("Filter Type", ["Range Filter", "Percentile Filter"], key="filter_type")
                            
                            filters = {}
                            numeric_cols = get_numeric_columns(df)
                            
                            if filter_type == "Range Filter":
                                st.markdown("#### Set Range Filters")
                                for col in numeric_cols:
                                    col_min, col_max = float(df[col].min()), float(df[col].max())
                                    st.markdown(f"**{col}**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        min_val = st.number_input(f"Min {col}", value=col_min, key=f"min_{col}")
                                    with col2:
                                        max_val = st.number_input(f"Max {col}", value=col_max, key=f"max_{col}")
                                    
                                    if min_val != col_min or max_val != col_max:
                                        filters[col] = {
                                            'type': 'range',
                                            'min': min_val,
                                            'max': max_val
                                        }
                            
                            elif filter_type == "Percentile Filter":
                                st.markdown("#### Set Percentile Filters")
                                for col in numeric_cols:
                                    st.markdown(f"**{col}**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        lower_percentile = st.slider(f"Lower Percentile {col}", 0, 100, 0, key=f"lower_{col}")
                                    with col2:
                                        upper_percentile = st.slider(f"Upper Percentile {col}", 0, 100, 100, key=f"upper_{col}")
                                    
                                    if lower_percentile > 0 or upper_percentile < 100:
                                        filters[col] = {
                                            'type': 'percentile',
                                            'lower': lower_percentile,
                                            'upper': upper_percentile
                                        }
                            
                            # Apply filters
                            if filters:
                                if st.button("Apply Filters", key="apply_filters"):
                                    filtered_df = filter_data_by_conditions(df, filters)
                                    st.success(f"Filtered data: {len(filtered_df)} points (from {len(df)} original)")
                                    
                                    # Show filtered data
                                    st.markdown("### 📊 Filtered Data")
                                    st.dataframe(filtered_df, use_container_width=True)
                                    
                                    # Show filter summary
                                    st.markdown("### 📋 Filter Summary")
                                    for col, condition in filters.items():
                                        if condition['type'] == 'range':
                                            st.write(f"{col}: {condition['min']} to {condition['max']}")
                                        elif condition['type'] == 'percentile':
                                            st.write(f"{col}: {condition['lower']}th to {condition['upper']}th percentile")
                            else:
                                st.info("No filters applied. Select filter conditions above.")
                        else:
                            st.warning("Could not load or parse the selected PCD file.")
                
                with tab_monitor:
                    file_obj = next((f for f in st.session_state.uploaded_files if f.name == selected_file), None)
                    if file_obj:
                        df, _ = load_data(file_obj, ".pcd", "monitor")
                        if df is not None and not df.empty:
                            st.markdown("### 📈 Real-Time Monitoring Dashboard")
                            create_monitoring_dashboard(df, selected_file)
                            
                            # Auto-refresh option
                            if st.button("�� Refresh Data", key="refresh_monitor"):
                                st.rerun()
                        else:
                            st.warning("Could not load or parse the selected PCD file.")
    elif st.session_state.analysis_type == "Frame-by-Frame Video":
        if not st.session_state.uploaded_files:
            st.info("📋 Please upload files to begin 3D stacked visualization.")
        else:
            st.markdown("### 3D Stacked PCD Visualization")
            all_points = []
            files_sorted = sorted([(f.name, f) for f in st.session_state.uploaded_files], key=lambda x: x[0])
            
            for idx, (fname, file_obj) in enumerate(files_sorted):
                df, _ = load_data(file_obj, ".pcd", f"frame_{idx}")
                if df is not None and not df.empty:
                    df = df.copy()
                    df['Z'] = idx  # Stack files along Z-axis
                    df['File'] = fname
                    all_points.append(df)
                else:
                    st.warning(f"⚠️ Could not load: {fname}")
            
            if all_points:
                # Combine all data
                df_all = pd.concat(all_points, ignore_index=True)
                st.info(f"📊 Total points: {len(df_all)} from {len(all_points)} files")
                
                # Create 3D stacked visualization
                fig = go.Figure()
                
                # Add scatter plot with temperature coloring
                fig.add_trace(go.Scatter3d(
                    x=df_all['X'],
                    y=df_all['Y'],
                    z=df_all['Z'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=df_all['Temp'] if 'Temp' in df_all.columns else df_all['Z'],
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title='Temperature' if 'Temp' in df_all.columns else 'File Index')
                    ),
                    text=df_all['File'],
                    hovertemplate='<b>File:</b> %{text}<br>' +
                                '<b>X:</b> %{x:.2f}<br>' +
                                '<b>Y:</b> %{y:.2f}<br>' +
                                '<b>Z (File Index):</b> %{z}<br>' +
                                '<b>Temp:</b> %{marker.color:.2f}<br>' +
                                '<extra></extra>',
                    name='Point Cloud'
                ))
                
                # Update layout
                fig.update_layout(
                    title="3D Stacked PCD Visualization (Files stacked along Z-axis)",
                    scene=dict(
                        xaxis_title='X Coordinate',
                        yaxis_title='Y Coordinate',
                        zaxis_title='File Index (Z-axis)',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5),
                            center=dict(x=0, y=0, z=len(all_points)//2)
                        ),
                        aspectmode='data'
                    ),
                    width=1000,
                    height=700,
                    showlegend=False,
                    margin=dict(l=0, r=0, b=0, t=50)
                )
                
                # Display the 3D plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Add file statistics
                st.markdown("### 📈 File Statistics")
                stats_cols = st.columns(min(4, len(all_points)))
                
                for i, (fname, file_obj) in enumerate(files_sorted):
                    if i < len(stats_cols):
                        with stats_cols[i]:
                            df_file = all_points[i] if i < len(all_points) else None
                            if df_file is not None:
                                st.markdown(f"**{fname}**")
                                st.metric("Points", len(df_file))
                                if 'Temp' in df_file.columns:
                                    st.metric("Avg Temp", f"{df_file['Temp'].mean():.2f}")
                                    st.metric("Temp Range", f"{df_file['Temp'].min():.2f} - {df_file['Temp'].max():.2f}")
                                st.metric("X Range", f"{df_file['X'].min():.2f} - {df_file['X'].max():.2f}")
                                st.metric("Y Range", f"{df_file['Y'].min():.2f} - {df_file['Y'].max():.2f}")
                st.markdown("### 🎛️ Interactive Controls")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    point_size = st.slider("Point Size", min_value=1, max_value=10, value=3, step=1)
                with col2:
                    opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
                with col3:
                    color_scheme = st.selectbox("Color Scheme", ["Viridis", "Plasma", "Inferno", "Magma", "Turbo"])
                
                # Update plot with new settings
                if st.button("🔄 Update Visualization", key="update_3d_plot"):
                    fig_updated = go.Figure()
                    fig_updated.add_trace(go.Scatter3d(
                        x=df_all['X'],
                        y=df_all['Y'],
                        z=df_all['Z'],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            color=df_all['Temp'] if 'Temp' in df_all.columns else df_all['Z'],
                            colorscale=color_scheme,
                            opacity=opacity,
                            colorbar=dict(title='Temperature' if 'Temp' in df_all.columns else 'File Index')
                        ),
                        text=df_all['File'],
                        hovertemplate='<b>File:</b> %{text}<br>' +
                                    '<b>X:</b> %{x:.2f}<br>' +
                                    '<b>Y:</b> %{y:.2f}<br>' +
                                    '<b>Z (File Index):</b> %{z}<br>' +
                                    '<b>Temp:</b> %{marker.color:.2f}<br>' +
                                    '<extra></extra>',
                        name='Point Cloud'
                    ))
                    
                    fig_updated.update_layout(
                        title="3D Stacked PCD Visualization (Files stacked along Z-axis)",
                        scene=dict(
                            xaxis_title='X Coordinate',
                            yaxis_title='Y Coordinate',
                            zaxis_title='File Index (Z-axis)',
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.5),
                                center=dict(x=0, y=0, z=len(all_points)//2)
                            ),
                            aspectmode='data'
                        ),
                        width=1000,
                        height=700,
                        showlegend=False,
                        margin=dict(l=0, r=0, b=0, t=50)
                    )
                    
                    st.plotly_chart(fig_updated, use_container_width=True)
                
            else:
                st.warning("No valid PCD data to plot. Please check your uploaded files.")

if (st.session_state.get('analysis_type') == 'Comparative Analysis' and len(st.session_state.uploaded_files) == 1):
    st.markdown("<div style='margin-top:2em; text-align:center;'>", unsafe_allow_html=True)
    if st.button('Upload More Files', key='upload_more_files_btn', use_container_width=True):
        st.session_state.show_upload_area = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
