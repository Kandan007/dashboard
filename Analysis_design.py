import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

st.set_page_config(layout="wide", page_title="ROTRIX Dashboard Mockup")

# --- Sidebar ---
st.sidebar.image("Rotrix-Logo.png", width=120)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

menu_items = [
    ("Dashboard", "grid"),
    ("Power train", "bolt"),
    ("New Device", "plus-circle"),
    ("Analysis", "bar-chart"),
    ("Test History", "clock"),
    ("Report", "file-text")
]
for item, icon in menu_items:
    if item == "Analysis":
        st.sidebar.markdown(f"<div style='background:#e6eaf7; border-radius:8px; padding:8px; margin-bottom:4px;'><b>{item}</b></div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"<div style='padding:8px; margin-bottom:4px;'>{item}</div>", unsafe_allow_html=True)

# --- Header ---
col_logo, col_title, col_user = st.columns([1, 6, 1])
with col_logo:
    st.write("")
with col_title:
    st.markdown("<h4 style='margin-bottom:0;'>Device Info</h4>", unsafe_allow_html=True)
with col_user:
    st.markdown("<div style='text-align:right;'>Kandan <span style='font-size:20px;'>👤</span></div>", unsafe_allow_html=True)
    st.selectbox("", ["Power Train"], label_visibility="collapsed")

st.markdown("<hr style='margin-top:0;margin-bottom:0;'>", unsafe_allow_html=True)

# --- Main Content ---
left, right = st.columns([1.1, 2])

with left:
    # Gauges (as placeholder circles)
    gauge_cols = st.columns(4)
    for i, label in enumerate(["Voltage ()", "Current ()", "Thrust ()", "Temperature()"]):
        with gauge_cols[i]:
            fig, ax = plt.subplots(figsize=(2,2))
            circle = Circle((0.5, 0.5), 0.45, color='#3d3d9b', fill=False, linewidth=8)
            ax.add_artist(circle)
            ax.text(0.5, 0.5, "0.0", fontsize=18, ha='center', va='center')
            ax.axis('off')
            st.pyplot(fig)
            st.markdown(f"<div style='text-align:center;font-size:14px;'>{label}</div>", unsafe_allow_html=True)

    st.markdown(f"**Controller**")
    st.slider("", 1000, 2000, 1200, key="controller_slider")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Throttle %", "", key="throttle")
    with col2:
        st.text_input("PWM Value", "", key="pwm")

    st.markdown(f"**Performance Monitor**")
    df = pd.DataFrame({
        "S.No": [],
        "Throttle %": [],
        "Timer (MM:SS)": [],
        "Status": []
    })
    st.dataframe(df, use_container_width=True, height=60)

    st.markdown("""
    <div style='margin-top:20px;'>
    <table style='width:100%; font-size:14px;'>
      <tr><td>Electrical Power ()</td><td style='text-align:right;'>0</td></tr>
      <tr><td>Mechanical Power ()</td><td style='text-align:right;'>0</td></tr>
      <tr><td>Motor Efficiency ()</td><td style='text-align:right;'>0</td></tr>
      <tr><td>Propeller Efficiency ()</td><td style='text-align:right;'>0</td></tr>
      <tr><td>Overall System Efficiency ()</td><td style='text-align:right;'>0</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

with right:
    # Test Modes Row (heading and icons in one row, no labels under icons)
    test_modes = [
        ("Calibration Test", "🛠️"),
        ("Free Control Test", "📊"),
        ("Standard Test", "🧪"),
        ("Custom Step Test", "🪜"),
        ("Performance Test (Ramp Test)", "📈"),
        ("Shared Test", "🗂️")
    ]
    # --- Free Control Test popup state ---
    if 'show_free_control_popup' not in st.session_state:
        st.session_state['show_free_control_popup'] = False
    if 'show_standard_popup' not in st.session_state:
        st.session_state['show_standard_popup'] = False
    if 'show_custom_step_popup' not in st.session_state:
        st.session_state['show_custom_step_popup'] = False
    if 'show_performance_popup' not in st.session_state:
        st.session_state['show_performance_popup'] = False
    if 'show_shared_test_popup' not in st.session_state:
        st.session_state['show_shared_test_popup'] = False

    # Popup CSS
    popup_css = '''
    <style>
    .cal-popup {
        margin: 8px auto 0 auto;
        background: #fff;
        border: none;
        border-radius: 4px;
        padding: -20px 10px 10px 10px;
        min-width: 220px;
        max-width: 350px;
        font-size: 10px;
        text-align: center;
        box-shadow: none;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .cal-btn {
        background: #3dc3cb;
        color: #fff;
        border: none;
        border-radius: 3px;
        padding: 7px 0;
        width: 70%;
        font-size: 15px;
        margin: 14px auto 0 auto;
        display: block;
        cursor: pointer;
        text-align: center;
    }
    .cal-btn:active { opacity: 0.9; }
    </style>
    '''
    # Inject custom CSS for icon buttons (ensure this is before rendering the buttons)
    cal_btn_css = """
    <style>
    div[data-testid='stButton'] button.cal-ico-btn {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0 !important;
        margin: 0 auto !important;
        font-size: 28px !important;
        cursor: pointer;
        color: inherit !important;
        height: auto !important;
        width: auto !important;
        line-height: 1 !important;
    }
    </style>
    """
    st.markdown(cal_btn_css, unsafe_allow_html=True)
    # Render icons in a row, with the first five icons as styled buttons
    icon_cols = st.columns([0.7] + [0.45]*len(test_modes) + [2.45])
    icon_cols[0].markdown("<span style='font-weight:bold;font-size:13px;'>Test modes:</span>", unsafe_allow_html=True)
    if icon_cols[1].button("🛠️", key="calibration_icon_btn", help="Calibration Test", use_container_width=True, type="secondary"):
        st.session_state['show_cal_popup'] = not st.session_state.get('show_cal_popup', False)
        st.session_state['calibrated_done'] = False
        st.session_state['show_free_control_popup'] = False
        st.session_state['show_standard_popup'] = False
        st.session_state['show_custom_step_popup'] = False
        st.session_state['show_performance_popup'] = False
    icon_cols[1].markdown("""
    <script>
    let btn = window.parent.document.querySelectorAll("button[kind='secondary']");
    if(btn && btn.length > 0) { btn[0].classList.add('cal-ico-btn'); }
    </script>
    """, unsafe_allow_html=True)
    # Free Control Test icon (📊)
    if icon_cols[2].button("📊", key="free_control_icon_btn", help="Free Control Test", use_container_width=True, type="secondary"):
        st.session_state['show_free_control_popup'] = not st.session_state.get('show_free_control_popup', False)
        st.session_state['show_cal_popup'] = False
        st.session_state['calibrated_done'] = False
        st.session_state['show_standard_popup'] = False
        st.session_state['show_custom_step_popup'] = False
        st.session_state['show_performance_popup'] = False
    icon_cols[2].markdown("""
    <script>
    let btns = window.parent.document.querySelectorAll("button[kind='secondary']");
    if(btns && btns.length > 1) { btns[1].classList.add('cal-ico-btn'); }
    </script>
    """, unsafe_allow_html=True)
    # Standard Test icon (🧪)
    if icon_cols[3].button("🧪", key="standard_icon_btn", help="Standard Test", use_container_width=True, type="secondary"):
        st.session_state['show_standard_popup'] = not st.session_state.get('show_standard_popup', False)
        st.session_state['show_cal_popup'] = False
        st.session_state['calibrated_done'] = False
        st.session_state['show_free_control_popup'] = False
        st.session_state['show_custom_step_popup'] = False
        st.session_state['show_performance_popup'] = False
    icon_cols[3].markdown("""
    <script>
    let btns = window.parent.document.querySelectorAll("button[kind='secondary']");
    if(btns && btns.length > 2) { btns[2].classList.add('cal-ico-btn'); }
    </script>
    """, unsafe_allow_html=True)
    # Custom Step Test icon (🪜)
    if icon_cols[4].button("🪜", key="custom_step_icon_btn", help="Custom Step Test", use_container_width=True, type="secondary"):
        st.session_state['show_custom_step_popup'] = not st.session_state.get('show_custom_step_popup', False)
        st.session_state['show_cal_popup'] = False
        st.session_state['calibrated_done'] = False
        st.session_state['show_free_control_popup'] = False
        st.session_state['show_standard_popup'] = False
        st.session_state['show_performance_popup'] = False
    icon_cols[4].markdown("""
    <script>
    let btns = window.parent.document.querySelectorAll("button[kind='secondary']");
    if(btns && btns.length > 3) { btns[3].classList.add('cal-ico-btn'); }
    </script>
    """, unsafe_allow_html=True)
    # Performance Test icon (📈)
    if icon_cols[5].button("📈", key="performance_icon_btn", help="Performance Test (Ramp Test)", use_container_width=True, type="secondary"):
        st.session_state['show_performance_popup'] = not st.session_state.get('show_performance_popup', False)
        st.session_state['show_cal_popup'] = False
        st.session_state['calibrated_done'] = False
        st.session_state['show_free_control_popup'] = False
        st.session_state['show_standard_popup'] = False
        st.session_state['show_custom_step_popup'] = False
    icon_cols[5].markdown("""
    <script>
    let btns = window.parent.document.querySelectorAll("button[kind='secondary']");
    if(btns && btns.length > 4) { btns[4].classList.add('cal-ico-btn'); }
    </script>
    """, unsafe_allow_html=True)
    # Shared Test icon (🗂️)
    if icon_cols[6].button("🗂️", key="shared_test_icon_btn", help="Shared Test", use_container_width=True, type="secondary"):
        st.session_state['show_shared_test_popup'] = not st.session_state.get('show_shared_test_popup', False)
        st.session_state['show_cal_popup'] = False
        st.session_state['calibrated_done'] = False
        st.session_state['show_free_control_popup'] = False
        st.session_state['show_standard_popup'] = False
        st.session_state['show_custom_step_popup'] = False
        st.session_state['show_performance_popup'] = False
    icon_cols[6].markdown("""
    <script>
    let btns = window.parent.document.querySelectorAll("button[kind='secondary']");
    if(btns && btns.length > 5) { btns[5].classList.add('cal-ico-btn'); }
    </script>
    """, unsafe_allow_html=True)

    # --- Initialize popup session state keys ---
    popup_keys = [
        'show_cal_popup', 'show_free_control_popup', 'show_standard_popup',
        'show_custom_step_popup', 'show_performance_popup', 'show_shared_test_popup'
    ]
    for key in popup_keys:
        if key not in st.session_state:
            st.session_state[key] = False

    # --- Calibration popup ---
    if st.session_state.get('show_cal_popup', False):
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown("<div class='cal-popup' style='text-align:center; max-width:350px;'>", unsafe_allow_html=True)
            st.markdown("""
                <div style='margin-bottom:10px;'><b>Remove Propeller for this procedure!</b></div>
                <div style='margin-bottom:12px;'>
                    After pushing the button,<br>
                    - Disconnect battery<br>
                    - Wait 10 secs, connect battery<br>
                    - ESCs should beep as they are calibrated<br>
                    - Disconnect device
                </div>
            """, unsafe_allow_html=True)
            st.markdown('''
            <style>
            .cal-btn {
                background: #3dc3cb;
                color: #fff;
                border: none;
                border-radius: 3px;
                padding: 7px 0;
                width: 70%;
                font-size: 15px;
                margin: 14px auto 0 0;
                display: block;
                cursor: pointer;
                text-align: center;
            }
            .cal-btn:active { opacity: 0.9; }
            </style>
            ''', unsafe_allow_html=True)
            if 'calibrated_done' not in st.session_state:
                st.session_state['calibrated_done'] = False
            if not st.session_state['calibrated_done']:
                if st.button("Calibrate ESC", key="cal_btn_popup"):
                    st.session_state['calibrated_done'] = True
            else:
                st.markdown("<div class='cal-btn' style='pointer-events:none;'>ESC calibrated successfully!!</div>", unsafe_allow_html=True)

    # --- Free Control Test popup ---
    if st.session_state.get('show_free_control_popup', False):
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown(popup_css, unsafe_allow_html=True)
            st.markdown("<div class='cal-popup' style='text-align:center;'>", unsafe_allow_html=True)
            st.markdown('''
                <style>
                .free-btn {
                    background: #3dc3cb;
                    color: #fff;
                    border: none;
                    border-radius: 3px;
                    padding: 7px 0;
                    width: 70%;
                    font-size: 15px;
                    margin: 14px auto 0 auto;
                    display: block;
                    cursor: pointer;
                    text-align: center;
                }
                .free-btn:active { opacity: 0.9; }
                </style>
            ''', unsafe_allow_html=True)
            # Row 1: Initial Throttle %
            col1, col2 = st.columns([.8, 1], gap="small")
            with col1:
                st.markdown("<div style='text-align:center;margin-top:8px;'>Initial Throttle % :</div>", unsafe_allow_html=True)
            with col2:
                st.text_input("", key="free_throttle", label_visibility="collapsed")
            # Row 2: Initial PWM value
            col3, col4 = st.columns([.8, 1], gap="small")
            with col3:
                st.markdown("<div style='text-align:center;margin-top:8px;'>Initial PWM value:</div>", unsafe_allow_html=True)
            with col4:
                st.text_input("", key="free_pwm", label_visibility="collapsed")
            # Row 3: Time step
            col5, col6 = st.columns([.8, 1], gap="small")
            with col5:
                st.markdown("<div style='text-align:center;margin-top:8px;'>Time step (in sec) :</div>", unsafe_allow_html=True)
            with col6:
                st.text_input("", key="free_time_step", label_visibility="collapsed")
            # Centered Start button
            st.markdown("<div style='text-align:center;margin-top:10px;'><button class='free-btn'>Start</button></div>", unsafe_allow_html=True)

    # --- Standard Test popup ---
    if st.session_state.get('show_standard_popup', False):
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.markdown(popup_css, unsafe_allow_html=True)
            st.markdown("<div class='cal-popup' style='text-align:center;'>", unsafe_allow_html=True)
            st.markdown('''
                <style>
                .num-btn {
                    background: white;
                    color: black;
                    border: 1px solid #b2e0e6;
                    border-radius: 3px;
                    padding: 8px 12px;
                    margin: 2px;
                    font-size: 14px;
                    cursor: pointer;
                    display: inline-block;
                    min-width: 40px;
                    text-align: center;
                }
                .num-btn:hover { background: #f0f8ff; }
                .num-btn.selected { background: #3dc3cb; color: white; border-color: #3dc3cb; }
                .standard-btn {
                    background: #3dc3cb;
                    color: #fff;
                    border: none;
                    border-radius: 3px;
                    padding: 7px 0;
                    width: 70%;
                    font-size: 15px;
                    margin: 14px auto 0 auto;
                    display: block;
                    cursor: pointer;
                    text-align: center;
                }
                .standard-btn:active { opacity: 0.9; }
                </style>
            ''', unsafe_allow_html=True)
            # Initialize selected numbers
            if 'selected_numbers' not in st.session_state:
                st.session_state['selected_numbers'] = []
            # Number grid
            numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            # Row 1: 10, 20, 30, 40
            row1_cols = st.columns(4)
            for i, num in enumerate([10, 20, 30, 40]):
                with row1_cols[i]:
                    if st.button(str(num), key=f"num_{num}_row1", use_container_width=True):
                        if num in st.session_state['selected_numbers']:
                            st.session_state['selected_numbers'].remove(num)
                        else:
                            st.session_state['selected_numbers'].append(num)
            # Row 2: 50, 60, 70, 80
            row2_cols = st.columns(4)
            for i, num in enumerate([50, 60, 70, 80]):
                with row2_cols[i]:
                    if st.button(str(num), key=f"num_{num}_row2", use_container_width=True):
                        if num in st.session_state['selected_numbers']:
                            st.session_state['selected_numbers'].remove(num)
                        else:
                            st.session_state['selected_numbers'].append(num)
            # Row 3: 90, 100 (centered)
            st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
            col90, col100 = st.columns(2)
            with col90:
                if st.button("90", key="num_90_row3", use_container_width=True):
                    if 90 in st.session_state['selected_numbers']:
                        st.session_state['selected_numbers'].remove(90)
                    else:
                        st.session_state['selected_numbers'].append(90)
            with col100:
                if st.button("100", key="num_100_row3", use_container_width=True):
                    if 100 in st.session_state['selected_numbers']:
                        st.session_state['selected_numbers'].remove(100)
                    else:
                        st.session_state['selected_numbers'].append(100)
            # Start button
            st.markdown("<div style='text-align:center;margin-top:15px;'><button class='standard-btn'>Start</button></div>", unsafe_allow_html=True)

    # --- Custom Step Test popup ---
    if st.session_state.get('show_custom_step_popup', False):
        st.markdown(popup_css, unsafe_allow_html=True)
        st.markdown("<div class='cal-popup' style='text-align:center;min-width:400px;max-width:500px;'>", unsafe_allow_html=True)
        st.markdown('''
            <style>
            .custom-btn {
                background: #3dc3cb;
                color: #fff;
                border: none;
                border-radius: 3px;
                padding: 7px 0;
                width: 70%;
                font-size: 15px;
                margin: 10px auto 0 auto;
                display: block;
                cursor: pointer;
                text-align: center;
            }
            .custom-btn:active { opacity: 0.9; }
            .step-btn {
                background: #f0f0f0;
                color: #333;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 4px 8px;
                margin: 2px;
                font-size: 12px;
                cursor: pointer;
                display: inline-block;
                min-width: 30px;
                text-align: center;
            }
            .step-btn:hover { background: #e0e0e0; }
            </style>
        ''', unsafe_allow_html=True)
        
        # Initialize custom step data
        if 'custom_steps' not in st.session_state:
            st.session_state['custom_steps'] = [{"step": 1, "time": "", "throttle": "", "pwm": ""}, {"step": 2, "time": "", "throttle": "", "pwm": ""}]
        if 'repeat_times' not in st.session_state:
            st.session_state['repeat_times'] = 5
        
        # Header: Throttle sequence with Repeat Sequence checkbox and times input in one row
        header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1, 1, 0.5])
        with header_col1:
            st.markdown("<div style='text-align:left;font-weight:bold;font-size:16px;margin-bottom:5px;'>Throttle sequence</div>", unsafe_allow_html=True)
        with header_col2:
            st.checkbox("Repeat Sequence", key="repeat_sequence", value=True)
        with header_col3:
            st.number_input("", min_value=1, value=st.session_state['repeat_times'], key="repeat_times_input", label_visibility="collapsed")
        with header_col4:
            st.markdown("<div style='font-size:12px;margin-top:8px;'>times</div>", unsafe_allow_html=True)
        
        # Table with +/- buttons
        st.markdown("<div style='margin:5px 0;max-height:150px;overflow-y:auto;'>", unsafe_allow_html=True)
        
        # Table header
        table_header_cols = st.columns([0.5, 1, 1, 1])
        with table_header_cols[0]:
            st.markdown("<div style='font-size:12px;font-weight:bold;'>Step</div>", unsafe_allow_html=True)
        with table_header_cols[1]:
            st.markdown("<div style='font-size:12px;font-weight:bold;'>Time (s)</div>", unsafe_allow_html=True)
        with table_header_cols[2]:
            st.markdown("<div style='font-size:12px;font-weight:bold;'>Throttle %</div>", unsafe_allow_html=True)
        with table_header_cols[3]:
            st.markdown("<div style='font-size:12px;font-weight:bold;'>PWM value</div>", unsafe_allow_html=True)
        
        # Table rows
        for i, step_data in enumerate(st.session_state['custom_steps']):
            row_cols = st.columns([0.5, 1, 1, 1])
            with row_cols[0]:
                st.markdown(f"<div style='font-size:12px;'>{i + 1}</div>", unsafe_allow_html=True)
            with row_cols[1]:
                st.text_input("", value=step_data['time'], key=f"time_{i}", label_visibility="collapsed")
            with row_cols[2]:
                st.text_input("", value=step_data['throttle'], key=f"throttle_{i}", label_visibility="collapsed")
            with row_cols[3]:
                st.text_input("", value=step_data['pwm'], key=f"pwm_{i}", label_visibility="collapsed")
        
        # Single row with add/remove buttons
        button_cols = st.columns([0.5, 1, 1, 1])
        with button_cols[0]:
            st.markdown("<div style='font-size:12px;'></div>", unsafe_allow_html=True)
        with button_cols[1]:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("−", key="remove_last", use_container_width=True):
                    if len(st.session_state['custom_steps']) > 1:
                        st.session_state['custom_steps'].pop()
                        st.rerun()
            with col2:
                if st.button("＋", key="add_new", use_container_width=True):
                    new_step = {"step": len(st.session_state['custom_steps']) + 1, "time": "", "throttle": "", "pwm": ""}
                    st.session_state['custom_steps'].append(new_step)
                    st.rerun()
        with button_cols[2]:
            st.markdown("<div style='font-size:12px;'></div>", unsafe_allow_html=True)
        with button_cols[3]:
            st.markdown("<div style='font-size:12px;'></div>", unsafe_allow_html=True)
        
        # PWM Value Graph (actual line plot)
        st.markdown("<div style='margin:10px 0;'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 1.5))
        time_points = [1, 2, 3, 4, 5]
        pwm_values = [20, 25, 35, 40, 40]  # Line starting at 20, increasing to 40, then flattening
        ax.plot(time_points, pwm_values, color='#3dc3cb', linewidth=2)
        ax.set_xlabel('Time (secs)', fontsize=8)
        ax.set_ylabel('PWM value', fontsize=8)
        ax.set_xlim(1, 5)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("<div style='text-align:center;margin-top:10px;'><button class='custom-btn'>Start</button></div>", unsafe_allow_html=True)

    # --- Performance Test popup ---
    if st.session_state.get('show_performance_popup', False):
        st.markdown(popup_css, unsafe_allow_html=True)
        st.markdown("<div class='cal-popup' style='text-align:center;min-width:350px;max-width:420px;'>", unsafe_allow_html=True)
        st.markdown('''
            <style>
            .perf-btn {
                background: #3dc3cb;
                color: #fff;
                border: none;
                border-radius: 3px;
                padding: 7px 0;
                width: 70%;
                font-size: 15px;
                margin: 10px auto 0 auto;
                display: block;
                cursor: pointer;
                text-align: center;
            }
            .perf-btn:active { opacity: 0.9; }
            </style>
        ''', unsafe_allow_html=True)
        # Header row: Throttle sequence with Repeat Sequence checkbox and times input in one row
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns([2, 1, 1, 0.5])
        with perf_col1:
            st.markdown("<div style='text-align:left;font-weight:bold;font-size:16px;margin-bottom:5px;'>Throttle sequence</div>", unsafe_allow_html=True)
        with perf_col2:
            st.checkbox("Repeat Sequence", key="perf_repeat_sequence", value=True)
        with perf_col3:
            st.number_input("", min_value=1, value=5, key="perf_repeat_times_input", label_visibility="collapsed")
        with perf_col4:
            st.markdown("<div style='font-size:12px;margin-top:8px;'>times</div>", unsafe_allow_html=True)
        # Input and radio row
        input_col1, input_col2 = st.columns([1.2, 1])
        with input_col1:
            row1 = st.columns([1, 1.2])
            with row1[0]:
                st.markdown("<div style='text-align:right;margin-top:2px;'>Min. PWM value :</div>", unsafe_allow_html=True)
            with row1[1]:
                min_pwm = st.text_input("", key="perf_min_pwm", label_visibility="collapsed")
            row2 = st.columns([1, 1.2])
            with row2[0]:
                st.markdown("<div style='text-align:right;margin-top:2px;'>Max. PWM value:</div>", unsafe_allow_html=True)
            with row2[1]:
                max_pwm = st.text_input("", key="perf_max_pwm", label_visibility="collapsed")
            row3 = st.columns([1, 1.2])
            with row3[0]:
                st.markdown("<div style='text-align:right;margin-top:2px;'>Total time interval:</div>", unsafe_allow_html=True)
            with row3[1]:
                total_time = st.text_input("", key="perf_total_time", label_visibility="collapsed")
            row4 = st.columns([1, 1.2])
            with row4[0]:
                st.markdown("<div style='text-align:right;margin-top:2px;'>No. of steps:</div>", unsafe_allow_html=True)
            with row4[1]:
                num_steps = st.text_input("", key="perf_num_steps", label_visibility="collapsed")
        with input_col2:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            ramp_type = st.radio("", ["Ramp up", "Ramp down", "Bi-directional ramp"], key="perf_ramp_type")
        # Graph
        st.markdown("<div style='margin:10px 0;'>", unsafe_allow_html=True) 
        fig, ax = plt.subplots(figsize=(6, 1.5))
        time_points = [1, 2, 3, 4, 5]
        pwm_values = [20, 25, 35, 40, 40]
        ax.plot(time_points, pwm_values, color='#3dc3cb', linewidth=2)
        ax.set_xlabel('Time (secs)', fontsize=8)
        ax.set_ylabel('PWM value', fontsize=8)
        ax.set_xlim(1, 5)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("<div style='text-align:center;margin-top:10px;'><button class='perf-btn'>Start</button></div>", unsafe_allow_html=True)


    # --- Shared Test popup ---
    if st.session_state.get('show_shared_test_popup', False):
        st.markdown(popup_css, unsafe_allow_html=True)
        st.markdown("<div class='cal-popup' style='text-align:center;min-width:400px;max-width:500px;'>", unsafe_allow_html=True)
        st.markdown('''
            <style>
            .shared-btn {
                background: #3dc3cb;
                color: #fff;
                border: none;
                border-radius: 3px;
                padding: 7px 0;
                width: 70%;
                font-size: 15px;
                margin: 14px auto 0 auto;
                display: block;
                cursor: pointer;
                text-align: center;
            }
            .shared-btn:active { opacity: 0.9; }
            .file-section {
                margin: 10px 0;
                padding: 8px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: #f9f9f9;
            }
            .file-label {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 8px;
                text-align: left;
            }
            </style>
        ''', unsafe_allow_html=True)
        # Target File Section
        st.markdown("<div class='file-label'>Target File:</div>", unsafe_allow_html=True)
        target_source = st.selectbox("Source:", ["Select source...", "My Files", "Shared Files", "System (Local Device)", "External Cloud Sources"], key="target_source")
        if target_source != "Select source...":
            target_files = ['file1.csv', 'file2.csv', 'file3.csv']
            target_file = st.selectbox("Select file:", target_files, key="target_file")
        # Benchmark File Section
        st.markdown("<div class='file-label'>Benchmark File:</div>", unsafe_allow_html=True)
        benchmark_source = st.selectbox("Source:", ["Select source...", "My Files", "Shared Files", "System (Local Device)", "External Cloud Sources"], key="benchmark_source")
        if benchmark_source != "Select source...":
            benchmark_files = ['file1.csv', 'file2.csv', 'file3.csv']
            benchmark_file = st.selectbox("Select file:", benchmark_files, key="benchmark_file")
        st.markdown("<div style='text-align:center;margin-top:15px;'><button class='shared-btn'>Start</button></div>", unsafe_allow_html=True)

    plot_labels = [
        ("Voltage", "Current"),
        ("Thrust", "Torque"),
        ("Motor Speed", "")
    ]
    for row in plot_labels:
        cols = st.columns(2)
        for i, label in enumerate(row):
            with cols[i]:
                if label:
                    st.markdown(f"**{label}**")
                    fig, ax = plt.subplots()
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel("TIME (secs)")
                    ax.set_ylabel(f"{label} ()")
                    st.pyplot(fig)
    # Legend
    st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
    legend_items = [
        ("Mechanical Power", True),
        ("Motor Efficiency", True),
        ("System Efficiency", True),
        ("Electrical Power", False),
        ("Propeller Efficiency", False),
        ("Max Temperature", False)
    ]
    legend_html = ""
    for name, checked in legend_items:
        box = "&#x2611;" if checked else "&#x2610;"
        legend_html += f"<span style='margin-right:20px;'>{box} {name}</span>"
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True) 
