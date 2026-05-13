import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# 1. Page Configuration
st.set_page_config(
    page_title="RPS | Gesture Game", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize Session State
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = True
if 'image_source' not in st.session_state:
    st.session_state.image_source = None

# 2. Studio Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700&display=swap');
    section[data-testid="stSidebar"] { display: none !important; }
    :root {
        --bg: #050505;
        --card: #0f0f0f;
        --accent: #ffffff;
    }
    .stApp {
        background-color: var(--bg);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    .header-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 2rem 0; border-bottom: 1px solid #1a1a1a;
        margin-bottom: 1rem; /* Reduced bottom margin to fit disclaimer nicely */
    }
    .result-container {
        background: var(--card);
        border: 1px solid #1a1a1a;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
    }
    .prediction-label { 
        font-size: 5rem; font-weight: 700; color: #fff; 
        margin: 0; letter-spacing: -3px; 
    }
    .stButton > button {
        background-color: transparent;
        color: white; border: 1px solid #333;
        border-radius: 8px; width: 100%; margin-top: 10px;
    }
    .stRadio > div { flex-direction: row !important; justify-content: center !important; gap: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# 3. Neural Runtime
@st.cache_resource
def load_engine():
    # Load the TFLite model specifically for 160x160 RGB tensors
    interpreter = tf.lite.Interpreter(model_path="rps_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_engine()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
CLASS_NAMES = ['Paper', 'Rock', 'Scissors']

def run_analysis(img):
    """Processes input with X-axis mirroring and 160x160 scaling"""
    frame = np.array(img)
    # Mirroring only for live camera feel; usually ignored for static uploads
    mirrored = cv2.flip(frame, 1) 
    
    pil_img = Image.fromarray(mirrored)
    prep = pil_img.resize((160, 160)) # Prep for MobileNetV2
    
    input_data = np.expand_dims(np.array(prep, dtype=np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = np.argmax(output)
    return CLASS_NAMES[idx], output[idx] * 100, pil_img

# 4. Interface Logic
st.markdown('''
    <div class="header-bar">
        <div style="font-weight:700; letter-spacing:1px;">ROCK PAPER SCISSORS | GESTURE GAME</div>
        <a href="https://github.com/themehmi/Rock-Paper-Scissors-Classifier" 
           target="_blank" style="text-decoration:none; color:#444; font-size:12px;">SOURCE_CODE</a>
    </div>
''', unsafe_allow_html=True)

# Prominent Disclaimer added here
st.warning("⚠️ **Disclaimer:** This app only functions properly when the background is clean (black or white).", icon="⚠️")
st.write("---")

mode = st.radio("", ("UPLOAD FILE", "USE CAMERA"), horizontal=True)
st.write("---")

col_left, col_right = st.columns([1, 1], gap="large")

# Reset image if mode switches
if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode
if st.session_state.last_mode != mode:
    st.session_state.image_source = None
    st.session_state.webcam_active = True
    st.session_state.last_mode = mode

with col_left:
    if mode == "UPLOAD FILE":
        # Always show file uploader in this mode
        uploaded_file = st.file_uploader("Select a gesture image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            st.session_state.image_source = Image.open(uploaded_file).convert('RGB')
    
    elif mode == "USE CAMERA":
        if st.session_state.webcam_active:
            cam_data = st.camera_input("CAPTURE GESTURE")
            if cam_data:
                st.session_state.image_source = Image.open(cam_data).convert('RGB')
                st.session_state.webcam_active = False
                st.rerun()
        else:
            if st.button("RESET & TAKE NEW PICTURE"):
                st.session_state.webcam_active = True
                st.session_state.image_source = None
                st.rerun()

with col_right:
    # Run prediction immediately when image_source is populated
    if st.session_state.image_source:
        label, conf, final_view = run_analysis(st.session_state.image_source)
        st.image(final_view, use_container_width=True)
        
        st.markdown(f"""
            <div class="result-container">
                <p style="font-size:10px; font-weight:700; color:#333; letter-spacing:2px; margin-bottom:10px;">ANALYSIS_COMPLETE</p>
                <h1 class="prediction-label">{label.upper()}</h1>
                <p style="color:#fff; font-size:14px; opacity:0.6;">Confidence: {conf:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="height:400px; display:flex; align-items:center; justify-content:center; border:1px solid #111; border-radius:16px;">
                <p style="color:#222; font-size:12px; letter-spacing:2px;">AWAITING_INPUT_SIGNAL</p>
            </div>
        """, unsafe_allow_html=True)
