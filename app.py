import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib

# ---------------- CLOUD-SAFE MICROPHONE IMPORT -----------------
try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except:
    MIC_AVAILABLE = False

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Sleep AI",
    page_icon="😴",
    layout="wide"
)

# --------------------------------------------------
# Global UI Styling (Dribbble-style dashboard)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0E1117, #151A2D);
    color: white;
}
.block-container {
    padding-top: 1.5rem;
}
.card {
    background: #1C203B;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}
.card-title {
    font-size: 14px;
    color: #A5B4FC;
}
.card-value {
    font-size: 30px;
    font-weight: bold;
    margin-top: 5px;
}
.sidebar .sidebar-content {
    background: #0E1117;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load ML Models
# --------------------------------------------------
time_model = joblib.load("sleep_time_model.pkl")
quality_model = joblib.load("sleep_quality_model.pkl")

# --------------------------------------------------
# Sidebar - Manual Inputs
# --------------------------------------------------
with st.sidebar:
    st.title("😴 Smart Sleep AI")
    st.caption("Environment-aware sleep prediction")

    temperature = st.slider("🌡️ Room Temperature (°C)", 10, 40, 25)
    heart_rate = st.number_input(
        "❤️ Heart Rate (BPM)",
        min_value=40,
        max_value=200,
        value=70
    )

    st.info(f"🫀 Current Heart Rate: {heart_rate} BPM")

# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.markdown("## 🌙 Sleep Environment Dashboard")
st.caption("Real-time sensor inputs & AI predictions")

# --------------------------------------------------
# Sensor Section
# --------------------------------------------------
col1, col2 = st.columns(2)

# ---------------- LIGHT SENSOR ---------------------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>💡 Light Detection</div>", unsafe_allow_html=True)

    # Cloud-safe: camera only local
    if st.runtime.exists():
        method_options = ("Upload Room Image",)
    else:
        method_options = ("Upload Room Image", "Use Camera")

    method = st.radio(
        "Input Method",
        method_options,
        horizontal=True
    )

    light_level = 50

    if method == "Use Camera":
        cam_img = st.camera_input("Capture room image")
        if cam_img:
            img_array = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            light_level = int(np.mean(gray) / 255 * 100)

    if method == "Upload Room Image":
        uploaded_img = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        if uploaded_img:
            image = Image.open(uploaded_img)
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            light_level = int(np.mean(gray) / 255 * 100)
            st.image(image, use_column_width=True)

    st.markdown(f"<div class='card-value'>{light_level}/100</div>", unsafe_allow_html=True)
    st.progress(light_level / 100)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- NOISE SENSOR ---------------------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🎤 Noise Level</div>", unsafe_allow_html=True)

    duration = 3
    fs = 44100
    noise_level = 30

    if MIC_AVAILABLE:
        if st.button("🎙️ Record Noise"):
            recording = sd.rec(
                int(duration * fs),
                samplerate=fs,
                channels=1
            )
            sd.wait()
            audio = recording.flatten()
            rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0
            if np.isnan(rms) or np.isinf(rms):
                rms = 0
            noise_level = int(min(max(rms * 1000, 0), 100))
    else:
        st.warning("🎤 Live microphone not available on cloud")
        noise_level = st.slider("Set Noise Level", 0, 100, 30)

    st.markdown(f"<div class='card-value'>{noise_level}/100</div>", unsafe_allow_html=True)
    st.progress(noise_level / 100)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.markdown("## 🧠 AI Sleep Prediction")

if st.button("🔮 Predict Sleep"):
    features = [[temperature, light_level, noise_level, heart_rate]]

    sleep_time = time_model.predict(features)[0]
    sleep_quality = quality_model.predict(features)[0]

    sleep_time = round(float(sleep_time), 1)
    sleep_quality = round(float(sleep_quality), 0)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🕒 Time to Fall Asleep</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-value'>{sleep_time} min</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>⭐ Sleep Quality</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-value'>{sleep_quality}/100</div>", unsafe_allow_html=True)
        st.progress(sleep_quality / 100)
        st.markdown("</div>", unsafe_allow_html=True)
