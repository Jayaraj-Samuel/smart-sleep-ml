import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sounddevice as sd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Sleep Predictor",
    page_icon="😴",
    layout="wide"
)

# --------------------------------------------------
# Load ML Models
# --------------------------------------------------
time_model = joblib.load("sleep_time_model.pkl")
quality_model = joblib.load("sleep_quality_model.pkl")

# --------------------------------------------------
# Sidebar - Manual Inputs
# --------------------------------------------------
st.sidebar.header("🛠️ Manual Inputs")

temperature = st.sidebar.slider("🌡️ Room Temperature (°C)", 10, 40, 25)
heart_rate = st.sidebar.number_input(
    "❤️ Heart Rate (BPM)",
    min_value=40,
    max_value=200,
    value=70
)
st.sidebar.info(f"🫀 Current Heart Rate: {heart_rate} BPM")

# --------------------------------------------------
# Main Title
# --------------------------------------------------
st.title("😴 Smart Sleep Predictor")
st.caption("AI-powered sleep time & quality prediction using real environmental data")

# Layout columns
col1, col2 = st.columns(2)

# --------------------------------------------------
# Light Detection (Camera / Upload)
# --------------------------------------------------
with col1:
    st.subheader("💡 Light Detection")

    method = st.radio(
        "Choose light input method:",
        ("Use Camera (if available)", "Upload Room Image")
    )

    light_level = 50  # default

    if method == "Use Camera (if available)":
        cam_img = st.camera_input("Capture room image")
        if cam_img:
            img_array = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            light_level = int(np.mean(gray) / 255 * 100)
            st.success(f"Detected Light Level: {light_level}/100")

    if method == "Upload Room Image":
        uploaded_img = st.file_uploader(
            "Upload room image",
            type=["jpg", "png", "jpeg"]
        )
        if uploaded_img:
            image = Image.open(uploaded_img)
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            light_level = int(np.mean(gray) / 255 * 100)
            st.image(image, use_column_width=True)
            st.success(f"Detected Light Level: {light_level}/100")

# --------------------------------------------------
# Noise Detection (Microphone)
# --------------------------------------------------
with col2:
    st.subheader("🎤 Noise Detection")

    duration = 3  # seconds
    fs = 44100

    noise_level = 30  # default fallback

    if st.button("🎙️ Record Room Noise"):
        st.info("Recording... please stay quiet")

        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            device=8  # YOUR MICROPHONE DEVICE ID
        )
        sd.wait()

        audio = recording.flatten()

        rms = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0

        if np.isnan(rms) or np.isinf(rms):
            rms = 0

        noise_level = int(min(max(rms * 1000, 0), 100))

        if noise_level == 0:
            st.info("Low or no ambient noise detected.")
        else:
            st.success(f"Detected Noise Level: {noise_level}/100")

    else:
        st.info("Click the button to record room noise.")
# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.markdown("---")
st.subheader("🧠 Sleep Prediction")

if st.button("🔮 Predict Sleep"):
    features = [[temperature, light_level, noise_level, heart_rate]]

    sleep_time = time_model.predict(features)[0]
    sleep_quality = quality_model.predict(features)[0]

    # Round outputs
    sleep_time = round(float(sleep_time), 1)
    sleep_quality = round(float(sleep_quality), 0)

    colA, colB = st.columns(2)

    with colA:
        st.metric("🕒 Time to Fall Asleep (minutes)", sleep_time)

    with colB:
        st.metric("⭐ Sleep Quality Score", f"{sleep_quality}/100")