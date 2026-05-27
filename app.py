import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import io
from model import CNN

st.set_page_config(
    page_title="SkyScan · Bird vs Drone",
    page_icon="📡",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #0a0a0f;
    color: #f0f0f8;
}

.block-container {
    padding-top: 3rem;
    max-width: 700px;
}

h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -2px !important;
    color: #f0f0f8 !important;
    line-height: 1 !important;
    margin-bottom: 0 !important;
}

.eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #6b6b88;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.model-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #6b6b88;
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 100px;
    padding: 5px 14px;
    margin-top: 0.8rem;
    margin-bottom: 2rem;
}

.result-bird {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -3px;
    color: #4ade80;
    line-height: 1;
    text-align: center;
    padding: 2rem 0 1rem;
}

.result-drone {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -3px;
    color: #f97316;
    line-height: 1;
    text-align: center;
    padding: 2rem 0 1rem;
}

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b6b88;
    text-align: center;
}

.score-box {
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 10px;
    padding: 12px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #6b6b88;
    display: flex;
    justify-content: space-between;
    margin-top: 1rem;
}

.divider {
    border-top: 1px solid #2a2a38;
    margin: 2rem 0;
}

/* File uploader styling */
[data-testid="stFileUploader"] {
    background: #111118;
    border: 1.5px dashed #2a2a38;
    border-radius: 12px;
    padding: 1rem;
}

[data-testid="stFileUploader"]:hover {
    border-color: #4ade80;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("model/best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


def process_image(image_bytes):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def predict(model, img_array):
    tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        score = output.item()
        label = "Drone" if round(score) == 1 else "Bird"
    return label, score


# ── Header ──────────────────────────────────────────────────────────────
st.markdown('<p class="eyebrow">Aerial Object Classifier</p>', unsafe_allow_html=True)
st.markdown("# SkyScan")
st.markdown("""
<div class="model-tag">
    <span style="width:7px;height:7px;border-radius:50%;background:#4ade80;display:inline-block;"></span>
    CNN · predict.py · Bird / Drone only
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a JPG image",
    type=["jpg", "jpeg"],
    help="Only .jpg / .jpeg files accepted — matching your model's training data format"
)

# ── Classify ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image_bytes, use_container_width=True, caption="Uploaded image")

    with col2:
        with st.spinner("Running CNN inference..."):
            try:
                model = load_model()
                img_array = process_image(image_bytes)

                if img_array is None:
                    st.error("Could not read image. Please upload a valid JPG file.")
                else:
                    label, score = predict(model, img_array)
                    icon = "🐦" if label == "Bird" else "🚁"
                    css_class = "result-bird" if label == "Bird" else "result-drone"

                    st.markdown(f'<p class="result-label">Your CNN model says</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="{css_class}">{icon} {label}</p>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="score-box">
                        <span>Sigmoid raw score (0=Bird · 1=Drone)</span>
                        <span style="color:#f0f0f8;font-weight:700;">{score:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Model error: {str(e)}")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#3a3a52;font-family:'Space Mono',monospace;font-size:12px;">
        Upload a .jpg image above to classify it as Bird or Drone
    </div>
    """, unsafe_allow_html=True)