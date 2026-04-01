import os
import sys

# make parent directory importable (so we can import models.*)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np

from models.tft_model import TemporalFusionModel
from models.i3d_model import I3DLikeModel
from models.fusion_model import AttentionFusion, FinalPredictionHead


# Load TFT model
tft_model = TemporalFusionModel(input_size=5, hidden_size=64)
tft_model.load_state_dict(torch.load("models/tft_checkpoint.pth", map_location="cpu"))
tft_model.eval()

# Load I3D model
i3d_model = I3DLikeModel(num_classes=4)
i3d_model.load_state_dict(torch.load("models/i3d_checkpoint.pth", map_location="cpu"))
i3d_model.eval()

# Load fusion model
fusion = AttentionFusion()
fusion.load_state_dict(torch.load("models/fusion_checkpoint.pth", map_location="cpu"))
fusion.eval()

prediction_head = FinalPredictionHead()
prediction_head.load_state_dict(torch.load("models/prediction_head.pth", map_location="cpu"))
prediction_head.eval()


def preprocess_video(video_bytes, num_frames=16, size=64):
    """Extract frames from uploaded video and resize."""
    file_bytes = np.asarray(bytearray(video_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    frames = []
    for _ in range(num_frames):
        resized = cv2.resize(frame, (size, size))
        rgb = resized[:, :, ::-1]
        frames.append(rgb)

    arr = np.stack(frames, axis=0)
    arr = arr.transpose(3, 0, 1, 2)
    return torch.tensor(arr).float().unsqueeze(0)


st.title("Multimodal Learning Analytics Dashboard")
st.write("TFT + I3D + Attention Fusion System")


# --- TFT Section ---
st.header("1. TFT Performance Prediction Demo")

if st.button("Run TFT Demo"):
    seq = torch.randn(1, 12, 5)
    with torch.no_grad():
        out = tft_model(seq)
        prob = torch.sigmoid(out).item()
    st.write(f"Predicted pass probability: **{prob:.4f}**")


# --- I3D Section ---
st.header("2. Emotion Recognition (I3D)")

uploaded = st.file_uploader("Upload a video frame or image", type=["jpg", "png", "jpeg"])

if uploaded:
    tensor = preprocess_video(uploaded.read())
    if tensor is not None:
        with torch.no_grad():
            logits = i3d_model(tensor)
            pred = torch.argmax(logits, dim=1).item()

        classes = ["Boredom", "Engagement", "Confusion", "Frustration"]
        st.success(f"Predicted Emotion: **{classes[pred]}**")


# --- Fusion Section ---
st.header("3. Cross-Modal Fusion Prediction Demo")

if st.button("Run Fusion Demo"):
    tft_vec = torch.randn(1, 64)
    i3d_vec = torch.randn(1, 512)

    with torch.no_grad():
        fused = fusion(tft_vec, i3d_vec)
        logit = prediction_head(fused)
        prob = torch.sigmoid(logit).item()

    st.write(f"Final Fusion Prediction Score: **{prob:.4f}**")
