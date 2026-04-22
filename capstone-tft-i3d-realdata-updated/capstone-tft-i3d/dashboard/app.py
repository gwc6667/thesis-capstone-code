from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import numpy as np
import streamlit as st
import torch

from models.tft_model import TemporalFusionModel
from models.i3d_model import I3DLikeModel
from models.fusion_model import GatedFusionModel
from utils import load_checkpoint

CHECKPOINTS = ROOT / 'checkpoints'


def load_models():
    tft = TemporalFusionModel(input_size=5, hidden_size=64)
    i3d = I3DLikeModel(num_classes=4, feature_dim=128)
    fusion = GatedFusionModel(academic_dim=64, emotion_dim=128, hidden_dim=128, num_classes=1)

    load_checkpoint(tft, CHECKPOINTS / 'tft_checkpoint.pth', map_location='cpu')
    load_checkpoint(i3d, CHECKPOINTS / 'i3d_checkpoint.pth', map_location='cpu')
    load_checkpoint(fusion, CHECKPOINTS / 'fusion_checkpoint.pth', map_location='cpu')

    tft.eval(); i3d.eval(); fusion.eval()
    return tft, i3d, fusion


def preprocess_uploaded_image(file_bytes, num_frames=8, size=32):
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    frames = []
    for _ in range(num_frames):
        resized = cv2.resize(image, (size, size))
        rgb = resized[:, :, ::-1]
        frames.append(rgb)
    video = np.stack(frames, axis=0).transpose(3, 0, 1, 2)
    return torch.tensor(video).float().unsqueeze(0)


def academic_feature_demo(seq):
    # Match the fusion model's expected 64-d academic feature size.
    return seq.mean(dim=1).repeat(1, 13)[:, :64]


def main():
    st.title('Multimodal Learning Analytics Dashboard')
    st.write('Demo app for TFT, I3D-like emotion model, and gated fusion.')

    try:
        tft_model, i3d_model, fusion_model = load_models()
    except FileNotFoundError:
        st.error('Checkpoint files not found. Run the training scripts first.')
        return

    st.header('1. TFT Performance Prediction Demo')
    if st.button('Run TFT Demo'):
        seq = torch.randn(1, 12, 5)
        with torch.no_grad():
            logit = tft_model(seq)
            prob = torch.sigmoid(logit).item()
        st.write(f'Predicted pass probability: **{prob:.4f}**')

    st.header('2. Emotion Recognition (I3D-like)')
    uploaded = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])
    emotion_features = None
    if uploaded:
        tensor = preprocess_uploaded_image(uploaded.read())
        if tensor is not None:
            classes = ['Boredom', 'Engagement', 'Confusion', 'Frustration']
            with torch.no_grad():
                logits = i3d_model(tensor)
                pred = torch.argmax(logits, dim=1).item()
                emotion_features = i3d_model.extract_features(tensor)
            st.success(f'Predicted Emotion: **{classes[pred]}**')

    st.header('3. Gated Fusion Prediction Demo')
    if st.button('Run Fusion Demo'):
        seq = torch.randn(1, 12, 5)
        academic_features = academic_feature_demo(seq)
        if emotion_features is None:
            emotion_features = torch.randn(1, 128)
        with torch.no_grad():
            logit = fusion_model(academic_features, emotion_features)
            prob = torch.sigmoid(logit).item()
        st.write(f'Final Fusion Prediction Score: **{prob:.4f}**')


if __name__ == '__main__':
    main()
