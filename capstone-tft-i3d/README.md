# Capstone Project: A Unified AI Framework for Academic Performance Prediction and Emotion Analysis

## Project Overview
This project presents a unified artificial intelligence framework for academic performance prediction and emotion analysis. The system combines an academic prediction branch, an emotion/video analysis branch, and a multimodal fusion branch to support early warning and decision support in educational settings.

The project currently includes:
- **TFT model** for academic performance prediction
- **I3D model** for emotion/video-based representation learning
- **Fusion model** for integrating academic and emotion features
- **Dashboard module** for visualization and demonstration

## Project Objectives
The main goals of this project are:
1. Predict academic outcomes from student-related learning data
2. Extract emotion-related information from video or visual inputs
3. Explore whether multimodal fusion improves prediction performance
4. Provide an interpretable and extensible framework for educational analytics

## Folder Structure
```text
capstone-tft-i3d/
│
├── dashboard/          # Dashboard or demo application
├── data/               # Raw and processed datasets
├── models/             # Model definitions and saved checkpoints
├── training/           # Training scripts for TFT, I3D, and Fusion
├── utils/              # Utility functions
└── __init__.py
```

## Main Models
### 1. TFT Model
The Temporal Fusion Transformer (TFT) is used for academic performance prediction. It models temporal learning patterns from sequential academic data and achieved the best performance among the current models.

### 2. I3D Model
The I3D model is used for emotion or video-based representation learning. It extracts visual-temporal features from video inputs.

### 3. Fusion Model
The fusion model combines academic features and emotion/video features. It is designed to test whether multimodal integration can improve predictive performance.

## Current Experimental Results
Based on the current experiments:
- **TFT model** achieved the strongest validation performance
- **I3D model** showed weaker standalone performance
- **Fusion model** did not outperform the TFT baseline in the current setting

This suggests that academic modality is currently the dominant source of predictive information, while the multimodal fusion branch still requires further refinement.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the TFT model
```bash
python -m training.train_tft
```

### 3. Train the I3D model
```bash
python -m training.train_i3d
```

### 4. Train the fusion model
```bash
python -m training.train_fusion
```

### 5. Run the dashboard
If your dashboard uses Streamlit, run:
```bash
streamlit run dashboard/app.py
```

If your dashboard uses another entry file, replace `app.py` with the correct file name.

## Example Results Summary
| Model | Final Validation Accuracy | Best Validation Accuracy |
|------|----------------------------|--------------------------|
| TFT | 94.00% | 96.00% |
| I3D | 35.00% | 35.00% |
| Fusion | 47.50% | 57.50% |

## Current Limitations
1. The fusion model does not yet outperform the academic-only baseline
2. The emotion/video branch is relatively weak compared with the TFT branch
3. Additional robustness experiments are still needed
4. The system requires further improvement in interpretability and deployment readiness

## Future Improvements
Possible future improvements include:
- improving the fusion strategy
- adding confidence-aware or gated fusion
- handling missing modalities more effectively
- improving the quality of emotion features
- expanding the dashboard for better result visualization
- adding more complete evaluation metrics and plots

## Author
Weichi Gao

## Project Type
Capstone Project / Undergraduate Thesis Project
