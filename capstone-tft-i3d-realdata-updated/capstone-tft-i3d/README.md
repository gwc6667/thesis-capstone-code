# Capstone Project: A Unified AI Framework for Academic Performance Prediction and Emotion Analysis

## Project Overview
This project now supports a real-data workflow for:
- **TFT academic prediction** using **OULAD**
- **I3D-like video emotion recognition** using **eNTERFACE'05**
- **Fusion** as a prototype feature-level integration stage built from the two trained branches

## Data Placement
Place the datasets in these folders before training:

```text
capstone-tft-i3d/
├── data/
│   ├── performance/
│   │   ├── studentInfo.csv
│   │   ├── studentAssessment.csv
│   │   ├── studentVle.csv
│   │   ├── assessments.csv
│   │   ├── courses.csv
│   │   └── ...
│   └── emotions/
│       └── enterface_database/
│           ├── subject 1/
│           ├── subject 2/
│           └── ...
```

The emotion folder also supports the original folder name `enterface database` with a space.

## How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Optional preprocessing checks
```bash
python -m preprocess.preprocess_performance
python -m preprocess.preprocess_emotions
```

### 3. Train models
```bash
python -m training.train_tft
python -m training.train_i3d
python -m training.train_fusion
```

### 4. Summarize results
```bash
python -m eval.summarize_results
```

## Notes
- The TFT branch builds 12-step academic sequences from OULAD using VLE activity, active days, assessment score, assessment count, and studied credits.
- The I3D branch reads `.avi/.mp4/.mov/.mkv` files from the eNTERFACE structure and samples a fixed number of frames from each video.
- The fusion stage is a **prototype** because OULAD and eNTERFACE are different datasets and do not provide natural subject-level multimodal pairing.
- The training scripts save feature tensors under `results/` for downstream fusion.

## Result Table Format
The summary script prints a thesis-friendly table with:
- Accuracy
- Precision
- Recall
- F1 Score
