# 🏐 VolleyVision AI
> Automatic tactical analysis and scouting system for volleyball using Computer Vision and Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1_CPU-red)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌐 Live Demo

| Platform | URL |
|---|---|
| 🤗 HuggingFace Spaces | [huggingface.co/spaces/norapfr/volleyvision-ai](https://huggingface.co/spaces/norapfr/volleyvision-ai) |
| 🚀 Render | [volleyvision-ai.onrender.com](https://volleyvision-ai.onrender.com) |
| 📦 Models (HuggingFace Hub) | [huggingface.co/norapfr/volleyvision-models](https://huggingface.co/norapfr/volleyvision-models) |

> ⚠️ **Demo note:** Free-tier deployments use shared CPU. Processing a 1-minute video may take 5-10 minutes. Upload videos under 50MB for best results.

---

## 📌 Overview

VolleyVision AI is an end-to-end Computer Vision pipeline that automatically analyzes volleyball match footage to extract tactical insights and generate scouting reports.

Upload a match video and the system automatically:
- 🎯 Detects players and ball in every frame (YOLOv8)
- ⚡ Classifies tactical events: serve, attack, block, reception, setting, defense
- 🔥 Generates player position heatmaps
- 📊 Computes event frequency and timeline statistics
- 🧠 Clusters zones by tactical profile (PCA + K-Means + KNN)
- 🚀 Exposes everything through a REST API (FastAPI)

---

## 🏗️ Architecture

```
Video Input
    │
    ▼
┌─────────────────────────────────┐
│  Phase 1 — Detection (YOLOv8)  │  Players · Ball · Bounding boxes
└─────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Phase 2 — Event Classification (YOLOv8 fine-tuned) │
│  + Multi-angle augmentation fine-tuning              │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  Phase 3 — Dashboard (Streamlit) │  Timeline · Heatmaps · Stats
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Phase 4 — Scouting ML               │  PCA · K-Means · KNN
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Phase 5 — MLOps                     │  FastAPI · Docker · MLflow
│                                      │  GitHub Actions · HF Spaces · Render
└──────────────────────────────────────┘
```

---

## 📊 Model Performance

### Phase 1 — Player & Ball Detection

Trained on combined Roboflow datasets (VolleyBallYolo + Volleyball Tracker + Ball Object Detection).
**YOLOv8n · 30 epochs · CPU Intel i7-1165G7 · imgsz=640**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| person | 0.711 | 0.967 | 0.972 | 0.635 |
| ball | 0.744 | 0.684 | 0.749 | 0.342 |
| **all** | **0.728** | **0.825** | **0.861** | **0.488** |

---

### Phase 2 — Event Classification

Training followed a two-stage pipeline:

#### Stage 1 — Base training

Trained from scratch on the Graz University Volleyball Activity Dataset (25,000 images, 7 classes).
Training was split across two environments: 12 epochs on CPU (Intel i7-1165G7) followed by 20 additional epochs on GPU (Tesla T4, Google Colab) after resuming from the CPU checkpoint.
**YOLOv8n · 33 total epochs · imgsz=416**

**Validation set results:**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| attack | 0.978 | 0.960 | 0.989 | 0.855 |
| block | 0.983 | 0.976 | 0.992 | 0.882 |
| reception | 0.981 | 0.966 | 0.993 | 0.924 |
| service | 0.989 | 0.998 | 0.993 | 0.889 |
| setting | 0.947 | 0.983 | 0.986 | 0.887 |
| Defense-Move | 0.948 | 0.951 | 0.984 | 0.899 |
| stand | 0.947 | 0.994 | 0.990 | 0.953 |
| **all (val)** | **0.966** | **0.976** | **0.991** | **0.898** |

**Test set results:**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| attack | 0.975 | 0.951 | 0.962 | 0.830 |
| block | 0.991 | 0.945 | 0.944 | 0.838 |
| reception | 0.989 | 0.976 | 0.984 | 0.926 |
| service | 0.992 | 0.996 | 0.992 | 0.890 |
| setting | 0.964 | 0.970 | 0.981 | 0.871 |
| Defense-Move | 0.973 | 0.945 | 0.950 | 0.868 |
| stand | 0.919 | 0.989 | 0.983 | 0.947 |
| **all (test)** | **0.972** | **0.968** | **0.971** | **0.881** |

#### Stage 2 — Multi-angle fine-tuning

To improve robustness across different broadcast camera angles and perspectives, the base model was fine-tuned with aggressive geometric and photometric augmentations on GPU (Tesla T4, Google Colab).
**YOLOv8n · 30 epochs · imgsz=640 · GPU Tesla T4**

```python
# Augmentation parameters used for multi-angle fine-tuning
perspective = 0.0005   # simulate different camera distances
degrees     = 12.0     # rotation for tilted cameras
shear       = 8.0      # horizontal shear
scale       = 0.6      # zoom variation
translate   = 0.15     # position shift
fliplr      = 0.5      # horizontal flip
flipud      = 0.05     # vertical flip
mosaic      = 1.0      # mosaic augmentation
mixup       = 0.15     # image mixing
copy_paste  = 0.1      # copy-paste augmentation
erasing     = 0.3      # random erasing
```

**Final test set results after fine-tuning:**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| attack | 0.975 | 0.951 | 0.962 | 0.830 |
| block | 0.991 | 0.945 | 0.944 | 0.838 |
| reception | 0.989 | 0.976 | 0.984 | 0.926 |
| service | 0.992 | 0.996 | 0.992 | 0.890 |
| setting | 0.964 | 0.970 | 0.981 | 0.871 |
| Defense-Move | 0.973 | 0.945 | 0.950 | 0.868 |
| stand | 0.919 | 0.989 | 0.983 | 0.947 |
| **all** | **0.972** | **0.968** | **0.971** | **0.881** |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10 or 3.11
- Git
- Docker (optional)

### Option A — Local with Python

```bash
git clone https://github.com/norapfr/volleyvision-ai.git
cd volleyvision-ai

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt


# edit .env and add your HF_TOKEN & ROBOFLOW_API

streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`

### Option B — Docker

```bash
git clone https://github.com/norapfr/volleyvision-ai.git
cd volleyvision-ai

# edit .env with your keys

docker-compose up --build
```

Open `http://localhost:8501` — models are downloaded automatically from HuggingFace Hub on first run.

### Option C — REST API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`

---

## 📁 Project Structure

```
volleyvision-ai/
├── src/
│   ├── detection/              # Phase 1: player & ball detection
│   │   ├── train.py
│   │   ├── run_detection.py
│   │   └── extract_frames.py
│   ├── events/                 # Phase 2: event classification
│   │   ├── train_events.py
│   │   └── explore_dataset.py
│   ├── dashboard/              # Phase 3: Streamlit dashboard
│   │   └── app.py
│   ├── scouting/               # Phase 4: ML scouting
│   │   └── scouting.py
│   ├── api/                    # Phase 5: FastAPI REST API
│   │   └── main.py
│   └── mlflow_tracking/        # Phase 5: experiment tracking
│       ├── log_experiments.py
│       └── upload_models.py
├── data/
│   ├── raw/                    # original match videos (gitignored)
│   ├── processed/              # detection/event logs (gitignored)
│   └── annotations/            # YOLO datasets (gitignored)
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

---

## 🧠 Technical Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch 2.1 CPU |
| Computer Vision | OpenCV |
| Dashboard | Streamlit + Plotly |
| Scouting ML | scikit-learn (PCA, K-Means, KNN) |
| REST API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Model Registry | HuggingFace Hub |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | HuggingFace Spaces + Render |

---

## 📈 Datasets

| Dataset | Source | Images | Classes | Phase |
|---------|--------|--------|---------|-------|
| VolleyBallYolo | Roboflow Universe | ~800 | ball, person | 1 |
| Volleyball Tracker | Roboflow Universe | 838 | ball | 1 |
| Ball Object Detection | Roboflow (primaryws) | ~1.2k | ball | 1 |
| Court Keypoints | Roboflow (primaryws) | — | keypoints | 1 |
| Volleyball Activity Dataset | Graz University of Technology | 25,000 | 7 events | 2 |

### Citation

```bibtex
@misc{volleyball-activity-dataset_dataset,
  title        = {Volleyball Activity Dataset},
  author       = {activity graz uni},
  howpublished = {\url{https://universe.roboflow.com/activity-graz-uni/volleyball-activity-dataset}},
  year         = {2023},
  month        = {may},
  publisher    = {Roboflow}
}

@misc{volleyball-tracker_dataset,
  title        = {Volleyball Tracker Dataset},
  author       = {Volleyball Analyser},
  howpublished = {\url{https://universe.roboflow.com/volleyball-analyser/volleyball-tracker}},
  year         = {2024},
  month        = {mar},
  publisher    = {Roboflow}
}

@misc{volleyballyolo_dataset,
  title        = {VolleyBallYolo Dataset},
  author       = {VolleyBallYolo},
  howpublished = {\url{https://universe.roboflow.com/volleyballyolo/volleyballyolo}},
  year         = {2025},
  month        = {jan},
  publisher    = {Roboflow}
}

@misc{volleyball_court_key_points_regression_dataset,
  title        = {volleyball\_court\_key\_points\_regression\_dataset},
  author       = {primaryws},
  howpublished = {\url{https://universe.roboflow.com/primaryws/volleyball_court_key_points_regression_dataset}},
  year         = {2024},
  month        = {nov},
  publisher    = {Roboflow}
}

@misc{volleyball_ball_object_detection_dataset,
  title        = {volleyball\_ball\_object\_detection\_dataset},
  author       = {primaryws},
  howpublished = {\url{https://universe.roboflow.com/primaryws/volleyball_ball_object_detection_dataset}},
  year         = {2024},
  month        = {nov},
  publisher    = {Roboflow}
}
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Model status |
| POST | `/analyze` | Upload video and start analysis |
| GET | `/results/{job_id}` | Get analysis results |
| GET | `/jobs` | List all jobs |

Example:
```bash
curl -X POST "http://localhost:8000/analyze" \
     -F "file=@match.mp4" \
     -F "every_n=10"
```

---

## 🚢 Deployment

The project is deployed on two platforms simultaneously:

### HuggingFace Spaces
- **URL:** [huggingface.co/spaces/norapfr/volleyvision-ai](https://huggingface.co/spaces/norapfr/volleyvision-ai)
- Docker-based deployment
- Models auto-downloaded from HuggingFace Hub at startup
- Free tier, shared CPU

### Render
- **URL:** [volleyvision-ai.onrender.com](https://volleyvision-ai.onrender.com)
- Docker-based deployment
- Free tier, shared CPU

### Models (HuggingFace Hub)
- **URL:** [huggingface.co/norapfr/volleyvision-models](https://huggingface.co/norapfr/volleyvision-models)
- Public model registry with both `.pt` weights
- Auto-downloaded by the app on first run

---

## 🔭 Roadmap

- [x] Phase 1 — Player & ball detection (YOLOv8)
- [x] Phase 2 — Event classification (base training + multi-angle fine-tuning)
- [x] Phase 3 — Streamlit dashboard with heatmaps and timeline
- [x] Phase 4 — Scouting with PCA + K-Means + KNN
- [x] Phase 5 — FastAPI REST API
- [x] Phase 5 — Docker containerization
- [x] Phase 5 — GitHub Actions CI/CD
- [x] Phase 5 — MLflow experiment tracking
- [x] Phase 5 — HuggingFace Hub model registry
- [x] Phase 5 — Deploy to HuggingFace Spaces
- [x] Phase 5 — Deploy to Render
- [ ] Player tracking across frames (ByteTrack / DeepSORT)
- [ ] Court homography for top-down tactical view
- [ ] Real-time video stream analysis
- [ ] Automated PDF scouting reports

---

## 👩‍💻 Author

**Nora** · Software Engineer

[![GitHub](https://img.shields.io/badge/GitHub-norapfr-black?logo=github)](https://github.com/norapfr)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-norapfr-yellow?logo=huggingface)](https://huggingface.co/norapfr)

---

## 📄 License

MIT License — feel free to use this project as a reference for your own work.

---

*End-to-end ML engineering project covering: data acquisition, model training, multi-angle fine-tuning, interactive dashboard, REST API, containerization, CI/CD, experiment tracking, and cloud deployment on two platforms.*
