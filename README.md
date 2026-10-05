# RoadWatch AI

> **AI-powered road safety and driver-behaviour analysis system for two-wheeler commuters.**

## Core Philosophy
RoadWatch AI analyzes **observable driving behaviour** (rapid approach, unsafe following distance, aggressive lane changes, swerving) to estimate immediate road-safety risks and warn the rider.

**Behaviour → Measurable Features → Risk Estimation → Warning**

We evaluate **observable actions**, not vehicle brands, models, or stereotypes.

---

## Project Structure
```text
RoadwatchAI/
├── README.md               # Project overview and documentation
├── requirements.txt        # Python package dependencies
├── .gitignore             # Ignored files (virtual envs, media, caches)
├── data/
│   ├── raw/               # Raw dashcam video files (.mp4)
│   ├── processed/         # Sampled frames, cropped objects
│   └── samples/           # Reference images and short clips
├── notebooks/              # Exploration and visual analysis notebooks
├── src/                    # Source code for the vision & risk pipeline
├── tests/                  # Unit and integration test suites
├── outputs/                # Generated videos, CSVs, logs, annotated frames
└── docs/                   # Architectural notes, experiment logs, guides
```

---

## Setup & Quickstart

### 1. Prerequisites
- macOS on Apple Silicon (M-series)
- Python 3.11
- Git

### 2. Environment Activation
```bash
# Activate the dedicated virtual environment
source .venv/bin/activate

# Verify Python version
python --version
# Should output: Python 3.11.x
```

### 3. Install Core Dependencies
```bash
pip install -r requirements.txt
```

---

## Development Milestones
- [x] **Milestone 0: Development Environment** (Audit, directory structure, Python 3.11 venv)
- [x] **Milestone 1: Video Input & Frame Inspection** (OpenCV basics, dashcam video loading)
- [x] **Milestone 2: Basic Vehicle Detection** (Pretrained YOLO model inference)
- [x] **Milestone 3: Object Tracking** (Persistent vehicle IDs across frames)
- [x] **Milestone 4: Road Geometry & Relative Position**
- [x] **Milestone 5: Behavioural Feature Extraction**
- [ ] **Milestone 6: Risk Model Prototype**
- [ ] **Milestone 7: Custom Dataset Curation**
- [ ] **Milestone 8: Warning Engine**
- [ ] **Milestone 9: Backend API (FastAPI)**
- [ ] **Milestone 10: iOS Client Prototype (SwiftUI)**
- [ ] **Milestone 11: End-to-End Pipeline Integration**
- [ ] **Milestone 12: Real-Time Stream Performance Profiling**
- [ ] **Milestone 13: Advanced Features & Context Analysis**
