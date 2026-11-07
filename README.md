# RoadWatch AI

<div align="center">

> **Intelligent road safety & predictive driver behaviour analysis system engineered for two-wheeler commuters in high-density urban Indian traffic.**

[![GitHub Repo](https://img.shields.io/badge/GitHub-aryan--patil21%2FRoadWatchAI-181717?logo=github&style=for-the-badge)](https://github.com/aryan-patil21/RoadWatchAI)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/MPS-41.6_FPS-FF9500?logo=apple&logoColor=white&style=for-the-badge)](https://developer.apple.com/metal/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0-009688?logo=fastapi&logoColor=white&style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Swift 6](https://img.shields.io/badge/Swift-6_SPM-F05138?logo=swift&logoColor=white&style=for-the-badge)](https://developer.apple.com/swift/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<br/>

<img src="assets/demo/showcase_demo.gif" alt="RoadWatch AI Live Stream Demo" width="850px" style="border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.5);" />

*Live Cockpit Stream: Top-down 2D BEV radar locking onto cutting vehicles, Inverse Perspective Mapping metric distance ($Z$) + closing speeds ($\Delta Z / \Delta t$), asphalt stripe lane departure corridor, and synchronized audio alert chimes.*

</div>

---

## 💡 The Core Engineering Principle

Motorcyclists and scooter riders in Indian urban environments (Pune, Mumbai) face severe collision hazards from rapid cut-ins, sudden blind-spot merges, and sudden braking by heavy vehicles. 

RoadWatch AI eliminates arbitrary heuristics and enforces a **zero-stereotyping** invariant:
$$\mathbf{Behaviour} \longrightarrow \mathbf{Measurable\ Kinematic\ Features} \longrightarrow \mathbf{Empirical\ Risk\ Estimation} \longrightarrow \mathbf{Multimodal\ Warning}$$

We evaluate **observable physical kinematics**, never vehicle brands, models, or stereotypes.

---

## ⚡ Key Highlights & Technical Capabilities

- 🎯 **Monocular 3D Metric Projection (IPM)**: Projects 2D bounding-box ground contact points into real-world longitudinal distance ($Z_{\text{m}}$) and lateral offset ($X_{\text{m}}$) using calibrated camera height ($1.25$m) and horizon homography ($y=265$px).
- 📡 **2D Top-Down Bird's-Eye View (BEV) Radar**: Real-time cockpit radar widget rendering $40$m range rings, travel lane boundaries, ego motorcycle position, and dynamic threat-tracking reticles.
- 🏎️ **Dynamic Approach Velocity & TTC**: Computes instantaneous relative closing velocity ($\Delta Z / \Delta t \times 3.6$ km/h) and Time-To-Collision ($\text{TTC}_{\text{proxy}}$) across a rolling Kalman-filtered history.
- 🛣️ **Top-Hat Asphalt Stripe Lane Tracker**: Morphological asphalt stripe extraction ($11 \times 1$ structuring element) with perspective boundary clamping ($140$px to $590$px) to prevent multi-lane cross-bleed.
- 🚨 **Multimodal Warning & QuickTime-Native Audio**: Multi-level state machine (`SAFE` $\to$ `CAUTION` $\to$ `CRITICAL`) with automatic debouncing, cooldown timers, and synchronized stereo AAC automotive warning beeps embedded directly into the video container.
- 📼 **Automated Forensic Blackbox Recorder**: Circular in-memory buffer (`collections.deque`) maintaining $5$s of pre-incident history. Automatically triggers upon any `CRITICAL` hazard to capture a $10$s incident video clip and structured telemetry JSON manifest.
- 📱 **FastAPI Backend & Native iOS Cockpit Client**: Sub-50ms push delivery over WebSockets to a native SwiftUI handlebar-mounted dashboard app.
- ⚡ **TorchScript Edge Model Export**: Zero-Python runtime edge artifact (`yolov8n.torchscript`, 12.4 MB) ready for on-device deployment via LibTorch C++, iOS, or Android NDK.

---

## 📊 Apple Silicon M4 Hardware Benchmarks

Stage-by-stage profiling benchmark across 220 commute frames on Apple Silicon Metal Performance Shaders (MPS):

| Pipeline Stage | Mean Latency | Median ($p50$) | Tail ($p95$) | % of Budget | Computational Profile |
|---|---|---|---|---|---|
| **1. Frame Ingestion & Decode** | 0.23 ms | 0.22 ms | 0.30 ms | 0.9% | Negligible CPU read |
| **2. YOLOv8n + ByteTrack** | 19.19 ms | 18.98 ms | 24.26 ms | **79.9%** | Primary Neural Load (MPS) |
| **3. Road Geometry Projection** | 0.01 ms | 0.01 ms | 0.02 ms | <0.1% | Instant vectorized projection |
| **4. Kinematic Feature Math** | 0.01 ms | 0.01 ms | 0.02 ms | 0.1% | Rolling window kinematics |
| **5. Risk Scoring Engine** | 0.01 ms | 0.00 ms | 0.01 ms | <0.1% | Instant heuristic evaluation |
| **6. Top-Hat Lane Corridor** | 2.68 ms | 2.67 ms | 3.13 ms | **11.1%** | Edge & morphology filters |
| **7. Warning Arbitration** | 0.00 ms | 0.00 ms | 0.01 ms | <0.1% | State machine evaluations |
| **8. Cockpit HUD & 2D Radar** | 0.05 ms | 0.04 ms | 0.06 ms | 0.2% | Lightweight OpenCV raster |
| **9. Frame Serialization** | 1.85 ms | 1.81 ms | 2.14 ms | 7.7% | Video buffer serialization |
| **Total End-to-End Latency** | **24.03 ms** | **23.58 ms** | **30.43 ms** | **100%** | **41.62 FPS Throughput** |

> **Real-Time Headroom**: At $p95$ latency (30.43 ms), the system operates with **+9.57 ms headroom** inside the 40.0 ms (25 FPS) real-time camera budget.

---

## 🏗️ Architecture Blueprint

```text
                                 [ Monocular Dashcam Stream (854x480 @ 25 FPS) ]
                                                        │
                                                        ▼
                                    [ YOLOv8 Nano + ByteTrack Tracker ]
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         ▼                                                             ▼
         [ Inverse Perspective Mapping (IPM) ]                        [ Top-Hat Lane Corridor & LDW ]
         - Metric Coordinates: X_m, Z_m                               - Asphalt Stripe Morphology
         - Relative Velocity: dZ/dt * 3.6 km/h                        - Clamped Travel Corridor (140-590px)
         - 2D Top-Down BEV Radar Widget                               - Lateral Deviation Warning (LDW)
                         │                                                             │
                         └──────────────────────────────┬──────────────────────────────┘
                                                        ▼
                                        [ Kinematic Feature Engine ]
                                        - Longitudinal/Lateral Velocity
                                        - Approach Acceleration & Rate
                                        - Empirical Time-To-Collision (TTC)
                                                        │
                                                        ▼
                                          [ Hybrid Risk Estimation ]
                                          - Spatial Proximity & Corridors
                                          - Threat Trajectory Gating
                                          - SAFE (0-39) | CAUTION (40-69) | CRITICAL (70-100)
                                                        │
                         ┌──────────────────────────────┴──────────────────────────────┐
                         ▼                                                             ▼
           [ Multimodal Warning Engine ]                               [ Incident Blackbox Recorder ]
           - Debounce & Cooldown Arbitration                           - Circular Ring Buffer (5s pre-event)
           - Direct QuickTime H.264 + Stereo AAC                       - 5s Post-Event Recovery Capture
           - Live Augmented Cockpit HUD                                - Export MP4 + Forensic JSON Dossier
                         │                                                             │
                         └──────────────────────────────┬──────────────────────────────┘
                                                        ▼
                                        [ FastAPI REST & WebSocket Server ]
                                                        │
                                                        ▼
                                         [ Native iOS SwiftUI Cockpit ]
```

---

## 🚀 Quickstart & Reproduction

### 1. Setup Environment
```bash
git clone https://github.com/aryan-patil21/RoadWatchAI.git
cd RoadWatchAI

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Interactive Web Cockpit & Telemetry Studio
```bash
# Launch FastAPI Backend + Web Studio
python3 -m uvicorn src.api:app --reload --port 8000
```
Open **[http://localhost:8000/studio](http://localhost:8000/studio)** in your browser to view the live cockpit, interactive 2D BEV radar, and inspect the blackbox incident dossiers.

### 3. Run the End-to-End Stream Runner
```bash
python3 src/stream_runner.py --start-frame 1000 --max-frames 250 --mode max
```

### 4. Run the Full Automated Test Suite
```bash
python3 -m unittest discover tests
# 38 passed in 5.5s (100% pass rate)
```

---

## 📁 Repository Structure
```text
RoadwatchAI/
├── README.md               # Visual portfolio presentation
├── requirements.txt        # Python package dependencies
├── yolov8n.torchscript     # Standalone TorchScript edge model (12.4 MB)
├── static/                 # Interactive Web Cockpit & Telemetry Studio
├── assets/                 # Sounds, icons, and animated demo media
├── data/raw/               # Raw commute dashcam footage
├── src/                    # End-to-end vision & risk pipeline
│   ├── vehicle_tracker.py  # YOLOv8 + ByteTrack multi-object tracking
│   ├── spatial_radar.py    # IPM metric distance, relative speed, 2D BEV radar
│   ├── lane_detector.py    # Top-Hat stripe extraction & LDW corridor
│   ├── feature_extractor.py# Kinematic acceleration, TTC, expansion rate
│   ├── risk_engine.py      # Rule-based & logistic threat evaluation
│   ├── warning_engine.py   # Debounce state machine & multimodal alerts
│   ├── incident_recorder.py# Circular buffer forensic blackbox auto-recorder
│   ├── stream_runner.py    # Closed-loop live dashcam simulator
│   ├── profiler.py         # Precision stage-by-stage latency profiler
│   └── api.py              # FastAPI REST, WebSockets & Studio backend
├── tests/                  # 38 unit and integration tests
├── outputs/                # Transcoded MP4s, forensic incident JSONs
├── docs/                   # Architectural blueprints & experiment logs
│   ├── architecture.md     # Full system architecture specification
│   └── experiments.md      # Detailed experiment diary (M0 to M14)
└── ios/RoadWatch/          # Native SwiftUI handlebar cockpit client
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
