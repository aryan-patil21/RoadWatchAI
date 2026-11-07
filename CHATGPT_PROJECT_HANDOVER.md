# PROJECT HANDOVER DOCUMENT: ROADWATCH AI

**Document Type:** Technical Project Handover & System Specification  
**Target Audience:** AI Assistant (ChatGPT) & Technical Evaluators  
**Project Name:** RoadWatch AI  
**Repository:** [https://github.com/aryan-patil21/RoadWatchAI](https://github.com/aryan-patil21/RoadWatchAI)  
**Target Hardware:** Apple Silicon (macOS Metal Performance Shaders / MPS), iOS (Swift 6), Edge Run-times (TorchScript)  
**Primary Language/Stack:** Python 3.11, PyTorch (YOLOv8 Nano), ByteTrack, OpenCV, FastAPI, SwiftUI 6  
**Verification Date:** November 2026  

---

## 1. Project Overview

### 1.1 Official Project Name
**RoadWatch AI** (AI-Powered Road Safety & Predictive Driver Behaviour Analysis System for Two-Wheeler Commuters).

### 1.2 Purpose and Vision
RoadWatch AI is an intelligent edge-vision safety and Advanced Driver Assistance System (ADAS) engineered specifically for two-wheeler commuters (motorcycles, scooters, and mopeds) navigating chaotic, high-density urban Indian traffic environments (e.g., Pune, Mumbai). 

The platform turns an ordinary handlebar-mounted smartphone or dashcam into an active safety copilot. It computes real-time vehicle distances, relative approach velocities, trajectory divergence, lane departure risks, and provides glanceable head-up display (HUD) warnings with synchronized automotive chime cues. Furthermore, it operates an automated circular-buffering blackbox incident recorder that permanently saves 10-second forensic dossiers upon near-miss encounters.

### 1.3 Problem Statement
Two-wheeler commuters represent over **70% of vehicle volumes in Indian metropolitan areas** and suffer disproportionate traffic fatalities and severe injuries. Unlike passenger cars equipped with advanced ADAS systems (radar, LiDAR, ultrasonic sensors, emergency automatic braking), two-wheeler riders have:
1. **Zero crumple zones** or passive structural protection.
2. **Minimal reaction windows (<400 milliseconds)** when heavy commercial vehicles or cars suddenly cut across travel lanes or merge blindly from perimeter roads.
3. **Severe cognitive load**: Handlebar riders cannot stare at smartphone screens or complex text telemetry tables while balancing and maneuvering through unstructured traffic.
4. **Lack of road markings**: Indian roads often have faded, broken, or completely absent lane markings, rendering traditional ADAS lane-detection algorithms (which rely on pristine white European/US highway lines) ineffective.

### 1.4 The Core Ethical & Engineering Invariant: Zero Stereotyping
A foundational design rule of RoadWatch AI is **zero brand, vehicle-type, or demographic stereotyping**:

$$\mathbf{Observable\ Behaviour} \longrightarrow \mathbf{Measurable\ Kinematic\ Features} \longrightarrow \mathbf{Empirical\ Risk\ Estimation} \longrightarrow \mathbf{Actionable\ Warning}$$

The system evaluates **strictly observable physical kinematics** (longitudinal distance, lateral encroachment rate, bounding box scale expansion rate, and Time-to-Collision). A vehicle is never categorized as hazardous because it is a truck, an auto-rickshaw, or an older vehicle; risk is exclusively computed from dynamic trajectory conflict.

### 1.5 Target Audience and Intended Users
1. **Daily Two-Wheeler Commuters**: Riders in dense Asian/Indian metros using handlebar-mounted phones or action cameras.
2. **Gig-Economy & Delivery Fleets**: Swiggy, Zomato, Zepto, Dunzo, and courier delivery riders who log 6–10 hours daily in peak traffic.
3. **Commercial Fleet Safety Managers & Insurers**: Organizations seeking objective, millisecond-level telemetric accident reconstruction and near-miss auditing rather than subjective driver testimony.

---

## 2. Features and Functionality

| Feature Name | Status | Technical Implementation Mechanism |
|---|---|---|
| **YOLOv8 Multi-Class Vehicle Detection** | **Implemented** | Ultralytics YOLOv8n fine-tuned inference for `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`. |
| **ByteTrack Persistent Object Tracking** | **Implemented** | Dual-stage Kalman filter + Hungarian spatial IoU association. Tracks vehicle IDs persistently across frame occlusions. |
| **Monocular 3D Ground Projection (IPM)** | **Implemented** | Inverse Perspective Mapping transforming $(u, v)$ tire contact coordinates to real-world ground coordinates $(X_{\text{m}}, Z_{\text{m}})$. |
| **Relative Velocity & Approach Acceleration** | **Implemented** | Rolling 10-frame delta derivation: $v_{\text{rel}} = (\Delta Z / \Delta t) \times 3.6$ km/h, plus longitudinal acceleration proxy $a_{\text{long}}$. |
| **Dynamic 2D Bird's-Eye View (BEV) Radar** | **Implemented** | Top-down cockpit radar widget rendering 10m–40m range rings, 3.5m physical corridor lines, and color-coded threat reticles. |
| **Top-Hat Asphalt Stripe Lane Tracker** | **Implemented** | $11 \times 1$ morphological structuring element isolating reflective pavement texture with perspective corridor clamping ($140$px to $590$px). |
| **Lane Departure Warning (LDW)** | **Implemented** | Continuous tracking of ego motorcycle center relative to detected travel corridor; flags boundary crossing without indicator intent. |
| **Empirical Risk Scoring Engine** | **Implemented** | Hybrid multi-factor fusion: spatial proximity, lateral zone gating, bounding box expansion ($dA/dt$), approach rate, and TTC proxy. |
| **Warning Engine & Debounce State Machine** | **Implemented** | Multi-tier alert hierarchy (`SAFE`, `CAUTION`, `CRITICAL`) with cooldown timers, 3-frame debounce, and immediate escalation override. |
| **Synchronized Audio Chimes** | **Implemented** | Programmatically synthesized/transcoded stereo AAC alert sounds embedded directly into H.264 video streams. |
| **Automated Blackbox Incident Recorder** | **Implemented** | In-memory circular buffer (`collections.deque`, 5s pre-event + 5s post-event) capturing 10s video clips and telemetry JSON upon `CRITICAL` alerts. |
| **FastAPI REST & WebSocket Backend** | **Implemented** | Asynchronous service exposing `/health`, `/api/v1/analyze/frame`, `/api/v1/alerts/active`, and `/api/v1/ws/alerts`. |
| **Interactive Telemetry Studio (Web)** | **Implemented** | Browser dashboard (`/studio`) syncing live video playback with dynamic SVG radar blips, metric cards, and incident dossier inspect. |
| **Native iOS Cockpit Client (SwiftUI)** | **Implemented** | Standalone Swift Package (`ios/RoadWatch`) featuring 3-lane glanceable HUD, audio alert playback, and WebSocket networking. |
| **TorchScript Edge Model Export** | **Implemented** | Standalone 12.4 MB TorchScript binary (`yolov8n.torchscript`) for zero-Python LibTorch, iOS, and Android embedded runtimes. |
| **Pothole & Surface Defect Detection** | *Planned* | Explicitly deferred: Not feasible with current forward-horizon camera pitch without downward road surface training data. |
| **Blind-Spot Radar (Rear / Side)** | *Planned* | Explicitly deferred: Requires physical rear-facing secondary camera hardware. |
| **CoreML On-Device iPhone Inference** | *Partially Implemented* | Architecture and Swift pipeline built; model export currently serialized in TorchScript format. |

---

## 3. Technical Architecture

### 3.1 Architecture Dataflow Diagram

```text
       [ Monocular Camera Input: 854x480 @ 25.0 FPS ]
                             │
                             ▼
     [ Stage 1: Frame Ingestion & Validation (OpenCV) ]
                             │
                             ▼
    [ Stage 2: YOLOv8 Nano Perception & ByteTrack Association ]
       ├── Class filter: person, bicycle, car, motorcycle, bus, truck
       └── Spatial Kalman Filter -> Persistent Track ID Assignment
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
[ Stage 3: Monocular IPM Projection ]   [ Stage 4: Road Geometry & Lanes ]
  - Camera height h = 1.25m               - Morphological Top-Hat filter
  - Vanishing Horizon y = 265px           - Perspective boundary clamping
  - Ground Coordinates: X_m, Z_m          - Active Corridor: [140px, 590px]
  - Relative Velocity: dZ/dt * 3.6        - Lane Departure Warning (LDW)
  - 2D Bird's-Eye View (BEV) Radar        - Opposing Traffic Divergence Gating
        │                                         │
        └────────────────────┬────────────────────┘
                             ▼
     [ Stage 5: Kinematic Feature Engine (10-frame history) ]
       - Lateral & Longitudinal Velocity (v_lat, v_long)
       - Approach Acceleration (a_long)
       - Empirical Time-To-Collision (TTC_proxy)
       - Bounding Box Scale Expansion Rate (dA/dt)
                             │
                             ▼
     [ Stage 6: Hybrid Risk Estimation & Classification ]
       - Gated Threat Scoring: [0.0 to 100.0]
       - Threat Levels: SAFE (0-39) | CAUTION (40-69) | CRITICAL (70-100)
                             │
                             ▼
     [ Stage 7: Warning Arbitration State Machine ]
       - 3-Frame Debounce Filter & 3.5s Cooldown Timers
       - Critical Hazard Escalation Override
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                         ▼
[ Stage 8: Cockpit HUD & Audio ]        [ Stage 9: Blackbox Incident Auto-Recorder ]
  - Glanceable warning banner             - 5.0s Pre-event Circular Buffer (deque)
  - Floating metric bounding tags         - 5.0s Post-event Recovery Capture
  - 2D BEV radar widget render            - Direct QuickTime H.264/AAC transcode
  - Embedded stereo AAC audio track       - Forensic telemetry manifest export (.json)
        │                                         │
        └────────────────────┬────────────────────┘
                             ▼
   [ Stage 10: Client Communication & Distribution Tier ]
     ├── FastAPI Backend Service (REST + WebSocket @ 10Hz push)
     ├── Interactive Telemetry Web Studio (HTML5 Canvas/SVG + JS)
     └── Native iOS Cockpit Client (Swift 6 / SwiftUI)
```

### 3.2 Component Details
1. **Frontend / Client Layer**:
   - **Interactive Web Studio** ([`static/index.html`](file:///Users/apple/Desktop/RoadwatchAI/static/index.html)): Single-page application consuming FastAPI endpoints. Synchronizes video playback timestamps with dynamic SVG radar target rendering and forensic JSON dumps.
   - **iOS Client** ([`ios/RoadWatch`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch)): Modular Swift Package built with Swift 6 and SwiftUI. Includes `CockpitDashboardView`, `AlertBannerCard`, `RoadWatchAPIService` (async/await and WebSocket tasks), and bundled `.wav` audio alert assets.
2. **Backend Service Layer** ([`src/api.py`](file:///Users/apple/Desktop/RoadwatchAI/src/api.py)):
   - FastAPI 0.142 + Uvicorn server providing CORS-enabled REST endpoints and real-time WebSocket communication channel (`/api/v1/ws/alerts`).
   - Thread-safe `PipelineState` orchestrating tracker, geometry analyzer, feature extractor, risk engine, and warning state machine.
3. **Data & Storage Layer**:
   - Zero SQL database dependency: Designed for lightweight edge execution.
   - Structured JSON manifests for incident forensics (`outputs/incidents/*.json`).
   - Tabular CSV telemetry logs for offline training and evaluation (`outputs/behavioural_features.csv`, `outputs/risk_telemetry.csv`).

---

## 4. Development Details

### 4.1 Repository Layout
```text
RoadwatchAI/
├── README.md               # Technical portfolio documentation & benchmarks
├── requirements.txt        # Python dependency specification
├── yolov8n.torchscript     # Standalone edge inference model (12.4 MB)
├── CHATGPT_PROJECT_HANDOVER.md # This comprehensive technical handover
├── static/
│   └── index.html          # Interactive Telemetry Studio & Dynamic Radar web app
├── assets/
│   ├── demo/               # Showcase animations (showcase_demo.gif)
│   └── sounds/             # Synthesized automotive alert chimes (.wav)
├── data/
│   ├── raw/                # Commute dashcam video (sample_dashcam.mp4)
│   └── processed/          # Curation datasets (train.csv, val.csv, test.csv)
├── src/                    # Source code for vision & safety pipeline
│   ├── vehicle_detector.py # YOLOv8 model loader and inference engine
│   ├── vehicle_tracker.py  # ByteTrack multi-object tracker
│   ├── road_geometry.py    # Vanishing point and spatial zone partitioner
│   ├── spatial_radar.py    # IPM metric projector and 2D BEV radar widget
│   ├── lane_detector.py    # Top-Hat asphalt stripe filter & corridor tracker
│   ├── feature_extractor.py# Kinematic acceleration, TTC, expansion rate
│   ├── risk_engine.py      # Rule-based and logistic regression risk models
│   ├── warning_engine.py   # Debouncing, cooldown, and alert arbitration
│   ├── incident_recorder.py# Forensic circular buffer auto-recorder
│   ├── stream_runner.py    # Live stream pipeline orchestrator
│   ├── profiler.py         # Stage-by-stage latency and jitter profiler
│   └── api.py              # FastAPI REST, WebSockets, and Studio server
├── tests/                  # 38 unit and integration tests (100% passing)
├── outputs/                # Transcoded MP4s, telemetry timelines, incident clips
├── docs/
│   ├── architecture.md     # In-depth system architecture blueprint
│   ├── dataset.md          # Data curation, taxonomy, and split specification
│   └── experiments.md      # Detailed experiment logs across Milestones 0 to 14
└── ios/RoadWatch/          # Native SwiftUI iOS client codebase
```

### 4.2 Key Source Files & Their Responsibilities
- [`src/spatial_radar.py`](file:///Users/apple/Desktop/RoadwatchAI/src/spatial_radar.py): Houses `MetricGroundProjector` (Inverse Perspective Mapping transforming pixel coordinates into real-world meters $X, Z$ and approach speed $\Delta Z / \Delta t$) and `CockpitRadarWidget` (renders the 2D top-down HUD radar widget).
- [`src/lane_detector.py`](file:///Users/apple/Desktop/RoadwatchAI/src/lane_detector.py): Implements `LaneDepartureDetector`. Uses an $11 \times 1$ morphological Top-Hat asphalt filter with perspective boundary clamping ($140$ px left curb, $590$ px right divider) to define the single physical travel corridor.
- [`src/feature_extractor.py`](file:///Users/apple/Desktop/RoadwatchAI/src/feature_extractor.py): Implements `BehaviouralFeatureExtractor`. Tracks per-vehicle rolling histories to compute bounding box expansion ($dA/dt$), longitudinal velocity, lateral swerve, and Time-To-Collision proxy.
- [`src/risk_engine.py`](file:///Users/apple/Desktop/RoadwatchAI/src/risk_engine.py): Implements `RiskEngine`. Combines spatial hazard rules, lateral corridor containment, and logistic regression risk scoring.
- [`src/warning_engine.py`](file:///Users/apple/Desktop/RoadwatchAI/src/warning_engine.py): Implements `WarningEngine`. Enforces 3-frame debouncing to eliminate flickering alerts, a 3.5-second cooldown timer, and alert sound mapping.
- [`src/incident_recorder.py`](file:///Users/apple/Desktop/RoadwatchAI/src/incident_recorder.py): Implements `IncidentBlackboxRecorder`. Maintains an in-memory `collections.deque` circular buffer (5s pre-event + 5s post-event) and auto-exports incident MP4 clips and JSON telemetry manifests.
- [`src/stream_runner.py`](file:///Users/apple/Desktop/RoadwatchAI/src/stream_runner.py): Implements `LiveStreamRunner`. The end-to-end simulation runner executing the entire pipeline on input video feeds.
- [`src/profiler.py`](file:///Users/apple/Desktop/RoadwatchAI/src/profiler.py): Implements `StreamProfiler`. Profiles stage-by-stage latencies, computing $p50, p95, p99$ distributions and real-time headroom metrics.

### 4.3 Significant Technical Challenges & Solutions

#### Challenge 1: Monocular Distance and Velocity without LiDAR or Stereo Cameras
- **Problem**: Monocular dashcam footage lacks native depth information. Naive pixel heights fluctuate wildly when vehicles change perspective.
- **Solution**: Implemented planar **Inverse Perspective Mapping (IPM)** under calibrated motorcycle mounting constraints ($h = 1.25$ m, $f = 720$ px, optical horizon $y_{\text{horizon}} = 265$ px). By tracking the bottom-center tire contact point ($u, v_{\text{contact}}$), the system projects physical ground distance $Z = (f \cdot h) / (v - y_{\text{horizon}})$ and derives differential approach velocity $v_{\text{rel}} = (\Delta Z / \Delta t) \times 3.6$ km/h over a rolling Kalman window.

#### Challenge 2: Unstructured Road Markings in Indian Traffic
- **Problem**: Standard Canny/Hough lane detectors failed completely on Indian asphalt due to patchy tar, oil slicks, fading paint, and absence of lane dividers.
- **Solution**: Developed a specialized morphological **Top-Hat asphalt stripe extraction algorithm** using an $11 \times 1$ horizontal structuring element. This isolates narrow reflective linear pavement features from wide pavement variations. Combined with geometric boundary clamping ($140$ px to $590$ px at the bottom plane), it strictly defines the physical travel corridor without bleeding across multiple lanes.

#### Challenge 3: False-Positive Collision Warnings from Opposing Traffic
- **Problem**: On divided urban avenues, oncoming traffic in opposing lanes triggered severe collision warnings because their bounding boxes expand rapidly as they approach.
- **Solution**: Implemented a **lateral trajectory divergence filter**. If a vehicle has a high closing rate but its lateral trajectory ($dX/dt$) is diverging away from the ego corridor into an opposing zone ($x < -2.0$ m), the hazard engine suppresses collision warnings and classifies the event as normal opposing traffic.

#### Challenge 4: QuickTime Audio Playback Incompatibility
- **Problem**: OpenCV's `mp4v` codec silently omits audio tracks, and macOS QuickTime muted separate audio streams or required fragile external player calls (`afplay`).
- **Solution**: Standardized pipeline output transcoding using embedded FFmpeg (`imageio-ffmpeg`) into native Apple QuickTime-compliant **H.264 video (`yuv420p`, `avc1`) with stereo AAC audio at 44.1 kHz**. Warning chimes are mixed directly at their exact millisecond timestamps into the MP4 file.

### 4.4 Known Limitations & Unfinished Scope
1. **Monocular Pitch Sensitivity**: Extreme motorcycle wheelies or steep downhill braking pitches alter the physical horizon line ($y=265$ px) slightly, introducing transient distance estimation error ($\pm 10\%$). Future revisions could integrate the smartphone's IMU/gyroscope data to dynamically compensate horizon tilt.
2. **Night / Extreme Weather**: The current model was benchmarked on daylight urban commute footage. Performance under severe monsoon rain or unlit night conditions has not been validated.
3. **Pothole Detection**: Omitted from current scope as forward dashcam horizon angles are suboptimal for micro-surface depth profiling.

---

## 5. AI / ML Details

### 5.1 Models and Frameworks
- **Perception Model**: Ultralytics **YOLOv8 Nano (`yolov8n.pt`)** with 3.2 million parameters.
- **Tracking Algorithm**: **ByteTrack** (`lapx` linear assignment + Kalman filter state estimation).
- **Risk Classifier**: Hybrid system combining:
  1. Deterministic kinematic rules (TTC thresholds, corridor invasion, rapid approach).
  2. A trained Scikit-Learn **Logistic Regression / Random Forest** risk classifier ([`src/train_risk_model.py`](file:///Users/apple/Desktop/RoadwatchAI/src/train_risk_model.py)).
- **Edge Serialization**: Standalone TorchScript binary ([`yolov8n.torchscript`](file:///Users/apple/Desktop/RoadwatchAI/yolov8n.torchscript), 12.4 MB) compiled via `torch.jit`.

### 5.2 Dataset Curation & Partitioning (Milestone 7)
- **Source Data**: Commute dashcam recordings captured in Indian metropolitan traffic conditions.
- **Feature Dataset**: 283 annotated feature vectors exported to [`outputs/behavioural_features.csv`](file:///Users/apple/Desktop/RoadwatchAI/outputs/behavioural_features.csv).
- **Leakage-Free Partitioning**: Split strictly by **unique vehicle tracking IDs** (never random frame shuffling) to prevent temporal data leakage:
  - **Train Set**: 182 samples (7 unique vehicle trajectories)
  - **Validation Set**: 59 samples
  - **Test Set**: 42 samples (2 isolated vehicle trajectories)
  - **Overlap**: 0% vehicle overlap between train and test.

### 5.3 Event Taxonomy & Evaluation Metrics
- **Taxonomy Labels**: `normal_driving`, `parallel_cruising`, `closing_distance`, `rapid_approach`, `sudden_braking`, `close_following`.
- **Safety Levels**: `NORMAL` (58%), `CAUTION` (23%), `HIGH_RISK` (19%).
- **Classifier Performance**:
  - Test Accuracy: **92.8%**
  - High-Risk Precision: **0.91**
  - High-Risk Recall: **0.95** (prioritizing zero missed critical hazards).

### 5.4 What is Real vs. What is Simulated
- **Real AI**: Object detection (YOLOv8), multi-object tracking (ByteTrack), feature extraction, inverse perspective mapping, risk estimation, and TorchScript export are **100% real and run live on input frames**.
- **Real-Time Video**: Input is real commute dashcam footage (`sample_dashcam.mp4`).
- **Simulated Hardware Input**: Video frames are fed via OpenCV `VideoCapture` rather than a live physical camera sensor hardware stream.

---

## 6. Project Development Journey (Milestone Evolution)

The project followed a rigorous 15-stage iterative engineering progression:

- **Milestone 0: Environment & Hardware Architecture** (`4f659f1`)
  - Configured Python 3.11 virtual environment, verified Apple Silicon Metal Performance Shaders (MPS) GPU acceleration, established zero-stereotyping core doctrine.
- **Milestone 1: Stream Ingestion & Video Inspection** (`e6ec1a3`)
  - Built `stream_inspector.py`, verified 854x480 @ 25 FPS stream parameters and extracted reference frames.
- **Milestone 2: Vehicle Detection Engine** (`14102f7`)
  - Integrated YOLOv8n, filtering specifically for two-wheeler commute actors (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`).
- **Milestone 3: Multi-Object Persistent Tracking** (`33579c9`)
  - Integrated ByteTrack to maintain continuous vehicle identity across occlusions; added spatial trajectory trails.
- **Milestone 4: Road Geometry & Spatial Partitioning** (`6e81535`)
  - Established optical horizon homography, vanishing point projection, and lateral zone bucketing (`LEFT_ZONE`, `EGO_LANE`, `RIGHT_ZONE`, `PERIMETER`).
- **Milestone 5: Kinematic Feature Extraction** (`c348757`)
  - Implemented rolling window kinematic derivations: longitudinal velocity proxy, approach acceleration ($a_{\text{long}}$), and empirical Time-To-Collision ($\text{TTC}$).
- **Milestone 6: Risk Estimation Modeling** (`3c4ab0f`)
  - Developed multi-factor threat scoring engine mapping empirical features into $[0, 100]$ risk scores across `SAFE`, `CAUTION`, and `CRITICAL` levels.
- **Milestone 7: Dataset Curation & Taxonomy** (`b08435e`)
  - Created 283-sample leakage-free dataset split by vehicle ID; documented objective driving taxonomy.
- **Milestone 8: Warning Engine & HUD** (`d948c04`)
  - Built finite-state debouncing machine, cooldown timers, status banner rasterization, and multi-tone audio mapping.
- **Milestone 9: FastAPI Backend Service** (`0e21d12`)
  - Built asynchronous REST API and WebSocket push notifications; added sound asset distribution.
- **Milestone 10: Native iOS Client Prototype** (`db76664`)
  - Created standalone SwiftUI application (`ios/RoadWatch`) featuring 3-lane radar strip and audio triggers.
- **Milestone 11: End-to-End Pipeline Integration** (`dfc6af5`)
  - Closed the loop: video stream feeding tracking, geometry, risk, warning, and synchronized live state.
- **Milestone 12: Real-Time Stream Performance Profiling** (`651f0ae`)
  - Benchmarked stage-by-stage latencies, achieving **41.62 FPS** on Apple Silicon M4 with **+9.57 ms headroom** at $p95$.
- **Milestone 13: Metric Distance, Speed & 2D BEV Radar** (`b32194d`)
  - Integrated monocular Inverse Perspective Mapping (metric meters and relative km/h) and 2D top-down BEV radar widget.
- **Milestone 14: System Hardening & Incident Blackbox** (`b5832e7`)
  - Exported YOLOv8 to standalone TorchScript (`yolov8n.torchscript`); implemented automated circular buffer blackbox incident recorder; authored system architecture specification.
- **Showcase Packaging & Dynamic Web Studio** (`7dfc0d0`, `8ff5591`, `ec8935a`)
  - Generated animated showcase demonstration; built interactive Web Studio with dynamic real-time SVG radar synchronization; authored publication-grade documentation.

---

## 7. Achievements and Differentiation

### 7.1 What Makes RoadWatch AI Different?
1. **Designed for Two-Wheelers in Chaos, Not Cars on Highways**: Most open-source ADAS projects (e.g., Openpilot) assume four-wheeled vehicles driving on well-striped multi-lane US/European freeways. RoadWatch AI is specifically designed for the chaotic, dense, non-lane-disciplined traffic of Indian cities.
2. **Strict Zero-Stereotyping Policy**: Avoids the trap of treating large vehicles as inherently dangerous. Risk is mathematically tied to **observable kinematics** (closing rate, approach angle, TTC).
3. **Monocular Top-Down Spatial Awareness**: Projects standard 2D dashcam pixels into a 3D physical coordinate plane and renders a glanceable 2D top-down radar showing true metric distances ($Z$) and approach velocities ($\Delta Z / \Delta t$).
4. **Automated Blackbox Incident Recorder**: Solves a real commuter problem: in high-stress collision or near-miss scenarios, riders cannot press a button. The circular buffer ensures the critical pre-event cause is permanently saved.

### 7.2 Strongest Technical Aspects
- **41.6 FPS Real-Time Throughput on Apple Silicon**: Operates with +9.57 ms headroom at $p95$ tail latency on Apple Silicon M4 MPS.
- **Top-Hat Morphological Lane Extraction**: Overcomes missing lane lines by extracting asphalt texture differences.
- **Zero-Dependency Edge Export**: TorchScript model allows mobile runtimes without requiring Python or Ultralytics dependencies.

---

## 8. Current Project Status

- **Operational Status**: 100% operational locally.
- **Automated Test Suite**: **38 tests passing with 0 failures** across all modules:
  ```bash
  .venv/bin/python3 -m unittest discover tests
  # Ran 38 tests in 5.562s -> OK (100% pass rate)
  ```
- **Live Local Studio**: Running locally via FastAPI at `http://localhost:8000/studio`.
- **Public GitHub Repository**: Active and synchronized at [https://github.com/aryan-patil21/RoadWatchAI](https://github.com/aryan-patil21/RoadWatchAI) on branch `main`.
- **Deployment Status**: Designed for local edge execution and client-server simulation; not hosted on cloud PaaS (intentional for edge computing privacy and latency).

---

## 9. LinkedIn Content Preparation Guide

This section provides specific instructions for ChatGPT to craft a viral, technically authoritative LinkedIn post.

### 9.1 The 5 Most Compelling Aspects to Highlight
1. **The Real-World Commute Problem**: Riders in Indian traffic (Pune/Mumbai) have $<400$ ms to react to cut-ins, with zero passive protection.
2. **Monocular 3D Ground Projection (IPM)**: Estimating true physical distance ($Z$ in meters) and relative approach speed ($\Delta Z / \Delta t$ in km/h) from a single 2D camera without LiDAR or stereo hardware.
3. **2D Top-Down Bird's-Eye View (BEV) Radar**: Transforming complex bounding boxes into a glanceable aviation-style cockpit radar widget.
4. **Automated Blackbox Incident Auto-Recorder**: Circular in-memory buffering (`deque`) that captures 10-second forensic incident clips + telemetry JSON upon critical threats.
5. **Real-Time Edge Performance**: Achieving 41.62 FPS with sub-30ms latency on Apple Silicon M4 MPS, plus standalone TorchScript export.

### 9.2 The Strongest Technical Achievement
Deriving **metric ground coordinates ($X_{\text{m}}, Z_{\text{m}}$) and continuous relative approach velocity** from monocular dashcam video using calibrated camera geometry and horizon homography, running in real-time at 41.6 FPS.

### 9.3 The Most Interesting Problem-Solving Experience
Solving **false-positive collision alerts from opposing traffic**: Vehicles traveling in the opposite direction on divided roads expand rapidly in size, tricking naive collision algorithms. Solved by implementing **lateral trajectory divergence gating**, verifying whether a closing vehicle is diverging away from the ego travel corridor.

### 9.4 Key Technologies to Mention
`Python 3.11`, `PyTorch`, `YOLOv8`, `ByteTrack`, `OpenCV`, `Apple Silicon MPS`, `TorchScript`, `FastAPI`, `SwiftUI (iOS)`, `WebSockets`, `Inverse Perspective Mapping (IPM)`.

### 9.5 Recommended Visual Media to Attach
- **Video Attachment**: `outputs/stream_runner_output.mp4` (or `outputs/incidents/incident_001_49s.mp4`) showing the HUD, metric distance tags, 2D radar, and embedded warning beeps.
- **GIF Alternative**: `assets/demo/showcase_demo.gif` for immediate looping visual engagement.
- **Studio Screenshot**: A screenshot of the web dashboard at `http://localhost:8000/studio`.

### 9.6 What NOT to Claim or Exaggerate
- **Do NOT claim**: "Deployed on thousands of commercial bikes" (it is an engineered prototype evaluated on real commute footage).
- **Do NOT claim**: "Includes pothole and road defect detection" (explicitly deferred to future hardware iterations).
- **Do NOT claim**: "Replaces human judgment" (it is an assistive advisory ADAS, not an autonomous driving autopilot).
- **Do NOT invent**: Unrealized industry awards, patent numbers, or venture capital funding.

---

## 10. Verification Checklist

- [x] Verified repository structure against local disk.
- [x] Verified all 38 automated unit and integration tests pass cleanly.
- [x] Verified mathematical formulas against [`src/spatial_radar.py`](file:///Users/apple/Desktop/RoadwatchAI/src/spatial_radar.py) and [`src/road_geometry.py`](file:///Users/apple/Desktop/RoadwatchAI/src/road_geometry.py).
- [x] Verified benchmark statistics against [`outputs/profiler_benchmark_report.json`](file:///Users/apple/Desktop/RoadwatchAI/outputs/profiler_benchmark_report.json).
- [x] Confirmed Git commit history and branch sync with remote origin.
- [x] Documented verified limitations and future roadmap items.
