# RoadWatch AI — System Architecture Specification

## 1. Executive Summary & Core Principle
**RoadWatch AI** is an intelligent road safety and driver behaviour analysis platform specifically tailored for two-wheeler commuters navigating high-density Indian urban traffic conditions (e.g. Pune, Mumbai).

### Foundational Invariant: Zero Stereotyping
The system adheres to an immutable behavioral paradigm:
$$\text{Behaviour} \longrightarrow \text{Measurable Features} \longrightarrow \text{Risk Estimation} \longrightarrow \text{Actionable Warning}$$

All risk decisions are derived purely from **empirical vehicle kinematics** (longitudinal/lateral velocity, Time-To-Collision proxy, lane deviation, relative approach velocity). Zero weight or bias is assigned to vehicle brand, vehicle model, age, or visual appearance.

---

## 2. End-to-End Pipeline Architecture

```text
+-----------------------------------------------------------------------------------------+
|                                1. VIDEO INGESTION & DECODE                              |
|   Monocular Dashcam Feed (854x480 @ 25 FPS) -> Frame Validation & Color Conversion      |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                               2. PERCEPTION & MULTI-OBJECT TRACKING                     |
|   - YOLOv8 Nano (MPS / TorchScript) -> Vehicle Bounding Boxes                           |
|   - ByteTrack Algorithm -> Spatial Kalman Filter + IoU Associator                       |
|   - Persistent IDs (car, bus, truck, motorcycle, bicycle, person)                       |
+-----------------------------------------------------------------------------------------+
                     |                                                 |
                     v                                                 v
+-----------------------------------------+   +-------------------------------------------+
|      3. SPATIAL PROJECTION & METRIC BEV |   |       4. LANE PERCEPTION & DEVIATION      |
| - Inverse Perspective Mapping (IPM)     |   | - Top-Hat Asphalt Stripe Filter           |
| - Vanishing point (y=265), f=720px      |   | - Bilateral Filtering + Canny Edges       |
| - Physical Ground Projection: X_m, Z_m  |   | - Clamped Travel Corridor (140px - 590px) |
| - Rolling Speed: v_rel = (dZ/dt) * 3.6  |   | - Lateral Deviation (LDW) Detection       |
| - 2D Top-Down BEV Radar Widget Render   |   |                                           |
+-----------------------------------------+   +-------------------------------------------+
                     \                                                 /
                      \                                               /
                       v                                             v
+-----------------------------------------------------------------------------------------+
|                               5. KINEMATIC FEATURE ENGINE                               |
| - Lateral & Longitudinal Velocities: v_lat, v_long                                      |
| - Closing Rate & Approach Acceleration: a_long                                          |
| - Empirical Time-To-Collision (TTC) Proxy: (y_bot - y_prev) / dt                        |
| - Bounding Box Scale Rate: dA / dt                                                      |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                               6. HYBRID RISK ESTIMATION                                 |
| - Spatial Threat Density: proximity, lateral alignment, blind-spot sector              |
| - Kinematic Trajectory Threat: rapid closure, cut-in divergence, hard braking           |
| - Dynamic Weight Fusion: Composite Risk Score [0, 100]                                  |
| - Decision Gates: SAFE (0-39), CAUTION (40-69), CRITICAL (70-100)                       |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
|                               7. WARNING ARBITRATION & HUD                              |
| - Debouncing & Escalation State Machine (cooldown timers, priority override)            |
| - Visual Augmented Cockpit HUD (HUD banner, bounding box tags, 2D radar reticle)       |
| - Audio Engine: Direct Apple H.264 + Stereo AAC synchronized warning beeps             |
+-----------------------------------------------------------------------------------------+
                     |                                                 |
                     v                                                 v
+-----------------------------------------+   +-------------------------------------------+
|     8. BLACKBOX INCIDENT AUTO-RECORDER  |   |        9. FASTAPI & CLIENT PLATFORMS      |
| - Circular Rolling Buffer (5s pre-event)|   | - FastAPI Async Backend (REST + WebSocket)|
| - Triggered on CRITICAL Alert           |   | - Thread-safe Global Pipeline State       |
| - Records 5s post-event recovery        |   | - Native iOS Client (SwiftUI + SPM Audio) |
| - Outputs Transcoded MP4 + JSON Telemetry|  | - Sub-50ms WebSocket Alert Streaming      |
+-----------------------------------------+   +-------------------------------------------+
```

---

## 3. Core Subsystems

### 3.1 Perception & Object Tracking
- **Model**: YOLOv8n optimized for Apple Silicon MPS (Metal Performance Shaders) and exported to standalone TorchScript (`yolov8n.torchscript`, 12.4 MB) for zero-dependency edge execution.
- **Tracker**: ByteTrack multi-object tracker using dual-stage matching:
  - High-confidence detection matching with Kalman filter state propagation.
  - Low-confidence detection recovery using Hungarian IoU matching to prevent track fragmentation during occlusions.

### 3.2 Monocular Metric Projection & 2D BEV Radar
- **Camera Calibration**: Assumed mounting height $h_{\text{cam}} = 1.25$ m, vertical focal length $f = 720$ px, optical center $y_{\text{horizon}} = 265$ px.
- **Metric Coordinate Derivation**:
  $$Z = \frac{f \cdot h_{\text{cam}}}{\max(1, y_{\text{contact}} - y_{\text{horizon}})}$$
  $$X = \frac{(x_{\text{contact}} - c_x) \cdot Z}{f}$$
- **Velocity Derivation**: Rolling 10-frame window computes relative closing velocity $v_{\text{rel}} = \frac{\Delta Z}{\Delta t} \times 3.6$ km/h.
- **2D Top-Down Radar**: Renders top-down bird's-eye view with 10m concentric range rings, ego vehicle marker, lane guidelines, and color-coded vehicle markers with dynamic lock-on threat reticles.

### 3.3 Lane Perception & Departure Warning
- **Top-Hat Morphological Asphalt Filter**: Isolates bright reflective pavement stripes and road edges against varied asphalt textures.
- **Travel Corridor Boundary Clamping**: Prevents cross-lane bleed by constraining the active travel corridor (bottom width clamped between 140 px and 590 px).
- **Lane Departure Warning (LDW)**: Tracks ego motorcycle trajectory relative to corridor center; alerts rider when lane boundary crossing occurs without indicator intent.

### 3.4 Behavioral Kinematics & Risk Modeling
- Evaluates observable vehicle dynamics:
  - $\Delta A / \Delta t$: Bounding box expansion rate.
  - $\text{TTC}_{\text{proxy}}$: Instantaneous approach rate toward ego vehicle.
  - $\text{Cut-in Hazard}$: Sudden lateral shift with narrowing longitudinal gap.
- **Composite Score Engine**: Blends spatial proximity, kinematic approach velocity, and lane encroachment into a standardized $[0, 100]$ score:
  - `SAFE`: $[0, 39]$ — Normal traffic flow.
  - `CAUTION`: $[40, 69]$ — Amber alert; heightened attention advised.
  - `CRITICAL`: $[70, 100]$ — Red alert; immediate rider intervention (brake/steer) required.

### 3.5 Incident Blackbox Auto-Recorder
- **Circular Buffer**: Maintains 5.0 seconds of historical video and telemetry in memory using `collections.deque`.
- **Automatic Event Trigger**: Activated instantly upon any `CRITICAL` risk event.
- **Post-Event Capture**: Continues recording for 5.0 seconds post-incident to capture aftermath and driver maneuvers.
- **Output Artifacts**:
  1. QuickTime-native MP4 (`libx264`, `yuv420p`, `avc1`) with embedded synchronized stereo AAC audio track.
  2. Structured telemetry JSON manifest (`incident_XXX_telemetry.json`) with full kinematic timeline.

### 3.6 Communication & Client Tier
- **FastAPI Service**: Provides REST endpoints (`/health`, `/api/v1/alerts/active`, `/api/v1/alerts/history`) and real-time bidirectional WebSocket stream (`/ws/alerts`).
- **iOS Cockpit Client**: Native SwiftUI dashboard with interactive 3-lane radar, audio cue playback, and live alert banner updates designed for handlebar-mounted iPhones.

---

## 4. Hardware Benchmarks & Performance Budget

Evaluated on Apple Silicon M4 under peak multi-vehicle tracking workloads:
- **Frame Resolution**: 854 x 480 @ 25 FPS
- **Frame Budget**: 40.00 ms
- **Achieved Throughput**: **41.62 FPS** (Max throughput mode)
- **Latency Distribution**:
  - Median ($p50$): **23.58 ms**
  - Tail ($p95$): **30.43 ms**
  - Peak Spike ($p99$): **36.61 ms**
- **Headroom**: **+9.57 ms** real-time headroom at $p95$.
- **Perception Load**: 79.9% of latency resides in neural inference + tracking; feature extraction, risk scoring, and warning arbitration execute in $< 0.3$ ms combined.

---

## 5. Milestone Progression Matrix

| Milestone | Scope | Key Artifact | Status |
|---|---|---|---|
| **M0** | Dev Environment & Audit | Python 3.11 venv, Apple Silicon MPS verification | Completed |
| **M1** | Video Stream Inspection | `src/stream_inspector.py`, metadata validation | Completed |
| **M2** | Vehicle Detection | `src/detector.py`, YOLOv8 inference pipeline | Completed |
| **M3** | Object Tracking | `src/tracker.py`, ByteTrack multi-object persistence | Completed |
| **M4** | Road Geometry | `src/road_geometry.py`, vanishing point & lane projection | Completed |
| **M5** | Kinematic Features | `src/features.py`, TTC, velocity, closing rates | Completed |
| **M6** | Risk Modeling | `src/risk_engine.py`, rule + logistic risk scoring | Completed |
| **M7** | Dataset Curation | `src/dataset_generator.py`, 283 annotated samples | Completed |
| **M8** | Warning Engine | `src/warning_engine.py`, debouncing & multi-level alerts | Completed |
| **M9** | Backend API | `src/api.py`, FastAPI REST & WebSocket server | Completed |
| **M10** | iOS Cockpit Client | `ios/RoadWatch`, SwiftUI handlebar cockpit app | Completed |
| **M11** | End-to-End Stream Runner | `src/stream_runner.py`, closed-loop pipeline runner | Completed |
| **M12** | Performance Profiling | `src/profiler.py`, latency breakdown & percentile report | Completed |
| **M13** | Metric BEV Radar | `src/spatial_radar.py`, IPM meters & 2D cockpit radar | Completed |
| **M14** | System Hardening & Export | `src/incident_recorder.py`, `yolov8n.torchscript` | Completed |
