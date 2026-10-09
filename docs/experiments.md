# RoadWatch AI - Experiment Log

## Experiment 01: Milestone 1 - Video Input & Frame Inspection
- **Objective**: Verify that raw dashcam footage from the ZOMAI dashcam can be decoded and processed sequentially using OpenCV, inspecting resolution, frame rate, total duration, and frame data structures.
- **Dataset**: `data/raw/sample_dashcam.mp4` (~11 MB, 1-minute real commute footage).
- **Method**: Implemented [`inspect_video`](file:///Users/apple/Desktop/RoadwatchAI/src/inspect_video.py) using `cv2.VideoCapture`. Read video properties via `cv2.CAP_PROP_*`, decoded frames at 0%, 25%, 50%, and 75% marks, and verified NumPy array representation `(H, W, C)`.
- **Result**:
  - Resolution: 854 x 480 (FWVGA 16:9 widescreen)
  - Frame Rate: 25.00 FPS
  - Total Frames: 1,895 frames
  - Duration: 75.80 seconds (~1.26 minutes)
  - Successfully decoded frames into `uint8` NumPy arrays with shape `(480, 854, 3)`.
  - Extracted sample frames saved in `outputs/`.
- **Problems**: None. The video container and H.264 video stream decoded cleanly on Apple Silicon without codec missing errors.
- **Next Step**: Milestone 2 — Introduce a lightweight pretrained YOLO model (e.g., YOLOv8n) to detect vehicles in each frame.

## Experiment 02: Milestone 2 - Basic Vehicle Detection (YOLOv8n)
- **Objective**: Implement vehicle detection on real commute footage using a pretrained lightweight YOLO model (`yolov8n.pt`) with Apple Metal (MPS) acceleration; evaluate inference throughput, bounding box precision, and class distribution.
- **Dataset**: `data/raw/sample_dashcam.mp4` (first 150 frames, 6.0 seconds).
- **Method**: Implemented [`VehicleDetector`](file:///Users/apple/Desktop/RoadwatchAI/src/vehicle_detector.py). Filtered COCO classes for road targets: `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`. Rendered bounding boxes, confidence tags, and HUD counters. Generated `outputs/annotated_detection_sample.mp4` and `outputs/detection_frame_00000.jpg`.
- **Result**:
  - Device: Apple Metal Performance Shaders (`MPS`) on Apple Silicon M4.
  - Processing speed: ~19.8 FPS on 854x480 video.
  - Total detections across 150 frames: 271 objects.
  - Class distribution: `car: 81`, `truck: 181`, `bus: 7`, `person: 2`.
  - Detection confidence: Stable (e.g., lead truck detected with 0.65+ confidence).
- **Problems / Key Observations**:
  1. *Frame-by-frame isolation*: In frame 10, a truck is detected; in frame 11, it is detected again. But the model has no concept that this is the *same* truck! It treats every frame as a completely fresh, disconnected world.
  2. *Indian Traffic context*: Heavy vehicles (buses/trucks/tempos) are prevalent and correctly flagged. However, autorickshaws (not in COCO) are typically detected as `car` or `truck` depending on viewing angle.
- **Next Step**: Milestone 3 — Object Tracking. Introduce persistent tracking IDs (e.g. ByteTRACK / BoT-SORT) so RoadWatch remembers vehicles across time.

## Experiment 03: Milestone 3 - Multi-Object Tracking (ByteTrack)
- **Objective**: Maintain temporal continuity across frames by assigning persistent tracking IDs to individual vehicles and rendering motion trajectory tails.
- **Dataset**: `data/raw/sample_dashcam.mp4` (200 frames, 8.0 seconds).
- **Method**: Implemented [`VehicleTracker`](file:///Users/apple/Desktop/RoadwatchAI/src/vehicle_tracker.py) integrating YOLOv8n with ByteTrack and `lapx`. Stored center coordinates in a rolling deque (`maxlen=30`) for each active `track_id`. Rendered dynamic trajectory paths and per-ID color codes. Generated `outputs/annotated_tracking_sample.mp4` and `outputs/tracking_frame_sample.jpg`.
- **Result**:
  - Processing throughput: **31.4 FPS** on Apple Silicon M4 (faster than 25 FPS real-time capture).
  - Maintained stable tracking IDs across all 200 frames (e.g. Vehicle ID #1 remained locked onto the leading truck for 70+ consecutive frames).
  - Trajectory tails clearly visualize the path taken by vehicles as they navigate the road.
- **Key Concepts Learned**:
  1. *Detection vs. Tracking*: Detection answers "what is where now"; tracking answers "where did that specific object move over time".
  2. *ByteTrack Mechanism*: Kalman Filter predicts where the box should be in frame $t+1$; Hungarian Algorithm pairs predicted boxes with new detections via Intersection over Union (IoU).
  3. *ID Switching & Occlusion*: When a vehicle is briefly hidden behind another or leaves camera boundaries, the tracker holds its state for a grace period before terminating the ID.
- **Next Step**: Milestone 4 — Road Geometry and Relative Position. Extract image-space spatial metrics: bounding-box center $(X, Y)$, relative distance proxy (box height/area), and approach/retreat direction.

## Experiment 04: Milestone 4 - Road Geometry & Relative Position
- **Objective**: Extract spatial properties (lateral lane zones, monocular proximity proxies, longitudinal approach/receding trends, and lateral shifts) and export structured telemetry.
- **Dataset**: `data/raw/sample_dashcam.mp4` (200 frames, 8.0 seconds).
- **Method**: Implemented [`RoadGeometryAnalyzer`](file:///Users/apple/Desktop/RoadwatchAI/src/road_geometry.py) and [`SpatialPipeline`](file:///Users/apple/Desktop/RoadwatchAI/src/spatial_pipeline.py).
  - Divided camera field into `LEFT_ZONE`, `EGO_LANE` (travel corridor), and `RIGHT_ZONE`.
  - Computed optical proximity proxy using bounding-box height ratio and ground-plane contact point $(cx, y_2)$.
  - Calculated longitudinal derivative $\Delta h / \Delta t$ and $\Delta y_2 / \Delta t$ to classify vehicles as `APPROACHING`, `RECEDING`, or `STABLE`.
  - Rendered guidelines and dynamic tags on `outputs/annotated_geometry_sample.mp4`.
  - Exported structured telemetry logs to `outputs/spatial_telemetry_sample.json` and `outputs/spatial_telemetry_sample.csv`.
- **Result**:
  - Processing speed: **31.1 FPS** on Apple Silicon M4.
  - Successfully logged 285 structured telemetry records.
  - Correctly flagged lead vehicle #1 approaching when accelerating towards it, and identified when it was situated in the rider's left zone vs. ego lane.
- **Key Concepts Learned**:
  1. *Scale Ambiguity in Monocular Vision*: Without depth sensors (LiDAR/Stereo) or extrinsic pitch calibration, exact metric distance in meters cannot be assumed; optical proxies (bounding box scale and ground-plane $Y$) provide reliable relative proximity.
  2. *Ground Contact Point*: The bottom edge of the bounding box $(cx, y_2)$ is far more informative than the geometric center because it represents the physical contact point with the road surface.
- **Next Step**: Milestone 5 — Behavioural Feature Extraction. Aggregate vehicle observations into numerical feature vectors (approach rate, velocity proxy, lateral velocity, acceleration proxy) ready for machine learning risk models.

## Experiment 05: Milestone 5 - Behavioural Feature Extraction
- **Objective**: Convert frame-by-frame temporal tracking and geometric snapshots into numerical behavioural feature vectors ($v_{\text{long}}$, $v_{\text{lat}}$, approach rate, acceleration, TTC proxy, cutting indicators) and export a complete tabular dataset for ML risk modeling.
- **Dataset**: `data/raw/sample_dashcam.mp4` (200 frames, 8.0 seconds).
- **Method**: Implemented [`BehaviouralFeatureExtractor`](file:///Users/apple/Desktop/RoadwatchAI/src/feature_extractor.py) and [`BehaviourPipeline`](file:///Users/apple/Desktop/RoadwatchAI/src/behaviour_pipeline.py).
  - Used a rolling window ($\Delta t \approx 0.32$s) with temporal smoothing to eliminate sensor bump noise.
  - Calculated:
    - Longitudinal velocity proxy: $v_{\text{long}} = \Delta y_2 / \Delta t$
    - Lateral velocity proxy: $v_{\text{lat}} = \Delta cx / \Delta t$
    - Approach rate: $\Delta P / \Delta t$
    - Longitudinal acceleration proxy: $a = \Delta v_{\text{long}} / \Delta t$
    - Time-to-Collision proxy: $TTC = (1 - P) / (\text{approach\_rate})$
    - Lane cutting flag: triggered by rapid lateral shifts across the ego corridor.
  - Exported numerical tabular dataset to `outputs/behavioural_features.csv` (283 samples).
  - Generated annotated video `outputs/annotated_behaviour_sample.mp4` and sample frame `outputs/behaviour_frame_sample.jpg`.
- **Result**:
  - Processing speed: **24.8 FPS** on Apple Silicon M4 with real-time video rendering.
  - Correctly differentiated between steady cruising ($TTC = \text{SAFE}$) and closing gaps ($TTC < 4.5$s).
  - Vehicle #1 approach rate accurately recorded peaking at $+0.125$/s with closing velocities up to $37.5$ px/s.
- **Key Concepts Learned**:
  1. *From Pixels to Numbers*: Raw RGB frames $(480, 854, 3)$ contain millions of uncurated numbers. Feature extraction condenses that visual chaos into structured rows of high-signal physical variables that standard ML algorithms (Random Forest, XGBoost, Logistic Regression) can easily classify.
  2. *Temporal Windowing & Noise Filtering*: Instantaneous differences between frame $t$ and frame $t-1$ are noisy due to road bumps and camera vibration. Rolling windows over $5-8$ frames ($\approx 0.2-0.3$s) produce smooth, reliable derivatives.
- **Next Step**: Milestone 6 — Risk Model Prototype. Implement an interpretable, rule-based baseline risk engine (LOW, MEDIUM, HIGH) and prepare for machine learning comparison.

## Experiment 06: Milestone 6 - Risk Model Prototype (Rule-Based & ML Comparison)
- **Objective**: Build a dual-layer risk estimation engine:
  1. An interpretable rule-based baseline mapping physical features to continuous risk scores [0 - 100] and categorical levels (`LOW`, `MEDIUM`, `HIGH`).
  2. Supervised Machine Learning classifiers (Logistic Regression and Random Forest) trained on the extracted behavioural features with balanced class weights.
- **Dataset**: `outputs/behavioural_features.csv` (283 samples: 215 LOW, 61 MEDIUM, 7 HIGH).
- **Method**:
  - Implemented [`RiskEngine`](file:///Users/apple/Desktop/RoadwatchAI/src/risk_engine.py): Transparent rules combining proximity (up to 30 pts), approach rate & TTC (up to 35 pts), trajectory/lane cutting (up to 30 pts), and sudden braking (up to 15 pts).
  - Implemented [`train_and_evaluate_models`](file:///Users/apple/Desktop/RoadwatchAI/src/train_risk_model.py): Trained Logistic Regression and Random Forest with `class_weight="balanced"`.
  - Implemented [`RiskPipeline`](file:///Users/apple/Desktop/RoadwatchAI/src/risk_pipeline.py): Overlaid dynamic risk bounding boxes (Green=LOW, Amber=MEDIUM, Red=HIGH) and rider alert HUD banners.
  - Generated `outputs/annotated_risk_sample.mp4` and `outputs/risk_frame_sample.jpg`.
- **Result**:
  - End-to-end pipeline speed: **32.2 FPS** on Apple Silicon M4.
  - Logistic Regression accuracy: **89.47%**.
  - Random Forest accuracy: **98.25%**.
  - Top feature drivers identified by Random Forest:
    1. `proximity_score`: 29.8%
    2. `ttc_proxy_sec`: 23.7%
    3. `approach_rate`: 17.6%
    4. `v_long_proxy`: 14.6%
    5. `acceleration_proxy`: 9.1%
- **Key Concepts Learned**:
  1. *Rule-Based vs. Machine Learning*: In safety-critical applications, having an interpretable rule baseline allows strict safety guarantees. ML models can then learn nuanced non-linear boundaries.
  2. *Class Imbalance in Road Safety*: In normal commuting, 80-95% of frames are `LOW_RISK`. Dangerous (`HIGH_RISK`) events are rare. Without `class_weight="balanced"`, ML models tend to ignore minority danger classes.
  3. *Zero Stereotyping*: The risk score is mathematically decoupled from vehicle class or plate context — a bus cruising steadily scores LOW (15-25), while an aggressive overtaking car or tempo scores HIGH (75-90).
- **Next Step**: Milestone 7 — Build a Small Custom Dataset. Systematically annotate and curate balanced road event clips (normal cruising, sudden cut-in, rapid approach, tailgate) to expand model training beyond single-clip evaluation.

## Experiment 07: Milestone 7 - Custom Dataset Curation & Leakage-Free Splitting
- **Objective**: Establish an objective kinematic road-event taxonomy, annotate behavioural samples, and generate train/validation/test splits using Entity Grouping to guarantee zero Video Data Leakage.
- **Dataset**: `outputs/behavioural_features.csv` (283 samples across 11 tracked vehicles).
- **Method**:
  - Authored [`docs/dataset.md`](file:///Users/apple/Desktop/RoadwatchAI/docs/dataset.md) documenting objective physical thresholds for 6 distinct road manoeuvres: `normal_driving`, `close_following` (tailgating), `rapid_approach`, `unsafe_lane_change`, `sudden_braking`, and `parallel_cruising`.
  - Implemented [`create_custom_dataset`](file:///Users/apple/Desktop/RoadwatchAI/src/dataset_generator.py) applying `GroupShuffleSplit` on `vehicle_id`.
  - Partitioned samples:
    - Train: 182 samples (7 unique vehicle entities)
    - Val: 59 samples
    - Test: 42 samples (2 unique vehicle entities)
  - Verified cross-split vehicle overlap: 0 overlapping vehicle IDs.
  - Exported `data/processed/roadwatch_dataset_all.csv`, `train.csv`, `val.csv`, `test.csv`, and `dataset_metadata.json`.
- **Result**:
  - Event Distribution: Parallel Cruising: 152, Closing Distance: 56, Rapid Approach: 42, Normal Driving: 12, Sudden Braking: 11, Close Following: 10.
  - Safety Level Distribution: NORMAL: 164, CAUTION: 66, HIGH_RISK: 53.
  - Data Leakage check: `is_leakage_free: True` (0 entity overlap).
- **Key Concepts Learned**:
  1. *Video Data Leakage*: In time-series vision, random train/test splitting contaminates test sets with near-identical adjacent frames. Grouping by vehicle identity guarantees models are evaluated on completely unseen vehicles.
  2. *Objective Kinematic Criteria*: Replacing vague human intuition ("this driver looks aggressive") with explicit math (approach rate $> 0.12$/s, $TTC \le 4.0$s) ensures fairness and reproducibility.
- **Next Step**: Milestone 8 — Warning Engine. Build thresholding, debouncing, and alert cooldown logic to prevent alert fatigue on riders.

## Experiment 08: Milestone 8 - Rider Warning Engine
- **Objective**: Implement a Human-Centered Warning Engine to eliminate Alert Fatigue through temporal debouncing (3-frame persistence), cooldown timers (3.5s repeat suppression), immediate alert escalation, and threat prioritization.
- **Dataset**: `data/raw/sample_dashcam.mp4` (200 frames, 8.0 seconds).
- **Method**:
  - Implemented [`WarningEngine`](file:///Users/apple/Desktop/RoadwatchAI/src/warning_engine.py):
    - Configured thresholds: CAUTION ($40.0$), CRITICAL ($70.0$).
    - Debouncing: Requires high-risk condition to persist across $\ge 3$ consecutive frames before firing.
    - Cooldown: Enforces a 3.5s quiet window per vehicle for repeated alerts of the same severity.
    - Escalation override: A transition from CAUTION to CRITICAL immediately overrides cooldown.
    - Prioritization: When multiple vehicles enter hazardous envelopes, the top-risk vehicle is presented.
  - Implemented [`WarningPipeline`](file:///Users/apple/Desktop/RoadwatchAI/src/warning_pipeline.py):
    - Rendered a motorcycle cockpit alert banner (Safe Green indicator, Amber Caution, Red Critical Warning with directional arrow).
  - Exported structured alert logs: `outputs/rider_warnings_log.json` and `outputs/rider_warnings_log.csv`.
  - Generated `outputs/annotated_warning_sample.mp4` and `outputs/warning_frame_sample.jpg`.
- **Result**:
  - Across 200 frames (8 seconds of traffic), raw detections produced 283 frame-level risk evaluations.
  - Without debouncing/cooldown, a naive system would have sounded 283 repetitive alarms.
  - The Warning Engine distilled this into **exactly 4 high-value, actionable alerts**:
    1. $t=0.32$s: `CAUTION: Truck from your left -> Truck nearby (closing distance)`
    2. $t=1.44$s: `DANGER: High-Risk TRUCK FROM YOUR LEFT -> Truck closing distance rapidly` (Escalation override)
    3. $t=2.64$s: `CAUTION: Car from your right -> Car nearby (rapid approach)`
    4. $t=6.88$s: `CAUTION: Truck from your left -> Truck nearby (closing distance)`
  - Processing speed: **24.3 FPS** on Apple Silicon M4.
- **Key Concepts Learned**:
  1. *Human-Centered AI & Alert Fatigue*: The effectiveness of a safety system is determined not just by its accuracy, but by rider trust. Excessive nuisance alerts cause riders to ignore or mute alarms.
  2. *Debounce vs. Latency Trade-off*: Debouncing over 3 frames adds only $0.12$s of latency (imperceptible to humans) while eliminating 95%+ of single-frame detection noise.
  3. *Cooldown with Escalation*: Silencing repeat alerts is safe only if an escalation bypass exists for rapid increases in danger.
- **Next Step**: Milestone 9 — Backend API. Wrap the detection, tracking, risk, and warning engines in a lightweight, high-performance FastAPI service ready for iOS integration.

## Experiment 09: Milestone 9 - Backend API (FastAPI REST & WebSocket Service)
- **Objective**: Expose the computer vision and warning intelligence over high-performance asynchronous HTTP and WebSocket endpoints to enable cross-platform clients (iOS app, web dashboard, external dashcam controllers).
- **Dataset**: `outputs/sample_frame_00000.jpg` and live frame streams.
- **Method**:
  - Implemented [`generate_all_sound_assets`](file:///Users/apple/Desktop/RoadwatchAI/src/generate_sound_assets.py): Synthesized open-source, royalty-free 16-bit PCM WAV chimes (`assets/sounds/chime_caution.wav` and `assets/sounds/chime_critical.wav`) using pure standard library Python.
  - Implemented [`app`](file:///Users/apple/Desktop/RoadwatchAI/src/api.py) with FastAPI and Uvicorn:
    - `GET /health`: System telemetry, device acceleration (MPS), frames processed, alert totals.
    - `POST /api/v1/analyze/frame`: Multipart image upload endpoint returning tracked vehicles, risk scores, and active alerts.
    - `GET /api/v1/alerts/active`: Current rider HUD alert status.
    - `GET /api/v1/alerts/history`: Complete session alert history.
    - `GET /api/v1/sounds/{name}`: Serves open-source .WAV alert chimes to mobile clients.
    - `WebSocket /api/v1/ws/alerts`: Push channel streaming live alerts at 10Hz without polling.
- **Result**:
  - All endpoints verified via `fastapi.testclient.TestClient`.
  - Image frame ingestion, full vision pipeline execution, and JSON payload serialization completed with 200 OK responses.
  - Open-source audio cues successfully served over HTTP.
- **Key Concepts Learned**:
  1. *Decoupled Client-Server Architecture*: The AI engine operates as an independent backend service. Frontend clients (iOS, Android, Web) focus entirely on user experience and native hardware rendering.
  2. *WebSockets vs. HTTP Polling*: On a moving two-wheeler, polling HTTP endpoints 25 times per second creates massive battery and network overhead; WebSockets maintain a persistent low-latency duplex connection, pushing data only when events occur.
- **Next Step**: Milestone 10 — iOS Client Prototype. Build the RoadWatch iOS interface in SwiftUI with audio/visual/haptic alerts connecting to this FastAPI service.

## Experiment 10: Milestone 10 - iOS Client Prototype (SwiftUI Cockpit & Audio Warnings)
- **Objective**: Build a native, driver-centric iOS client in SwiftUI designed specifically for two-wheeler cockpit mountings. The app must render high-contrast visual hazard cards, directional lane radar, session alert history, and trigger local open-source audio chimes and haptic feedback.
- **Components Developed**:
  - `ios/RoadWatch/Package.swift`: Swift Package Manager configuration supporting iOS 16+ and macOS 13+.
  - [`RiderAlertDTO`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Sources/Models/AlertModels.swift): Swift Codable models mapping API JSON alerts, direction, risk scores, and severity.
  - [`AudioAlertService`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Sources/Services/AudioAlertService.swift): AVAudioPlayer audio manager playing bundled open-source `.wav` sounds (`chime_caution.wav`, `chime_critical.wav`) and invoking `UINotificationFeedbackGenerator` haptics on physical devices.
  - [`RoadWatchAPIService`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Sources/Services/RoadWatchAPIService.swift): Async/await networking service connecting to FastAPI `/health` and `/api/v1/alerts/active`, plus built-in demo simulation scenarios.
  - [`CockpitViewModel`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Sources/ViewModels/CockpitViewModel.swift): State manager supporting dual operating modes: live backend polling and offline interactive simulation mode.
  - [`AlertBannerCard`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Sources/Views/AlertBannerCard.swift): Glanceable high-contrast HUD banner displaying Safe, Caution (Amber), and Critical Emergency (Red) states with directional arrows and risk scores.
  - [`CockpitDashboardView`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Sources/Views/CockpitDashboardView.swift): Complete dashboard UI with 3-lane radar strip, interactive simulation controls, sound triggers, and recent session alert feed.
  - [`RoadWatchTests`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Tests/RoadWatchTests/RoadWatchTests.swift): XCTest suite testing DTO decoding, API responses, and simulation scenario flow.
- **Result**:
  - Swift compilation: Successful with Swift 6 / Xcode 16 toolchain.
  - Test suite: 4 tests executed and passed (`100% pass rate`, 0 failures).
  - Audio playback verified: Bundled `.wav` sound assets packaged via SPM `.process("Resources")`.
  - Visual hierarchy: Glanceable for riders at 40-60 km/h with high contrast color coding and minimal text distractions.
- **Key Concepts Learned**:
  1. *Rider Cognitive Load*: Two-wheeler riders have fractions of a second to glance at their handlebar mount. UI must prioritize color, iconography, and spatial lane cues over dense text tables.
  2. *Multimodal Warning Redundancy*: In Indian city traffic, ambient noise (horns, engines) can drown out phone speakers. Combining loud open-source chimes with visual banners and physical vibration ensures critical warnings are never missed.
- **Next Step**: Milestone 11 — End-to-End Prototype. Integrate video replay with the FastAPI service and connect the iOS client for live streaming inference and alert generation.

## Experiment 11: Milestone 11 - End-to-End Pipeline Integration & Live Stream Runner
- **Objective**: Unify the entire perception, kinematic feature extraction, risk scoring, warning debouncing, FastAPI backend state, and native iOS client into a closed-loop real-time streaming pipeline.
- **Components Developed**:
  - [`LiveStreamRunner`](file:///Users/apple/Desktop/RoadwatchAI/src/stream_runner.py): Simulates a dashcam hardware camera stream. Operates in two modes:
    1. *Paced Mode*: Enforces true real-time 25.0 FPS cadence for realistic rider hardware simulation.
    2. *Max Throughput Mode*: Delivers benchmark performance (up to 27.8 FPS average on Apple Silicon M4 with full video rendering).
  - Synchronized Backend State: Feeds detections and warnings directly into [`pipeline_state`](file:///Users/apple/Desktop/RoadwatchAI/src/api.py), exposing `/health`, `/api/v1/alerts/active`, and `/api/v1/alerts/history` to client queries.
  - Native Swift WebSocket Integration: Extended [`RoadWatchAPIService`](file:///Users/apple/Desktop/RoadwatchAI/ios/RoadWatch/Sources/Services/RoadWatchAPIService.swift) with `createWebSocketTask` and message payload parsing for sub-50ms push notification delivery without polling overhead.
  - Automated Integration Tests: Created [`tests/test_end_to_end.py`](file:///Users/apple/Desktop/RoadwatchAI/tests/test_end_to_end.py) validating the complete data path from video input to API state and audio cue serving.
- **Result**:
  - Successfully processed 200 frames of real commute footage at **27.8 FPS** average speed on Apple Silicon M4.
  - Generated annotated cockpit video feed at `outputs/stream_runner_output.mp4`.
  - Accurately triggered and synchronized 4 high-value alerts across the session:
    1. $t=0.32$s: `CAUTION: Truck from your left` (Left lane, 42 pts)
    2. $t=1.44$s: `CRITICAL: High-Risk Truck from your left` (Escalation override, 78 pts)
    3. $t=2.64$s: `CAUTION: Car from your right` (Right lane, 48 pts)
    4. $t=6.88$s: `CAUTION: Truck from your left` (Left lane, 44 pts)
  - All 33 Python unit & integration tests and all 4 Swift unit tests passed with 0 failures (`100% pass rate`).
- **Key Concepts Learned**:
  1. *Closed-Loop Synchronization*: When processing video streams asynchronously, keeping backend pipeline state thread-safe and coupled with WebSocket pushes ensures client latency matches physical vehicle motion.
  2. *Adaptive Pacing*: Benchmarking requires unconstrained hardware execution, but client testing requires strict 25 FPS pacing to mirror realistic rider experience.
- **Next Step**: Milestone 12 — Real-Time Stream Performance Profiling. Measure per-stage latency breakdowns (inference, tracking, feature calculation, risk modeling, network serialization, client render) and identify potential bottlenecks.

## Experiment 12: Milestone 12 - Real-Time Stream Performance Profiling & Latency Breakdown
- **Objective**: Rigorously profile stage-by-stage execution latencies, identify computational bottlenecks, and measure percentile distributions ($p50, p95, p99$), frame jitter, and throughput headroom under peak tracking workloads on Apple Silicon M4 Metal Performance Shaders (MPS).
- **Components Developed**:
  - [`StreamProfiler`](file:///Users/apple/Desktop/RoadwatchAI/src/profiler.py): Precision profiler tracking per-frame timestamps across 9 granular pipeline stages:
    1. Frame Ingestion & Decode (`cv2.VideoCapture`)
    2. YOLOv8 Detection + ByteTrack Multi-Object Association
    3. Road Geometry & Spatial Positioning (horizon/vanishing point projection)
    4. Kinematic Feature Extraction ($v_{\text{long}}, v_{\text{lat}}, \text{TTC}$, approach rate)
    5. Hybrid Risk Engine (spatial + kinematic + divergence gating)
    6. Lane Corridor & Departure Warning (LDW Hough perspective math)
    7. Warning Engine (debouncing, cooldown, prioritization)
    8. Cockpit HUD Rendering (`cv2.rectangle`, `cv2.putText`, status banners)
    9. Frame Encoding & Disk Serialization (`cv2.VideoWriter`)
  - Warmup Routine: Eliminates Apple MPS shader compilation spikes on initial frames to ensure accurate jitter estimation.
  - QuickTime-Native Benchmark Output: Automatically encodes test output to H.264 (`yuv420p`) + Stereo AAC (`44.1 kHz`) with synchronized automotive alert beeps.
  - Automated Unit Testing: [`TestStreamProfiler`](file:///Users/apple/Desktop/RoadwatchAI/tests/test_profiler.py) verifying stage timing calculations, percentile accuracy, and JSON/Markdown export generation.
- **Profiling Benchmark Results (220 frames, starting at frame #1050)**:
  - **Hardware Backend**: Apple Silicon (M4) via Metal Performance Shaders (MPS)
  - **Resolution**: 854 x 480
  - **Target FPS Budget**: 25.0 FPS (Budget: 40.00 ms/frame)
  - **Achieved Throughput**: **41.62 FPS** (1.66x faster than real-time video playback)
  - **Latency Percentiles**:
    - Median ($p50$): **23.58 ms**
    - Tail ($p95$): **30.43 ms**
    - Peak Spike ($p99$): **36.61 ms**
    - Frame Jitter ($\sigma$): **7.18 ms**
  - **Real-Time Headroom**: **+9.57 ms** buffer at the 95th percentile relative to the 40 ms budget.
  - **Stage-by-Stage Breakdown**:
    | Stage | Mean (ms) | p50 (ms) | p95 (ms) | % of Pipeline | Bottleneck Analysis |
    |---|---|---|---|---|---|
    | `1_frame_decode` | 0.23 ms | 0.22 ms | 0.30 ms | 0.9% | Negligible CPU read overhead |
    | `2_yolo_bytetrack` | 19.19 ms | 18.98 ms | 24.26 ms | **79.9%** | **Primary Computational Load** (Neural weights on MPS) |
    | `3_road_geometry` | 0.01 ms | 0.01 ms | 0.02 ms | <0.1% | Instant vectorized spatial projection |
    | `4_kinematic_features`| 0.01 ms | 0.01 ms | 0.02 ms | 0.1% | Highly optimized rolling window math |
    | `5_risk_engine` | 0.01 ms | 0.00 ms | 0.01 ms | <0.1% | Zero-latency rule & regression scoring |
    | `6_lane_corridor_ldw` | 2.68 ms | 2.67 ms | 3.13 ms | **11.1%** | Secondary load (Canny + Hough transforms) |
    | `7_warning_engine` | 0.00 ms | 0.00 ms | 0.01 ms | <0.1% | Instant state machine evaluations |
    | `8_cockpit_hud_render`| 0.05 ms | 0.04 ms | 0.06 ms | 0.2% | Ultra-lightweight OpenCV rasterization |
    | `9_frame_encode_write`| 1.85 ms | 1.81 ms | 2.14 ms | 7.7% | Video frame serialization |
- **Key Concepts Learned**:
  1. *Perception Dominance*: 79.9% of total pipeline latency is consumed by YOLOv8 + ByteTrack. The mathematical feature extraction, risk scoring, and warning state machine together consume less than 0.3% of the frame budget.
  2. *Predictable Real-Time Execution*: With a $p95$ of 30.43 ms against a 40 ms threshold, RoadWatch AI operates safely within real-time limits on Apple Silicon without dropping frames during high-traffic maneuvers.
  3. *Optimization Path for Milestone 13*: To push throughput beyond 60 FPS for high-refresh mobile displays, quantizing YOLO to CoreML Int8 / FP16 on the Apple Neural Engine (ANE) and ROI-cropping for lane detection will yield immediate latency gains.


