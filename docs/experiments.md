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







