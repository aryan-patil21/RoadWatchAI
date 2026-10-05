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




