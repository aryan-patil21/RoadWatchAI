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


