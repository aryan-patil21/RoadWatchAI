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

