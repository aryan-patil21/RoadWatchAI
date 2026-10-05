"""
RoadWatch AI - Milestone 2: Basic Vehicle Detection
---------------------------------------------------
This module provides vehicle and road-object detection using a pretrained YOLO model:
- Loads a pretrained YOLOv8 model (yolov8n.pt)
- Filters for relevant road classes (car, motorcycle, bus, truck, bicycle, person)
- Performs inference on individual frames or entire video streams
- Draws bounding boxes, class labels, and confidence scores
- Logs detection metrics and processing speed (FPS)
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO dataset class indices relevant to road-safety analysis
ROAD_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Color palette for bounding boxes (BGR format for OpenCV)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 200, 0),         # Green
    "motorcycle": (0, 140, 255), # Orange
    "bus": (255, 100, 0),       # Blue-ish
    "truck": (200, 50, 0),      # Darker Blue
    "bicycle": (200, 200, 0),   # Cyan
    "person": (0, 0, 230),      # Red
}
DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Yellow fallback


class VehicleDetector:
    """
    Wraps a pretrained YOLO model to detect road vehicles and pedestrians.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        """
        Initializes the vehicle detector.

        Args:
            model_path: Path or identifier for pretrained YOLO weights (default 'yolov8n.pt').
            conf_threshold: Minimum confidence threshold [0.0 - 1.0] to accept a detection.
            device: 'mps' for Apple Silicon GPU, 'cpu', or None for auto-selection.
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.target_class_ids = list(ROAD_CLASSES.keys())

        print(f"[VehicleDetector] Loading model '{model_path}' on device: {self.device.upper()}")
        self.model = YOLO(model_path)
        print(f"[VehicleDetector] Target classes: {list(ROAD_CLASSES.values())}")
        print(f"[VehicleDetector] Confidence threshold: {self.conf_threshold}")

    def detect_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs object detection on a single video frame.

        Args:
            frame: BGR NumPy array representing one video frame.

        Returns:
            Tuple of:
            - annotated_frame: Copy of frame with drawn bounding boxes and labels.
            - detections: List of dicts containing bbox coordinates, label, confidence, and class_id.
        """
        annotated_frame = frame.copy()
        detections: List[Dict[str, Any]] = []

        # Run model inference on the frame
        results = self.model.predict(
            source=frame,
            classes=self.target_class_ids,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        if not results:
            return annotated_frame, detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return annotated_frame, detections

        # Extract detected boxes
        xyxy_coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()  # confidence scores
        class_ids = boxes.cls.cpu().numpy().astype(int)  # class indices

        for i in range(len(xyxy_coords)):
            x1, y1, x2, y2 = [int(v) for v in xyxy_coords[i]]
            conf = float(confidences[i])
            cls_id = int(class_ids[i])
            cls_name = ROAD_CLASSES.get(cls_id, self.model.names.get(cls_id, f"class_{cls_id}"))

            detection_record = {
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": cls_name,
                "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                "width": x2 - x1,
                "height": y2 - y1,
            }
            detections.append(detection_record)

            # Draw bounding box and text banner on annotated_frame
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"{cls_name.upper()} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            # Background banner for readability
            banner_y1 = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(
                annotated_frame,
                (x1, banner_y1),
                (x1 + text_w + 6, y1),
                color,
                -1,
            )
            # Text inside banner
            cv2.putText(
                annotated_frame,
                label_text,
                (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text for high contrast
                1,
                cv2.LINE_AA,
            )

        return annotated_frame, detections

    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes a video file frame-by-frame, annotating detected vehicles,
        and saves an annotated output video.

        Args:
            input_path: Path to source dashcam video.
            output_path: Destination path for annotated MP4 video.
            max_frames: Optional maximum number of frames to process (useful for quick experiments).

        Returns:
            Dictionary of processing metrics (total frames, elapsed time, average FPS, counts).
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found at: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_available_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        limit_frames = min(total_available_frames, max_frames) if max_frames else total_available_frames

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[VehicleDetector] Processing '{input_path}' ({limit_frames} frames) -> '{output_path}'")

        frame_count = 0
        total_detections = 0
        class_histogram: Dict[str, int] = {name: 0 for name in ROAD_CLASSES.values()}

        start_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success or (max_frames and frame_count >= max_frames):
                break

            annotated_frame, detections = self.detect_frame(frame)

            # Record stats
            frame_count += 1
            total_detections += len(detections)
            for det in detections:
                c_name = det["class_name"]
                class_histogram[c_name] = class_histogram.get(c_name, 0) + 1

            # Overlay frame counter and processing HUD
            hud_text = f"Frame {frame_count}/{limit_frames} | Detected Objects: {len(detections)}"
            cv2.putText(
                annotated_frame,
                hud_text,
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            out.write(annotated_frame)

            if frame_count % 50 == 0 or frame_count == limit_frames:
                elapsed = time.time() - start_time
                curr_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {frame_count}/{limit_frames} frames ({curr_fps:.1f} FPS)...")

        elapsed_total = time.time() - start_time
        avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0

        cap.release()
        out.release()

        metrics = {
            "input_path": input_path,
            "output_path": output_path,
            "processed_frames": frame_count,
            "elapsed_seconds": round(elapsed_total, 2),
            "average_fps": round(avg_fps, 2),
            "total_detections": total_detections,
            "detections_per_frame": round(total_detections / frame_count, 2) if frame_count > 0 else 0,
            "class_histogram": class_histogram,
        }

        print("\n" + "=" * 60)
        print(" RoadWatch AI - Vehicle Detection Summary")
        print("=" * 60)
        print(f"Frames Processed : {frame_count}")
        print(f"Elapsed Time     : {elapsed_total:.2f} s")
        print(f"Inference Speed  : {avg_fps:.1f} FPS (Metal/MPS Accelerated)")
        print(f"Total Detections : {total_detections}")
        print(f"Class Breakdown  : {class_histogram}")
        print(f"Output Video     : {output_path}")
        print("=" * 60)

        return metrics


if __name__ == "__main__":
    video_input = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_dashcam.mp4"
    video_output = "outputs/annotated_detection_sample.mp4"

    detector = VehicleDetector(conf_threshold=0.25)

    # Process first 150 frames (6 seconds at 25 FPS) for quick visual inspection
    sample_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    detector.process_video(video_input, video_output, max_frames=sample_limit)
