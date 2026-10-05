"""
Unit test for Milestone 2: Vehicle Detector (YOLO)
"""

import os
import unittest
import numpy as np
import cv2
from src.vehicle_detector import VehicleDetector, ROAD_CLASSES


class TestVehicleDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use CPU for deterministic lightweight unit tests
        cls.detector = VehicleDetector(conf_threshold=0.25, device="cpu")
        cls.sample_image_path = "outputs/sample_frame_00000.jpg"

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(len(self.detector.target_class_ids), len(ROAD_CLASSES))

    def test_detect_frame_synthetic(self):
        # Test on a dummy black image
        dummy_frame = np.zeros((480, 854, 3), dtype=np.uint8)
        annotated, detections = self.detector.detect_frame(dummy_frame)

        self.assertEqual(annotated.shape, dummy_frame.shape)
        self.assertEqual(annotated.dtype, dummy_frame.dtype)
        self.assertIsInstance(detections, list)

    def test_detect_frame_real_dashcam(self):
        if not os.path.exists(self.sample_image_path):
            self.skipTest(f"Sample image not found at {self.sample_image_path}")

        frame = cv2.imread(self.sample_image_path)
        annotated, detections = self.detector.detect_frame(frame)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertIsInstance(detections, list)

        # Inspect detection structure if any objects were found
        for det in detections:
            self.assertIn("bbox", det)
            self.assertIn("confidence", det)
            self.assertIn("class_name", det)
            self.assertIn("class_id", det)
            self.assertIn("center", det)
            self.assertIn("width", det)
            self.assertIn("height", det)

            x1, y1, x2, y2 = det["bbox"]
            self.assertGreaterEqual(x2, x1)
            self.assertGreaterEqual(y2, y1)
            self.assertGreaterEqual(det["confidence"], 0.25)
            self.assertIn(det["class_id"], ROAD_CLASSES)


if __name__ == "__main__":
    unittest.main()
