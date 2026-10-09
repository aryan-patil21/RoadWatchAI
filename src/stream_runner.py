"""
RoadWatch AI - Milestone 11: Real-Time End-to-End Stream Runner
--------------------------------------------------------------
Executes the live closed-loop pipeline:
  Video Stream / Camera Feed -> Frame Ingestion -> YOLO Detection
  -> ByteTrack -> Spatial Geometry -> Kinematic Features -> Risk Engine
  -> Human-Centered Warning Engine -> FastAPI Shared State & WebSocket Broadcast.

Can run in two modes:
1. PACE_MODE (default): Paces frames to exactly 25.0 FPS to simulate a real live dashcam feed.
2. MAX_SPEED_MODE: Processes frames at maximum Apple Silicon hardware throughput.
"""

import os
import sys
import time
import argparse
import threading
from typing import Optional, Dict, Any, List, Tuple
import cv2
import requests
import subprocess

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vehicle_tracker import VehicleTracker
from src.road_geometry import RoadGeometryAnalyzer
from src.feature_extractor import BehaviouralFeatureExtractor
from src.risk_engine import RiskEngine
from src.warning_engine import WarningEngine, RiderAlert
from src.lane_detector import LaneDepartureDetector
from src.api import pipeline_state


class LiveStreamRunner:
    """
    Simulates a live dashcam hardware stream connected directly to the RoadWatch AI engine.
    Updates the shared backend state, sounds live warning chimes, detects lane departures,
    and renders continuous cockpit HUD telemetry.
    """

    def __init__(
        self,
        video_path: str = "data/raw/sample_dashcam.mp4",
        target_fps: float = 25.0,
        conf_threshold: float = 0.25,
        debounce_frames: int = 3,
        cooldown_seconds: float = 3.5,
        pace_stream: bool = True,
        save_annotated_stream: bool = True,
        output_stream_path: str = "outputs/stream_runner_output.mp4",
        enable_sound_chimes: bool = True,
        enable_ldw: bool = True,
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0.04
        self.pace_stream = pace_stream
        self.save_annotated_stream = save_annotated_stream
        self.output_stream_path = output_stream_path
        self.enable_sound_chimes = enable_sound_chimes
        self.enable_ldw = enable_ldw

        # Vision, Lane & Risk Modules
        self.tracker = VehicleTracker(conf_threshold=conf_threshold)
        self.geometry = RoadGeometryAnalyzer()
        self.lane_detector = LaneDepartureDetector() if enable_ldw else None
        self.feature_extractor: Optional[BehaviouralFeatureExtractor] = None
        self.risk_engine = RiskEngine()
        self.warning_engine = WarningEngine(
            debounce_frames=debounce_frames,
            cooldown_seconds=cooldown_seconds,
        )

        self.alert_audio_events: List[Tuple[float, str]] = []
        self.running = False
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "total_alerts": 0,
            "caution_alerts": 0,
            "critical_alerts": 0,
            "average_fps": 0.0,
            "active_vehicles_peak": 0,
        }

    def run(self, max_frames: Optional[int] = None, start_frame: int = 0) -> Dict[str, Any]:
        """
        Executes stream loop, updating FastAPI pipeline_state in real time.
        """
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found at '{self.video_path}'")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video capture for '{self.video_path}'")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        remaining_frames = total_video_frames - start_frame
        limit_frames = min(remaining_frames, max_frames) if max_frames else remaining_frames

        # Configure geometry and feature extractor bounds
        self.geometry.frame_width = width
        self.geometry.frame_height = height
        self.geometry.ego_min_x = 0.35 * width
        self.geometry.ego_max_x = 0.65 * width

        self.feature_extractor = BehaviouralFeatureExtractor(
            fps=self.target_fps,
            frame_width=width,
            frame_height=height,
            window_size=8,
        )

        out_writer = None
        if self.save_annotated_stream:
            os.makedirs(os.path.dirname(self.output_stream_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(self.output_stream_path, fourcc, video_fps, (width, height))

        print("=" * 66)
        print("  RoadWatch AI - Live End-to-End Stream Runner (Milestone 11)")
        print("=" * 66)
        print(f" Source Video       : {self.video_path}")
        print(f" Frame Resolution   : {width} x {height}")
        print(f" Stream Rate        : {self.target_fps} FPS ({'Paced real-time' if self.pace_stream else 'Max throughput'})")
        print(f" Start Frame        : {start_frame}")
        print(f" Total Frames       : {limit_frames}")
        print(f" Target Device      : {self.tracker.device.upper()}")
        print("-" * 66)

        self.running = True
        frame_idx = start_frame
        processed_count = 0
        overall_start = time.time()

        try:
            while cap.isOpened() and self.running:
                loop_start = time.time()
                success, frame = cap.read()
                if not success or (max_frames and processed_count >= max_frames):
                    break

                frame_idx += 1
                processed_count += 1
                timestamp_sec = round(frame_idx / self.target_fps, 2)

                # 1. Multi-Object Tracking (YOLOv8 + ByteTrack)
                annotated_frame, tracks = self.tracker.track_frame(frame, persist=True)

                vehicle_evaluations = []
                response_vehicles = []

                for t in tracks:
                    track_id = t["track_id"]
                    if track_id is None:
                        continue

                    # 2. Road Geometry & Spatial Positioning
                    spatial_data = self.geometry.analyze_vehicle(frame_idx, t)
                    proximity = spatial_data["proximity_score"]

                    # 3. Kinematic Feature Extraction
                    features = self.feature_extractor.extract_features(
                        frame_idx=frame_idx,
                        vehicle_id=track_id,
                        class_name=t["class_name"],
                        bbox=t["bbox"],
                        proximity_score=proximity,
                    )

                    # 4. Hybrid Risk Estimation
                    risk_eval = self.risk_engine.evaluate_risk(features)
                    combined = {**features, **risk_eval, "lateral_zone": spatial_data["lateral_zone"]}
                    vehicle_evaluations.append(combined)

                    response_vehicles.append({
                        "vehicle_id": track_id,
                        "class_name": t["class_name"],
                        "confidence": t["confidence"],
                        "bbox": t["bbox"],
                        "center": t["center"],
                        "lateral_zone": spatial_data["lateral_zone"],
                        "proximity_band": spatial_data["proximity_band"],
                        "proximity_score": proximity,
                        "movement": spatial_data["longitudinal_movement"],
                        "risk_score": risk_eval["risk_score"],
                        "risk_level": risk_eval["risk_level"],
                        "primary_factor": risk_eval["primary_factor"],
                    })

                # 5. Lane Departure Warning (LDW) Analysis
                ldw_info = {"status": "NORMAL", "offset_px": 0.0}
                if self.lane_detector:
                    ldw_info = self.lane_detector.process_frame(frame)
                    self.lane_detector.draw_lanes(annotated_frame, ldw_info)

                # 6. Warning Engine (Debouncing, Cooldown, Prioritization)
                new_alert = self.warning_engine.process_frame(frame_idx, timestamp_sec, vehicle_evaluations)
                active_alert = self.warning_engine.active_display_alert

                # 7. Synchronize with FastAPI backend shared pipeline state
                pipeline_state.frame_counter = frame_idx
                if new_alert:
                    alert_dict = new_alert.to_dict()
                    pipeline_state.session_alerts.append(alert_dict)
                    self.stats["total_alerts"] += 1
                    if new_alert.level.value == "CRITICAL":
                        self.stats["critical_alerts"] += 1
                    else:
                        self.stats["caution_alerts"] += 1

                    # Log relative seconds from start of output clip for audio muxing
                    clip_relative_sec = (processed_count - 1) / self.target_fps
                    self.alert_audio_events.append((clip_relative_sec, new_alert.level.value))

                    # Trigger open-source sound chime live on Mac speakers in background thread
                    if self.enable_sound_chimes:
                        self._play_sound_chime(new_alert.level.value)

                pipeline_state.warning_engine = self.warning_engine

                if len(tracks) > self.stats["active_vehicles_peak"]:
                    self.stats["active_vehicles_peak"] = len(tracks)

                # Render Continuous Cockpit HUD on output video stream
                if out_writer:
                    self._render_hud(annotated_frame, frame_idx, timestamp_sec, active_alert, len(tracks), ldw_info)
                    out_writer.write(annotated_frame)

                # Print terminal status update every 25 frames (1 second of video)
                if frame_idx % 25 == 0 or new_alert:
                    elapsed = time.time() - overall_start
                    fps_current = frame_idx / elapsed if elapsed > 0 else 0
                    hud_status = "NORMAL" if not active_alert else f"{active_alert.level.value} ({active_alert.direction.upper()})"
                    ldw_tag = f" | LDW: {ldw_info['status']}" if ldw_info['status'] != "NORMAL" else ""
                    alert_tag = f" -> FIRED: {new_alert.title}" if new_alert else ""
                    print(
                        f"Frame {frame_idx:04d}/{limit_frames:04d} | "
                        f"Time: {timestamp_sec:5.2f}s | "
                        f"Vehicles: {len(tracks):2d} | "
                        f"HUD: {hud_status:12s}{ldw_tag} | "
                        f"Speed: {fps_current:4.1f} FPS{alert_tag}"
                    )

                # Pace stream if real-time pacing is enabled
                if self.pace_stream:
                    loop_time = time.time() - loop_start
                    sleep_time = self.frame_interval - loop_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            self.running = False

        total_elapsed = time.time() - overall_start
        self.stats["processed_frames"] = frame_idx
        self.stats["total_frames"] = limit_frames
        self.stats["average_fps"] = round(processed_count / total_elapsed, 2) if total_elapsed > 0 else 0

        # Mux synchronized audio track directly into the video file so it plays sound anywhere
        if self.save_annotated_stream and os.path.exists(self.output_stream_path):
            self._mux_audio_into_video(total_duration_sec=processed_count / self.target_fps)

        print("-" * 66)
        print("  Stream Runner Execution Completed")
        print("-" * 66)
        print(f" Processed Frames : {processed_count} / {limit_frames}")
        print(f" Processing Time  : {total_elapsed:.2f} seconds")
        print(f" Average Speed    : {self.stats['average_fps']} FPS")
        print(f" Peak Vehicles    : {self.stats['active_vehicles_peak']} tracked simultaneously")
        print(f" Fired Warnings   : {self.stats['total_alerts']} ({self.stats['caution_alerts']} Caution, {self.stats['critical_alerts']} Critical)")
        if self.save_annotated_stream:
            print(f" Output Video (with Audio Track): {self.output_stream_path}")
        print("=" * 66)

        return self.stats

    def _mux_audio_into_video(self, total_duration_sec: float) -> None:
        """Constructs an exact-duration audio track with alert beeps and muxes into MP4."""
        import wave, struct
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        sample_rate = 44100
        total_samples = int(sample_rate * total_duration_sec)
        audio_buffer = [0.0] * max(total_samples, 44100)

        # Load raw wav samples for caution and critical beeps
        def load_wav_samples(path):
            if not os.path.exists(path):
                return []
            with wave.open(path, "r") as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                ints = struct.unpack(f"<{n_frames}h", raw)
                return [val / 32767.0 for val in ints]

        caution_samples = load_wav_samples("assets/sounds/chime_caution.wav")
        critical_samples = load_wav_samples("assets/sounds/chime_critical.wav")

        # Overlay each warning beep at its exact timestamp
        for event_time_sec, alert_level in self.alert_audio_events:
            start_idx = int(event_time_sec * sample_rate)
            beep_samples = critical_samples if alert_level == "CRITICAL" else caution_samples
            for i, s in enumerate(beep_samples):
                pos = start_idx + i
                if pos < len(audio_buffer):
                    audio_buffer[pos] = max(-1.0, min(1.0, audio_buffer[pos] + s))

        # Write composite WAV file
        temp_wav_path = self.output_stream_path.replace(".mp4", "_audio.wav")
        with wave.open(temp_wav_path, "w") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)
            out_wf.setframerate(sample_rate)
            raw_out = bytearray()
            for s in audio_buffer:
                val = int(max(-1.0, min(1.0, s)) * 32767)
                raw_out.extend(struct.pack("<h", val))
            out_wf.writeframes(raw_out)

        # Mux audio and video using ffmpeg
        temp_mux_path = self.output_stream_path.replace(".mp4", "_muxed.mp4")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", self.output_stream_path,
            "-i", temp_wav_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            temp_mux_path,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(temp_mux_path):
                os.replace(temp_mux_path, self.output_stream_path)
                print(f"✓ Embedded synchronized audio track directly into '{self.output_stream_path}'")
        except Exception as e:
            print(f"[Warning] Audio muxing skipped: {e}")
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

    def _play_sound_chime(self, level: str) -> None:
        """Plays open-source sound chime asynchronously without blocking vision loop."""
        sound_file = "assets/sounds/chime_critical.wav" if level == "CRITICAL" else "assets/sounds/chime_caution.wav"
        if os.path.exists(sound_file):
            def _play():
                try:
                    subprocess.run(["/usr/bin/afplay", sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            threading.Thread(target=_play, daemon=True).start()

    def _render_hud(
        self,
        frame: cv2.Mat,
        frame_idx: int,
        timestamp_sec: float,
        alert: Optional[RiderAlert],
        vehicle_count: int,
        ldw_info: Dict[str, Any],
    ) -> None:
        """Renders continuous, high-visibility Cockpit banner and lane radar on the frame."""
        h, w = frame.shape[:2]

        # 1. Continuous Safety Status Banner at top (Green / Amber / Red)
        if alert and alert.level.value == "CRITICAL":
            # 🔴 Critical Emergency Alert (Red)
            banner_color = (0, 0, 205)
            status_text = f"[EMERGENCY] {alert.title.upper()} ({int(alert.risk_score)} pts)"
            sub_text = f"ACTION: {alert.suggested_action}"
            cv2.rectangle(frame, (0, 0), (w, 52), banner_color, -1)
            cv2.putText(frame, status_text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub_text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        elif alert and alert.level.value == "CAUTION":
            # 🟡 Caution Hazard (Amber)
            banner_color = (0, 140, 255)
            status_text = f"[CAUTION] {alert.title.upper()} ({int(alert.risk_score)} pts)"
            sub_text = f"ACTION: {alert.suggested_action}"
            cv2.rectangle(frame, (0, 0), (w, 52), banner_color, -1)
            cv2.putText(frame, status_text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub_text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        else:
            # 🟢 Normal Safe Conditions (Clean Green)
            banner_color = (25, 90, 25)
            status_text = "ROAD CONDITIONS NORMAL | ZERO THREATS DETECTED"
            sub_text = f"FRAME: {frame_idx:04d} | TIME: {timestamp_sec:4.1f}s | ACTIVE TARGETS: {vehicle_count}"
            cv2.rectangle(frame, (0, 0), (w, 46), banner_color, -1)
            cv2.putText(frame, status_text, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (140, 255, 140), 2, cv2.LINE_AA)
            cv2.putText(frame, sub_text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 200), 1, cv2.LINE_AA)

        # 2. Lane Departure Warning (LDW) Banner
        ldw_status = ldw_info.get("status", "NORMAL")
        if ldw_status != "NORMAL":
            ldw_color = (0, 140, 255)
            direction_str = "LEFT" if ldw_status == "DRIFT_LEFT" else "RIGHT"
            cv2.rectangle(frame, (w // 2 - 180, 56), (w // 2 + 180, 86), ldw_color, -1)
            cv2.putText(
                frame,
                f"LANE DRIFT: VEERING {direction_str}",
                (w // 2 - 160, 77),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # 3. Cockpit Spatial Direction Pill (Only visible when a hazard has a clear direction)
        if alert and alert.direction:
            dir_text = f"< HAZARD FROM {alert.direction.upper()} >"
            pill_w = 260
            pill_x = (w - pill_w) // 2
            pill_y = h - 34
            pill_bg = (0, 0, 200) if alert.level.value == "CRITICAL" else (0, 140, 255)
            cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + 26), pill_bg, -1)
            cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + 26), (255, 255, 255), 1)
            cv2.putText(
                frame,
                dir_text,
                (pill_x + 22, pill_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )



def main():
    parser = argparse.ArgumentParser(description="RoadWatch AI Live End-to-End Stream Runner")
    parser.add_argument("--video", type=str, default="data/raw/sample_dashcam.mp4", help="Video path")
    parser.add_argument("--frames", type=int, default=200, help="Max frames to run")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from")
    parser.add_argument("--no-pace", action="store_true", help="Run at max hardware speed instead of 25 FPS")
    args = parser.parse_args()

    runner = LiveStreamRunner(
        video_path=args.video,
        pace_stream=not args.no_pace,
        save_annotated_stream=True,
    )
    runner.run(max_frames=args.frames, start_frame=args.start_frame)


if __name__ == "__main__":
    main()
