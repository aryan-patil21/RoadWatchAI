"""
RoadWatch AI - Milestone 12: Real-Time Stream Performance Profiler
------------------------------------------------------------------
Profiles stage-by-stage latency percentiles (p50, p95, p99), frame jitter,
hardware resource consumption (MPS / CPU), and throughput bottlenecks across:
1. Frame Decode & Ingestion
2. YOLOv8 Detection & ByteTrack Multi-Object Tracking
3. Road Geometry & Spatial Positioning
4. Kinematic Feature Extraction
5. Hybrid Risk Modeling
6. Lane Departure Corridor (LDW)
7. Warning Engine (Debouncing, Prioritization & Cooldown)
8. Cockpit HUD Rendering
9. Video Encoding / Disk Write
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vehicle_tracker import VehicleTracker
from src.road_geometry import RoadGeometryAnalyzer
from src.feature_extractor import BehaviouralFeatureExtractor
from src.risk_engine import RiskEngine
from src.warning_engine import WarningEngine, RiderAlert
from src.lane_detector import LaneDepartureDetector


@dataclass
class StageLatency:
    """Latency profile of an individual processing stage."""
    name: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    percentage_of_total: float


@dataclass
class ProfilerReport:
    """Consolidated system performance benchmarking report."""
    device: str
    platform_name: str
    total_frames: int
    resolution: Tuple[int, int]
    target_fps: float
    achieved_fps: float
    total_duration_sec: float
    average_frame_latency_ms: float
    p50_frame_latency_ms: float
    p95_frame_latency_ms: float
    p99_frame_latency_ms: float
    max_frame_latency_ms: float
    jitter_ms: float
    real_time_budget_ms: float
    budget_headroom_ms: float
    is_realtime_capable: bool
    stages: List[StageLatency]
    vehicle_tracking_peak: int


class StreamProfiler:
    """Precision profiler for RoadWatch AI pipeline."""

    def __init__(
        self,
        video_path: str = "data/raw/sample_dashcam.mp4",
        target_fps: float = 25.0,
        device: str = "mps",
        save_annotated: bool = True,
        output_dir: str = "outputs",
    ):
        self.video_path = video_path
        self.target_fps = target_fps
        self.device = device
        self.save_annotated = save_annotated
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Stage latency history (in milliseconds)
        self.stage_timings: Dict[str, List[float]] = {
            "1_frame_decode": [],
            "2_yolo_bytetrack": [],
            "3_road_geometry": [],
            "4_kinematic_features": [],
            "5_risk_engine": [],
            "6_lane_corridor_ldw": [],
            "7_warning_engine": [],
            "8_cockpit_hud_render": [],
            "9_frame_encode_write": [],
        }
        self.total_frame_times: List[float] = []

    def run(self, max_frames: int = 220, start_frame: int = 1050) -> ProfilerReport:
        """Executes profiling run across the pipeline."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(start_frame, total_vid_frames - 1))

        # Initialize pipeline modules
        tracker = VehicleTracker(device=self.device, conf_threshold=0.25)
        geometry = RoadGeometryAnalyzer(frame_width=orig_w, frame_height=orig_h)
        feature_extractor = BehaviouralFeatureExtractor(fps=self.target_fps, frame_width=orig_w, frame_height=orig_h)
        risk_engine = RiskEngine()
        warning_engine = WarningEngine(cooldown_seconds=3.0, debounce_frames=2)
        lane_detector = LaneDepartureDetector(frame_width=orig_w, frame_height=orig_h)

        # Video Writer for annotated stream
        out_writer = None
        raw_output_path = os.path.join(self.output_dir, "profiler_benchmark_temp.mp4")
        final_output_path = os.path.join(self.output_dir, "profiler_benchmark_output.mp4")
        if self.save_annotated:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(raw_output_path, fourcc, self.target_fps, (orig_w, orig_h))

        # Warm up MPS / GPU pipeline with dummy frame to isolate JIT graph compilation
        dummy_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for _ in range(3):
            tracker.track_frame(dummy_frame, persist=False)

        peak_vehicles = 0
        frames_processed = 0
        overall_start = time.perf_counter()

        alert_events: List[Tuple[float, str]] = []

        print("=" * 70)
        print("  RoadWatch AI - Real-Time Performance Profiler (Milestone 12)")
        print("=" * 70)
        print(f" Target Device       : {self.device.upper()}")
        print(f" Input Video         : {self.video_path} ({orig_w}x{orig_h})")
        print(f" Frames to Profile   : {max_frames} (starting at frame #{start_frame})")
        print(f" Target Frame Rate   : {self.target_fps:.1f} FPS (Budget: {1000.0/self.target_fps:.2f} ms/frame)")
        print("-" * 70)

        try:
            for f_idx in range(max_frames):
                frame_start = time.perf_counter()

                # Stage 1: Frame Ingestion & Decode
                t0 = time.perf_counter()
                ret, frame = cap.read()
                t1 = time.perf_counter()
                if not ret:
                    break
                self.stage_timings["1_frame_decode"].append((t1 - t0) * 1000.0)

                current_frame_idx = start_frame + f_idx
                timestamp_sec = current_frame_idx / self.target_fps

                # Stage 2: YOLOv8 Detection + ByteTrack Multi-Object Tracking
                t0 = time.perf_counter()
                annotated_frame, tracks = tracker.track_frame(frame, persist=True)
                t1 = time.perf_counter()
                self.stage_timings["2_yolo_bytetrack"].append((t1 - t0) * 1000.0)

                peak_vehicles = max(peak_vehicles, len(tracks))
                vehicle_evaluations = []

                # Stage 3, 4, 5: Road Geometry, Kinematic Features, Risk Engine
                geom_time = 0.0
                feat_time = 0.0
                risk_time = 0.0

                for t in tracks:
                    track_id = t["track_id"]
                    if track_id is None:
                        continue

                    # Stage 3: Road Geometry
                    s_t0 = time.perf_counter()
                    spatial_data = geometry.analyze_vehicle(current_frame_idx, t)
                    proximity = spatial_data["proximity_score"]
                    s_t1 = time.perf_counter()
                    geom_time += (s_t1 - s_t0) * 1000.0

                    # Stage 4: Kinematic Features
                    s_t0 = time.perf_counter()
                    features = feature_extractor.extract_features(
                        frame_idx=current_frame_idx,
                        vehicle_id=track_id,
                        class_name=t["class_name"],
                        bbox=t["bbox"],
                        proximity_score=proximity,
                    )
                    s_t1 = time.perf_counter()
                    feat_time += (s_t1 - s_t0) * 1000.0

                    # Stage 5: Hybrid Risk Engine
                    s_t0 = time.perf_counter()
                    risk_eval = risk_engine.evaluate_risk(features)
                    combined = {**features, **risk_eval, "lateral_zone": spatial_data["lateral_zone"]}
                    vehicle_evaluations.append(combined)
                    s_t1 = time.perf_counter()
                    risk_time += (s_t1 - s_t0) * 1000.0

                self.stage_timings["3_road_geometry"].append(geom_time)
                self.stage_timings["4_kinematic_features"].append(feat_time)
                self.stage_timings["5_risk_engine"].append(risk_time)

                # Stage 6: Lane Corridor & Departure Warning (LDW)
                t0 = time.perf_counter()
                ldw_info = lane_detector.process_frame(frame)
                lane_detector.draw_lanes(annotated_frame, ldw_info)
                t1 = time.perf_counter()
                self.stage_timings["6_lane_corridor_ldw"].append((t1 - t0) * 1000.0)

                # Stage 7: Warning Engine (Debouncing, Prioritization & Cooldown)
                t0 = time.perf_counter()
                new_alert = warning_engine.process_frame(current_frame_idx, timestamp_sec, vehicle_evaluations)
                active_alert = warning_engine.active_display_alert
                t1 = time.perf_counter()
                self.stage_timings["7_warning_engine"].append((t1 - t0) * 1000.0)

                if new_alert:
                    clip_sec = frames_processed / self.target_fps
                    alert_events.append((clip_sec, new_alert.level.value))

                # Stage 8: Cockpit HUD Rendering
                t0 = time.perf_counter()
                self._render_hud(
                    annotated_frame,
                    frame_idx=current_frame_idx,
                    timestamp_sec=timestamp_sec,
                    alert=active_alert,
                    vehicle_count=len(tracks),
                    ldw_info=ldw_info,
                )
                t1 = time.perf_counter()
                self.stage_timings["8_cockpit_hud_render"].append((t1 - t0) * 1000.0)

                # Stage 9: Frame Encode & Write
                t0 = time.perf_counter()
                if out_writer:
                    out_writer.write(annotated_frame)
                t1 = time.perf_counter()
                self.stage_timings["9_frame_encode_write"].append((t1 - t0) * 1000.0)

                # Total Frame Time
                frame_end = time.perf_counter()
                self.total_frame_times.append((frame_end - frame_start) * 1000.0)
                frames_processed += 1

                if frames_processed % 50 == 0:
                    inst_fps = 1000.0 / self.total_frame_times[-1] if self.total_frame_times[-1] > 0 else 0
                    avg_fps_so_far = frames_processed / (time.perf_counter() - overall_start)
                    print(
                        f"  Profiled {frames_processed:03d}/{max_frames:03d} frames | "
                        f"Instant: {self.total_frame_times[-1]:.1f} ms ({inst_fps:4.1f} FPS) | "
                        f"Avg: {avg_fps_so_far:4.1f} FPS"
                    )

        finally:
            cap.release()
            if out_writer:
                out_writer.release()

        overall_duration = time.perf_counter() - overall_start
        achieved_fps = frames_processed / overall_duration if overall_duration > 0 else 0.0

        # Calculate Percentiles & Stage Breakdowns
        total_arr = np.array(self.total_frame_times)
        mean_frame_ms = float(np.mean(total_arr))
        p50_frame_ms = float(np.percentile(total_arr, 50))
        p95_frame_ms = float(np.percentile(total_arr, 95))
        p99_frame_ms = float(np.percentile(total_arr, 99))
        max_frame_ms = float(np.max(total_arr))
        jitter_ms = float(np.std(total_arr))

        budget_ms = 1000.0 / self.target_fps
        headroom_ms = budget_ms - p95_frame_ms
        is_realtime = p95_frame_ms <= budget_ms

        stages_report: List[StageLatency] = []
        for stage_name, timings in self.stage_timings.items():
            if not timings:
                continue
            arr = np.array(timings)
            m_ms = float(np.mean(arr))
            stages_report.append(
                StageLatency(
                    name=stage_name,
                    mean_ms=m_ms,
                    p50_ms=float(np.percentile(arr, 50)),
                    p95_ms=float(np.percentile(arr, 95)),
                    p99_ms=float(np.percentile(arr, 99)),
                    min_ms=float(np.min(arr)),
                    max_ms=float(np.max(arr)),
                    percentage_of_total=round((m_ms / mean_frame_ms) * 100.0, 2) if mean_frame_ms > 0 else 0.0,
                )
            )

        report = ProfilerReport(
            device=self.device.upper(),
            platform_name=f"{platform.system()} {platform.machine()} ({platform.processor() or 'Apple Silicon'})",
            total_frames=frames_processed,
            resolution=(orig_w, orig_h),
            target_fps=self.target_fps,
            achieved_fps=round(achieved_fps, 2),
            total_duration_sec=round(overall_duration, 2),
            average_frame_latency_ms=round(mean_frame_ms, 2),
            p50_frame_latency_ms=round(p50_frame_ms, 2),
            p95_frame_latency_ms=round(p95_frame_ms, 2),
            p99_frame_latency_ms=round(p99_frame_ms, 2),
            max_frame_latency_ms=round(max_frame_ms, 2),
            jitter_ms=round(jitter_ms, 2),
            real_time_budget_ms=round(budget_ms, 2),
            budget_headroom_ms=round(headroom_ms, 2),
            is_realtime_capable=is_realtime,
            stages=stages_report,
            vehicle_tracking_peak=peak_vehicles,
        )

        # Mux Audio into Benchmark Video (Apple QuickTime Native H.264 + Stereo AAC)
        if self.save_annotated and os.path.exists(raw_output_path):
            self._mux_audio_native(
                raw_video_path=raw_output_path,
                final_video_path=final_output_path,
                alert_events=alert_events,
                total_duration_sec=frames_processed / self.target_fps,
            )

        # Save Structured JSON & Markdown Reports
        self._export_reports(report)
        self._print_terminal_summary(report)

        return report

    def _render_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        alert: Any,
        vehicle_count: int,
        ldw_info: Dict[str, Any],
    ) -> None:
        """Renders standard Cockpit safety banner and active warning."""
        h, w = frame.shape[:2]
        if alert and alert.level.value == "CRITICAL":
            cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 205), -1)
            cv2.putText(frame, f"[EMERGENCY] {alert.title.upper()} ({int(alert.risk_score)} pts)", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"ACTION: {alert.suggested_action}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        elif alert and alert.level.value == "CAUTION":
            cv2.rectangle(frame, (0, 0), (w, 52), (0, 140, 255), -1)
            cv2.putText(frame, f"[CAUTION] {alert.title.upper()} ({int(alert.risk_score)} pts)", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"ACTION: {alert.suggested_action}", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (0, 0), (w, 46), (25, 90, 25), -1)
            cv2.putText(frame, "ROAD CONDITIONS NORMAL | ZERO THREATS DETECTED", (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (140, 255, 140), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FRAME: {frame_idx:04d} | TIME: {timestamp_sec:4.1f}s | ACTIVE TARGETS: {vehicle_count}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 240, 200), 1, cv2.LINE_AA)

        # LDW Banner
        ldw_status = ldw_info.get("status", "NORMAL")
        if ldw_status != "NORMAL":
            dir_str = "LEFT" if ldw_status == "DRIFT_LEFT" else "RIGHT"
            cv2.rectangle(frame, (w // 2 - 180, 56), (w // 2 + 180, 86), (0, 140, 255), -1)
            cv2.putText(frame, f"LANE DRIFT: VEERING {dir_str}", (w // 2 - 160, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # Hazard direction pill
        if alert and alert.direction:
            dir_text = f"< HAZARD FROM {alert.direction.upper()} >"
            pill_w = 260
            pill_x = (w - pill_w) // 2
            pill_y = h - 34
            pill_bg = (0, 0, 200) if alert.level.value == "CRITICAL" else (0, 140, 255)
            cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + 26), pill_bg, -1)
            cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + 26), (255, 255, 255), 1)
            cv2.putText(frame, dir_text, (pill_x + 22, pill_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2, cv2.LINE_AA)

    def _mux_audio_native(
        self,
        raw_video_path: str,
        final_video_path: str,
        alert_events: List[Tuple[float, str]],
        total_duration_sec: float,
    ) -> None:
        """Bakes synchronized audio beeps and re-encodes as Apple QuickTime H.264 + Stereo AAC."""
        import struct
        import wave
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        sample_rate = 44100
        total_samples = int(sample_rate * total_duration_sec)
        audio_buffer = [0.0] * max(total_samples, 44100)

        def load_wav(path: str) -> List[float]:
            if not os.path.exists(path):
                return []
            with wave.open(path, "r") as wf:
                n = wf.getnframes()
                raw = wf.readframes(n)
                ints = struct.unpack(f"<{n}h", raw)
                return [val / 32767.0 for val in ints]

        caution_samples = load_wav("assets/sounds/chime_caution.wav")
        critical_samples = load_wav("assets/sounds/chime_critical.wav")

        for event_time_sec, alert_level in alert_events:
            start_idx = int(event_time_sec * sample_rate)
            beep = critical_samples if alert_level == "CRITICAL" else caution_samples
            for i, s in enumerate(beep):
                pos = start_idx + i
                if pos < len(audio_buffer):
                    audio_buffer[pos] = max(-1.0, min(1.0, audio_buffer[pos] + s))

        temp_wav = os.path.join(self.output_dir, "temp_profile_audio.wav")
        with wave.open(temp_wav, "w") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)
            out_wf.setframerate(sample_rate)
            raw_out = bytearray()
            for s in audio_buffer:
                raw_out.extend(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)))
            out_wf.writeframes(raw_out)

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", raw_video_path,
            "-i", temp_wav,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ac", "2",
            "-b:a", "192k",
            "-shortest",
            final_video_path,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(final_video_path):
                print(f"✓ Generated QuickTime Native H.264 + Stereo AAC Benchmark Video: '{final_video_path}'")
        except Exception as e:
            print(f"[Warning] Audio muxing skipped: {e}")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            if os.path.exists(raw_video_path):
                os.remove(raw_video_path)

    def _export_reports(self, report: ProfilerReport) -> None:
        """Saves JSON and Markdown reports to outputs/ directory."""
        json_path = os.path.join(self.output_dir, "profiler_benchmark_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

        md_path = os.path.join(self.output_dir, "profiler_benchmark_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Milestone 12: Real-Time Stream Performance Profiling Report\n\n")
            f.write(f"- **Hardware Platform**: {report.platform_name}\n")
            f.write(f"- **Acceleration Device**: {report.device}\n")
            f.write(f"- **Resolution**: {report.resolution[0]}x{report.resolution[1]}\n")
            f.write(f"- **Target Frame Rate**: {report.target_fps:.1f} FPS (Budget: {report.real_time_budget_ms:.2f} ms)\n")
            f.write(f"- **Achieved Throughput**: **{report.achieved_fps:.2f} FPS**\n")
            f.write(f"- **Real-Time Compliant (p95 <= budget)**: **{'YES' if report.is_realtime_capable else 'NO'}** (Headroom: {report.budget_headroom_ms:+.2f} ms)\n\n")
            f.write("### End-to-End Latency Metrics\n\n")
            f.write("| Metric | Latency (ms) |\n")
            f.write("|---|---|\n")
            f.write(f"| **Average Latency** | {report.average_frame_latency_ms:.2f} ms |\n")
            f.write(f"| **p50 (Median)** | {report.p50_frame_latency_ms:.2f} ms |\n")
            f.write(f"| **p95 (Tail Latency)** | {report.p95_frame_latency_ms:.2f} ms |\n")
            f.write(f"| **p99 (Peak Spike)** | {report.p99_frame_latency_ms:.2f} ms |\n")
            f.write(f"| **Max Worst-Case** | {report.max_frame_latency_ms:.2f} ms |\n")
            f.write(f"| **Frame Jitter (Std Dev)** | {report.jitter_ms:.2f} ms |\n\n")
            f.write("### Stage-by-Stage Latency Breakdown\n\n")
            f.write("| Stage | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | % of Pipeline |\n")
            f.write("|---|---|---|---|---|---|\n")
            for st in report.stages:
                f.write(f"| `{st.name}` | {st.mean_ms:.2f} ms | {st.p50_ms:.2f} ms | {st.p95_ms:.2f} ms | {st.p99_ms:.2f} ms | **{st.percentage_of_total:.1f}%** |\n")

    def _print_terminal_summary(self, report: ProfilerReport) -> None:
        """Prints a clean CLI summary table."""
        print("\n" + "=" * 70)
        print("  Milestone 12: Performance Profiling Results")
        print("=" * 70)
        print(f" Platform           : {report.platform_name}")
        print(f" Hardware Backend   : {report.device}")
        print(f" Target FPS Budget  : {report.target_fps:.1f} FPS ({report.real_time_budget_ms:.2f} ms/frame)")
        print(f" Achieved Speed     : {report.achieved_fps:.2f} FPS")
        print(f" p50 (Median)       : {report.p50_frame_latency_ms:.2f} ms")
        print(f" p95 (Tail)         : {report.p95_frame_latency_ms:.2f} ms")
        print(f" p99 (Peak)         : {report.p99_frame_latency_ms:.2f} ms")
        print(f" Frame Jitter (Std) : {report.jitter_ms:.2f} ms")
        print(f" Real-Time Capable  : {'✓ YES (Comfortable Headroom)' if report.is_realtime_capable else '✗ NO (Frame Drops Detected)'}")
        print(f" 95th Headroom      : {report.budget_headroom_ms:+.2f} ms relative to 25 FPS budget")
        print("-" * 70)
        print("  Stage-by-Stage Latency Distribution:")
        print("-" * 70)
        print(f" {'Stage':<26} | {'Mean':>7} | {'p50':>7} | {'p95':>7} | {'% Total':>8}")
        print("-" * 70)
        for st in report.stages:
            print(f" {st.name:<26} | {st.mean_ms:5.2f}ms | {st.p50_ms:5.2f}ms | {st.p95_ms:5.2f}ms | {st.percentage_of_total:6.1f}%")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RoadWatch AI Milestone 12 Real-Time Profiler")
    parser.add_argument("--video", type=str, default="data/raw/sample_dashcam.mp4", help="Video path")
    parser.add_argument("--frames", type=int, default=220, help="Frames to profile")
    parser.add_argument("--start-frame", type=int, default=1050, help="Start frame index")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps or cpu)")
    parser.add_argument("--no-video", action="store_true", help="Skip saving annotated video")
    args = parser.parse_args()

    profiler = StreamProfiler(
        video_path=args.video,
        device=args.device,
        save_annotated=not args.no_video,
    )
    profiler.run(max_frames=args.frames, start_frame=args.start_frame)


if __name__ == "__main__":
    main()
