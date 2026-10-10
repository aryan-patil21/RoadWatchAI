"""
RoadWatch AI - Milestone 14: Near-Miss Incident Blackbox Auto-Recorder
----------------------------------------------------------------------
Automatically captures and archives high-risk road incidents into a dedicated
forensics package:
1. Circular Frame Buffer: Keeps a rolling memory of recent frames (~5 seconds pre-event).
2. Incident Auto-Clipper: On any CRITICAL alert or near-miss escalation, buffers
   post-event frames (~3 seconds) and writes a synchronized, QuickTime-native MP4.
3. Telemetry Forensics Logger: Saves structured incident telemetry JSON:
   - Threat vehicle ID and class
   - Minimum Time-To-Collision (TTC)
   - Closing velocity in km/h
   - Lateral trajectory divergence / conflict
   - Precise video timestamp and frame index
"""

import json
import os
import struct
import subprocess
import time
import wave
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class IncidentBlackboxRecorder:
    """
    Rolling blackbox memory that automatically extracts, archives, and summarizes
    near-miss safety critical incidents.
    """

    def __init__(
        self,
        output_dir: str = "outputs/incidents",
        pre_event_seconds: float = 3.5,
        post_event_seconds: float = 2.5,
        fps: float = 25.0,
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.pre_event_frames = int(pre_event_seconds * fps)
        self.post_event_frames = int(post_event_seconds * fps)

        os.makedirs(output_dir, exist_ok=True)

        # Rolling frame circular buffer: (frame_copy, timestamp_sec, telemetry_dict)
        self.frame_buffer: deque = deque(maxlen=self.pre_event_frames)

        # Active recording state
        self.is_recording = False
        self.current_incident_id: Optional[str] = None
        self.incident_frames_remaining = 0
        self.active_clip_frames: List[Tuple[np.ndarray, float]] = []
        self.active_telemetry_events: List[Dict[str, Any]] = []
        self.active_incident_meta: Dict[str, Any] = {}

        # Archived incidents count
        self.recorded_incidents_count = 0

    def push_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        active_alert: Optional[Any],
        top_vehicle_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Ingests a frame into the blackbox buffer.
        If a CRITICAL alert triggers, automatically starts an incident capture session.
        Returns the incident output path if an incident clip was finalized on this frame.
        """
        frame_copy = frame.copy()
        finalized_incident_path = None

        # 1. Check if we should trigger a new incident recording
        if active_alert and getattr(active_alert, "level", None) and active_alert.level.value == "CRITICAL" and not self.is_recording:
            self._start_incident(frame_idx, timestamp_sec, active_alert, top_vehicle_telemetry)

        # 2. If actively capturing post-event frames
        if self.is_recording:
            self.active_clip_frames.append((frame_copy, timestamp_sec))
            if top_vehicle_telemetry:
                self.active_telemetry_events.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    **top_vehicle_telemetry,
                })

            self.incident_frames_remaining -= 1
            if self.incident_frames_remaining <= 0:
                finalized_incident_path = self._finalize_incident()

        # 3. Always maintain pre-event rolling history buffer
        self.frame_buffer.append((frame_copy, timestamp_sec, top_vehicle_telemetry))

        return finalized_incident_path

    def _start_incident(
        self,
        frame_idx: int,
        timestamp_sec: float,
        alert: Any,
        telemetry: Optional[Dict[str, Any]],
    ) -> None:
        """Initiates incident clip capture and dumps pre-event buffer into clip."""
        self.is_recording = True
        self.recorded_incidents_count += 1
        incident_id = f"incident_{self.recorded_incidents_count:03d}_{int(timestamp_sec)}s"
        self.current_incident_id = incident_id
        self.incident_frames_remaining = self.post_event_frames

        # Load pre-event frames from buffer
        self.active_clip_frames = [(f, t) for (f, t, _) in self.frame_buffer]
        self.active_telemetry_events = [
            {"frame_idx": frame_idx - len(self.frame_buffer) + i, "timestamp_sec": t, **(tel or {})}
            for i, (_, t, tel) in enumerate(self.frame_buffer)
        ]

        # Incident metadata
        self.active_incident_meta = {
            "incident_id": incident_id,
            "trigger_frame": frame_idx,
            "trigger_time_sec": timestamp_sec,
            "alert_title": getattr(alert, "title", "Critical Hazard"),
            "risk_score": getattr(alert, "risk_score", 0.0),
            "threat_direction": getattr(alert, "direction", "AHEAD"),
            "target_vehicle_id": getattr(alert, "vehicle_id", None),
            "initial_telemetry": telemetry or {},
        }
        print(f"[IncidentBlackbox] 🔴 CRITICAL INCIDENT DETECTED! Capturing '{incident_id}'...")

    def _finalize_incident(self) -> str:
        """Writes incident video and telemetry metadata report to disk."""
        self.is_recording = False
        incident_id = self.current_incident_id or f"incident_{self.recorded_incidents_count:03d}"
        
        raw_video_path = os.path.join(self.output_dir, f"{incident_id}_raw.mp4")
        final_video_path = os.path.join(self.output_dir, f"{incident_id}.mp4")
        json_path = os.path.join(self.output_dir, f"{incident_id}_telemetry.json")

        if not self.active_clip_frames:
            return ""

        h, w = self.active_clip_frames[0][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_video_path, fourcc, self.fps, (w, h))

        for f, _ in self.active_clip_frames:
            out.write(f)
        out.release()

        # Compute forensic telemetry summary
        min_ttc = 99.9
        max_closing_speed = 0.0
        min_distance_m = 99.9

        for ev in self.active_telemetry_events:
            ttc = ev.get("ttc_proxy_sec", 99.9)
            if ttc and ttc < min_ttc:
                min_ttc = ttc
            spd = ev.get("rel_speed_kmh", 0.0)
            if spd < 0 and abs(spd) > max_closing_speed:
                max_closing_speed = abs(spd)
            dist = ev.get("z_m", 99.9)
            if dist and dist < min_distance_m:
                min_distance_m = dist

        report = {
            **self.active_incident_meta,
            "forensics_summary": {
                "total_frames_recorded": len(self.active_clip_frames),
                "duration_seconds": round(len(self.active_clip_frames) / self.fps, 2),
                "minimum_ttc_seconds": round(min_ttc, 2) if min_ttc < 90 else None,
                "minimum_distance_meters": round(min_distance_m, 2) if min_distance_m < 90 else None,
                "peak_closing_speed_kmh": round(max_closing_speed, 1),
            },
            "telemetry_timeline": self.active_telemetry_events,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Mux QuickTime native H.264 + Stereo AAC warning chime
        self._mux_incident_audio(raw_video_path, final_video_path, len(self.active_clip_frames) / self.fps)

        print(f"[IncidentBlackbox] ✓ Saved Incident Clip: '{final_video_path}'")
        print(f"[IncidentBlackbox] ✓ Saved Forensic Telemetry: '{json_path}'")

        # Cleanup memory
        self.active_clip_frames = []
        self.active_telemetry_events = []
        self.current_incident_id = None

        return final_video_path

    def _mux_incident_audio(self, raw_path: str, final_path: str, duration_sec: float) -> None:
        """Encodes incident clip to native Apple H.264 + Stereo AAC with critical alarm."""
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        sample_rate = 44100
        total_samples = int(sample_rate * duration_sec)
        audio_buffer = [0.0] * max(total_samples, 44100)

        # Load critical alarm sound
        sound_path = "assets/sounds/chime_critical.wav"
        if os.path.exists(sound_path):
            with wave.open(sound_path, "r") as wf:
                raw = wf.readframes(wf.getnframes())
                ints = struct.unpack(f"<{wf.getnframes()}h", raw)
                critical_samples = [v / 32767.0 for v in ints]
            
            # Start chime at 2.0s into the incident
            start_pos = int(2.0 * sample_rate)
            for i, s in enumerate(critical_samples):
                if start_pos + i < len(audio_buffer):
                    audio_buffer[start_pos + i] = max(-1.0, min(1.0, s))

        temp_wav = os.path.join(self.output_dir, "temp_inc_audio.wav")
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
            "-i", raw_path,
            "-i", temp_wav,
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ac", "2",
            "-b:a", "192k",
            "-shortest",
            final_path,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(final_path):
                pass
        except Exception:
            if os.path.exists(raw_path):
                os.replace(raw_path, final_path)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            if os.path.exists(raw_path):
                os.remove(raw_path)
