"""
RoadWatch AI - Milestone 8: Rider Warning Engine
------------------------------------------------
Transforms raw vehicle risk scores into actionable, rider-friendly warnings.
Prevents Alert Fatigue through:
- Debouncing (requires danger to persist across consecutive frames)
- Cooldown timers (suppresses repetitive alarms for the same vehicle)
- Alert escalation (elevates Caution -> Critical immediately)
- Hysteresis (prevents flickering near decision boundaries)
- Threat prioritization (selects single highest-priority threat)
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Optional


class AlertLevel(str, Enum):
    INFO = "INFO"
    CAUTION = "CAUTION"
    CRITICAL = "CRITICAL"


@dataclass
class RiderAlert:
    """Represents an actionable alert event delivered to the rider."""
    alert_id: str
    timestamp_sec: float
    frame_idx: int
    vehicle_id: int
    level: AlertLevel
    title: str
    message: str
    suggested_action: str
    risk_score: float
    direction: str  # 'left', 'center', 'right'

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


class WarningEngine:
    """
    Stateful warning manager that filters, debounces, and prioritizes road hazard alerts.
    """

    def __init__(
        self,
        caution_threshold: float = 40.0,
        critical_threshold: float = 70.0,
        debounce_frames: int = 3,
        cooldown_seconds: float = 3.5,
        hysteresis_delta: float = 8.0,
    ):
        """
        Args:
            caution_threshold: Score needed to trigger a CAUTION alert.
            critical_threshold: Score needed to trigger a CRITICAL alert.
            debounce_frames: Number of consecutive frames risk must persist before alerting.
            cooldown_seconds: Duration to silence repeat alerts for the same vehicle.
            hysteresis_delta: Points risk must drop below threshold before clearing state.
        """
        self.caution_threshold = caution_threshold
        self.critical_threshold = critical_threshold
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds
        self.hysteresis_delta = hysteresis_delta

        # State tracking per vehicle_id
        # Consecutive frame counters: {vid: count}
        self.consecutive_danger_frames: Dict[int, int] = {}
        # Last alert timestamp: {vid: timestamp_sec}
        self.last_alert_timestamp: Dict[int, float] = {}
        # Last alert level issued: {vid: AlertLevel}
        self.last_alert_level: Dict[int, AlertLevel] = {}

        # System alert sequence counter
        self.alert_counter = 0

        # Currently active display alert (with expiry timestamp)
        self.active_display_alert: Optional[RiderAlert] = None
        self.active_alert_expiry: float = 0.0

    def _determine_direction(self, lateral_zone: str) -> str:
        """Maps spatial zone to human-friendly rider direction."""
        if "LEFT" in lateral_zone.upper():
            return "left"
        elif "RIGHT" in lateral_zone.upper():
            return "right"
        return "ahead"

    def _compose_alert(
        self,
        timestamp_sec: float,
        frame_idx: int,
        vehicle_eval: Dict[str, Any],
        level: AlertLevel,
    ) -> RiderAlert:
        """Creates a structured, actionable RiderAlert."""
        self.alert_counter += 1
        vid = vehicle_eval.get("vehicle_id", 0)
        score = vehicle_eval.get("risk_score", 0.0)
        cls_name = vehicle_eval.get("class_name", "vehicle")
        primary_factor = vehicle_eval.get("primary_factor", "hazard")
        zone = vehicle_eval.get("lateral_zone", "EGO_LANE")
        direction = self._determine_direction(zone)

        dir_label = "from your left" if direction == "left" else (
            "from your right" if direction == "right" else "directly ahead"
        )

        if level == AlertLevel.CRITICAL:
            title = f"DANGER: High-Risk {cls_name.upper()} {dir_label.upper()}"
            if "cutting" in primary_factor.lower():
                message = f"{cls_name.capitalize()} cutting abruptly across your lane."
                action = "Check mirrors, maintain steering control, and create distance."
            elif "braking" in primary_factor.lower():
                message = f"{cls_name.capitalize()} braking sharply ahead of you."
                action = "Apply brakes progressively and scan escape path."
            else:
                message = f"{cls_name.capitalize()} closing distance rapidly ({dir_label})."
                action = "Slow down and give vehicle wide berth."
        else:  # CAUTION
            title = f"CAUTION: {cls_name.capitalize()} {dir_label}"
            message = f"{cls_name.capitalize()} nearby ({primary_factor})."
            action = "Monitor vehicle and maintain safe buffer."

        return RiderAlert(
            alert_id=f"RW-ALERT-{self.alert_counter:04d}",
            timestamp_sec=timestamp_sec,
            frame_idx=frame_idx,
            vehicle_id=vid,
            level=level,
            title=title,
            message=message,
            suggested_action=action,
            risk_score=score,
            direction=direction,
        )

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        vehicle_evaluations: List[Dict[str, Any]],
    ) -> Optional[RiderAlert]:
        """
        Processes all vehicle risk evaluations for the current frame.
        Applies debouncing, cooldown, and prioritization.

        Returns:
            A newly fired RiderAlert if triggered this frame, or None if suppressed.
        """
        # Clear active display alert if display duration has passed (2.0s display persistence)
        if self.active_display_alert and timestamp_sec > self.active_alert_expiry:
            self.active_display_alert = None

        candidate_alerts: List[RiderAlert] = []

        # Identify which vehicles currently exceed caution or critical thresholds
        active_vehicle_ids = set()

        for v in vehicle_evaluations:
            vid = v.get("vehicle_id")
            if vid is None:
                continue
            active_vehicle_ids.add(vid)

            score = v.get("risk_score", 0.0)

            # Check if vehicle exceeds caution threshold
            if score >= self.caution_threshold:
                # Increment debounce counter
                self.consecutive_danger_frames[vid] = self.consecutive_danger_frames.get(vid, 0) + 1
            else:
                # Apply hysteresis before resetting counter
                if score < (self.caution_threshold - self.hysteresis_delta):
                    self.consecutive_danger_frames[vid] = 0

            danger_frames = self.consecutive_danger_frames.get(vid, 0)

            # Debounce check: Danger must persist for at least debounce_frames
            if danger_frames >= self.debounce_frames:
                target_level = (
                    AlertLevel.CRITICAL if score >= self.critical_threshold else AlertLevel.CAUTION
                )

                # Cooldown & Escalation Check
                last_time = self.last_alert_timestamp.get(vid, -999.0)
                last_level = self.last_alert_level.get(vid, AlertLevel.INFO)
                time_since_last = timestamp_sec - last_time

                # Escalation rule: Caution -> Critical overrides cooldown immediately!
                is_escalation = (last_level == AlertLevel.CAUTION and target_level == AlertLevel.CRITICAL)
                is_cooldown_expired = time_since_last >= self.cooldown_seconds

                if is_cooldown_expired or is_escalation:
                    alert = self._compose_alert(timestamp_sec, frame_idx, v, target_level)
                    candidate_alerts.append(alert)

        # Clean up state for vehicles that have left the scene
        known_vids = list(self.consecutive_danger_frames.keys())
        for k_vid in known_vids:
            if k_vid not in active_vehicle_ids:
                # Vehicle exited frame; reset debounce counter
                self.consecutive_danger_frames[k_vid] = 0

        # Prioritization: If multiple candidate alerts exist, pick the highest risk score
        if candidate_alerts:
            candidate_alerts.sort(key=lambda a: a.risk_score, reverse=True)
            top_alert = candidate_alerts[0]

            # Update alert state for the chosen vehicle
            self.last_alert_timestamp[top_alert.vehicle_id] = timestamp_sec
            self.last_alert_level[top_alert.vehicle_id] = top_alert.level

            # Set as active display alert for rider HUD (persists on screen for 2.5 seconds)
            self.active_display_alert = top_alert
            self.active_alert_expiry = timestamp_sec + 2.5

            return top_alert

        return None
