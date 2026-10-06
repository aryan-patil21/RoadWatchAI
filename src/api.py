"""
RoadWatch AI - Milestone 9: FastAPI Backend Service
---------------------------------------------------
REST & WebSocket API providing communication between the RoadWatch AI vision pipeline
and client applications (iOS SwiftUI app, Web Dashboards, External Dashcams).

Endpoints:
- GET  /health                 : Service health and hardware status
- POST /api/v1/analyze/frame   : Analyzes uploaded image frame, returns detections, risk, and active alerts
- GET  /api/v1/alerts/active   : Returns currently active rider alert
- GET  /api/v1/alerts/history  : Returns historical log of fired alerts
- GET  /api/v1/sounds/{name}   : Serves open-source .WAV alert chimes
- WS   /api/v1/ws/alerts       : Real-time WebSocket channel pushing alerts to iOS client
"""

import os
import sys
import io
import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vehicle_tracker import VehicleTracker
from src.road_geometry import RoadGeometryAnalyzer
from src.feature_extractor import BehaviouralFeatureExtractor
from src.risk_engine import RiskEngine
from src.warning_engine import WarningEngine, RiderAlert

app = FastAPI(
    title="RoadWatch AI API",
    description="AI-powered road safety and driver-behaviour analysis API for two-wheeler commuters.",
    version="1.0.0",
)

# Enable CORS for mobile and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Pipeline State
class PipelineState:
    def __init__(self):
        self.tracker = VehicleTracker(conf_threshold=0.25)
        self.geometry_analyzer = RoadGeometryAnalyzer()
        self.feature_extractor = BehaviouralFeatureExtractor(fps=25.0, frame_width=854, frame_height=480)
        self.risk_engine = RiskEngine()
        self.warning_engine = WarningEngine(debounce_frames=3, cooldown_seconds=3.5)
        self.frame_counter = 0
        self.session_alerts: List[Dict[str, Any]] = []


pipeline_state = PipelineState()


# Response Models
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    device: str
    frames_processed: int
    alerts_fired: int


@app.get("/", tags=["General"])
def root():
    return {
        "message": "RoadWatch AI Backend Service is running",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return HealthResponse(
        status="healthy",
        service="RoadWatch AI",
        version="1.0.0",
        device=pipeline_state.tracker.device.upper(),
        frames_processed=pipeline_state.frame_counter,
        alerts_fired=len(pipeline_state.session_alerts),
    )


@app.post("/api/v1/analyze/frame", tags=["Vision Pipeline"])
async def analyze_frame(file: UploadFile = File(...)):
    """
    Accepts an uploaded image frame (JPEG/PNG), processes it through:
    YOLO -> ByteTrack -> Geometry -> Features -> Risk Engine -> Warning Engine,
    and returns detected vehicles, spatial metrics, risk scores, and active alerts.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG or PNG).")

    # Read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    height, width = frame.shape[:2]
    pipeline_state.frame_counter += 1
    frame_idx = pipeline_state.frame_counter
    timestamp_sec = round(frame_idx * 0.04, 2)

    # 1. Tracking
    _, tracks = pipeline_state.tracker.track_frame(frame, persist=True)

    vehicle_evaluations: List[Dict[str, Any]] = []
    response_vehicles: List[Dict[str, Any]] = []

    for t in tracks:
        track_id = t["track_id"]
        if track_id is None:
            continue

        # 2. Geometry
        spatial_data = pipeline_state.geometry_analyzer.analyze_vehicle(frame_idx, t)
        proximity = spatial_data["proximity_score"]

        # 3. Features
        features = pipeline_state.feature_extractor.extract_features(
            frame_idx=frame_idx,
            vehicle_id=track_id,
            class_name=t["class_name"],
            bbox=t["bbox"],
            proximity_score=proximity,
        )

        # 4. Risk Evaluation
        risk_eval = pipeline_state.risk_engine.evaluate_risk(features)
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

    # 5. Warning Engine
    new_alert = pipeline_state.warning_engine.process_frame(frame_idx, timestamp_sec, vehicle_evaluations)
    if new_alert:
        pipeline_state.session_alerts.append(new_alert.to_dict())

    active_alert = pipeline_state.warning_engine.active_display_alert

    return {
        "frame_idx": frame_idx,
        "timestamp_sec": timestamp_sec,
        "frame_width": width,
        "frame_height": height,
        "tracked_vehicle_count": len(response_vehicles),
        "tracked_vehicles": response_vehicles,
        "newly_fired_alert": new_alert.to_dict() if new_alert else None,
        "active_rider_alert": active_alert.to_dict() if active_alert else None,
        "system_status": "NORMAL" if not active_alert else active_alert.level.value,
    }


@app.get("/api/v1/alerts/active", tags=["Alerts"])
def get_active_alert():
    """Returns currently active alert being shown on the rider's cockpit display."""
    active = pipeline_state.warning_engine.active_display_alert
    return {
        "has_active_alert": (active is not None),
        "alert": active.to_dict() if active else None,
        "timestamp": time.time(),
    }


@app.get("/api/v1/alerts/history", tags=["Alerts"])
def get_alerts_history():
    """Returns all alerts triggered during the current riding session."""
    return {
        "total_alerts": len(pipeline_state.session_alerts),
        "alerts": pipeline_state.session_alerts,
    }


@app.get("/api/v1/sounds/{sound_filename}", tags=["Audio Assets"])
def get_sound_asset(sound_filename: str):
    """Serves open-source .WAV alert chimes to mobile clients or web apps."""
    sound_path = os.path.join("assets", "sounds", sound_filename)
    if not os.path.exists(sound_path):
        raise HTTPException(status_code=404, detail=f"Sound asset '{sound_filename}' not found.")
    return FileResponse(sound_path, media_type="audio/wav")


@app.websocket("/api/v1/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket connection for iOS app to receive live alert notifications
    without polling.
    """
    await websocket.accept()
    last_sent_alert_id = None
    try:
        while True:
            active = pipeline_state.warning_engine.active_display_alert
            if active and active.alert_id != last_sent_alert_id:
                last_sent_alert_id = active.alert_id
                await websocket.send_json({
                    "event": "NEW_ALERT",
                    "payload": active.to_dict(),
                })
            elif not active and last_sent_alert_id is not None:
                last_sent_alert_id = None
                await websocket.send_json({
                    "event": "ALL_CLEAR",
                    "payload": {"status": "NORMAL"},
                })
            # Sleep 100ms between pushes (10Hz push cycle)
            import asyncio
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
