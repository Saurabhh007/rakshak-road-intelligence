import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from app import schemas, crud, models
from app.database import get_db
from app.config import settings
from app.services.video_processor import video_processor
from app.services.geofence import haversine_distance, calculate_bearing
from app.services.road_health import calculate_road_health
from app.seed_db import seed_database

router = APIRouter()

def gen_frames():
    """
    Generator yielding the latest processed JPEG frame bytes as an MJPEG boundary stream.
    Serves frames directly from the VideoProcessor shared memory buffer,
    avoiding duplicate inference or redundant DB writes per browser client connection.
    """
    while True:
        frame_bytes = video_processor.get_latest_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Yield delay corresponding to ~25 FPS
        time.sleep(0.04)

@router.get("/stream/video")
def get_video_stream():
    """
    Exposes the OpenCV-annotated camera video stream.
    Rendered directly in React using an ordinary <img> tag.
    """
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/camera/status")
def get_camera_status():
    """Expose source connectivity without leaking a network-camera URL."""
    return video_processor.get_camera_status()

@router.post("/telemetry", response_model=schemas.TelemetryResponse)
def post_telemetry(request: schemas.TelemetryRequest, db: Session = Depends(get_db)):
    """
    Updates the driver's current position telemetry, updates the video capture
    feed coordinates index, and runs warning geofence checks.
    """
    # 1. Update VideoProcessor position state
    video_processor.set_coordinates(request.latitude, request.longitude)
    
    # 2. Query active hazards from SQLite
    active_hazards = db.query(models.Hazard).filter(
        models.Hazard.status.in_(["DETECTED", "VERIFIED", "ACTIVE"])
    ).all()
    
    # 3. Find closest warning target in alert geofence radius
    closest_warning = None
    min_distance = float('inf')
    
    for hazard in active_hazards:
        dist = haversine_distance(
            request.latitude, request.longitude,
            hazard.latitude, hazard.longitude
        )
        if dist <= settings.WARNING_DISTANCE_METERS:
            if dist < min_distance:
                min_distance = dist
                bearing = calculate_bearing(
                    request.latitude, request.longitude,
                    hazard.latitude, hazard.longitude
                )
                closest_warning = schemas.TelemetryWarning(
                    hazard_id=hazard.id,
                    type=hazard.type,
                    distance_meters=round(dist, 1),
                    bearing=round(bearing, 1),
                    severity=hazard.severity,
                    status=hazard.status
                )
                
    # 4. Formulate overall system status flags
    ai_status = "REAL" if not video_processor.detector.is_mock else "SIMULATED"
    
    return schemas.TelemetryResponse(
        warning_active=(closest_warning is not None),
        warning=closest_warning,
        system_status=schemas.SystemStatus(
            ai_engine=ai_status,
            gps=request.gps_source,
            backend="CONNECTED"
        )
    )

@router.get("/hazards/legacy", response_model=List[schemas.Hazard])
def list_hazards(status: str = None, db: Session = Depends(get_db)):
    """
    Lists road hazards stored in the database.
    """
    return crud.get_hazards(db, status=status)

@router.get("/road-health", response_model=schemas.RoadHealthResponse)
def get_road_health(latitude: float, longitude: float, radius_meters: int = 500, db: Session = Depends(get_db)):
    """
    Computes a Prototype Road Health Score rating index for the region
    surrounding the provided coordinates.
    """
    all_hazards = db.query(models.Hazard).all()
    local_hazards = []
    
    for h in all_hazards:
        dist = haversine_distance(latitude, longitude, h.latitude, h.longitude)
        if dist <= radius_meters:
            local_hazards.append(h)
            
    # Calculate health score using road_health logic
    score_data = calculate_road_health(local_hazards)
    
    return schemas.RoadHealthResponse(
        prototype_road_health_score=score_data["prototype_road_health_score"],
        evaluation_radius_meters=float(radius_meters),
        active_hazards=score_data["active_hazards"],
        breakdown=schemas.SeverityBreakdown(
            high_severity=score_data["breakdown"]["high_severity"],
            medium_severity=score_data["breakdown"]["medium_severity"],
            low_severity=score_data["breakdown"]["low_severity"]
        ),
        notice=score_data["notice"]
    )

@router.post("/simulation/reset", response_model=schemas.SimulationResetResponse)
def reset_simulation(request: schemas.SimulationResetRequest, db: Session = Depends(get_db)):
    """
    Wipes the database and re-seeds it with default hazards coordinates.
    """
    try:
        # Wipe tables
        db.query(models.Observation).delete()
        db.query(models.Hazard).delete()
        db.commit()
        
        # Reseed database from seed files
        seed_database()
        
        ai_status = "REAL" if not video_processor.detector.is_mock else "SIMULATED"
        
        return schemas.SimulationResetResponse(
            status="ok",
            ai_mode=ai_status,
            gps_source="SIMULATED"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database reset error: {e}")

@router.post("/hazards/{hazard_id}/status", response_model=schemas.Hazard)
def update_hazard_status(hazard_id: int, request: schemas.HazardStatusUpdate, db: Session = Depends(get_db)):
    """
    Manually overrides the lifecycle status of a road hazard record.
    Used for manual/admin reports (e.g. REPORTED_REPAIRED, RESOLVED).
    """
    db_hazard = crud.get_hazard(db, hazard_id=hazard_id)
    if not db_hazard:
        raise HTTPException(status_code=404, detail="Hazard not found")
        
    updated = crud.update_hazard(db, db_hazard, {"status": request.status})
    return updated
