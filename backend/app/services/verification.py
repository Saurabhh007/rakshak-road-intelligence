from sqlalchemy.orm import Session
from app import models, schemas, crud
from app.config import settings
from app.services.geofence import haversine_distance
from app.services.severity import calculate_severity

def process_observation(db: Session, observation_in: schemas.ObservationCreate) -> models.Hazard:
    """
    Process an incoming raw Observation event:
    1. Search for an existing physical Hazard within settings.HAZARD_CLUSTER_RADIUS_METERS.
    2. If found, average the coordinate positions, take the maximum confidence,
       and record the link in the database.
    3. If not found, create a new Hazard initialized with 'DETECTED' status.
    4. Apply Prototype Verification Logic: transition DETECTED -> VERIFIED when
       at least 3 temporally/spatially consistent observations are recorded, or if 
       manually reported.
    5. Save the observation and update/save the hazard record.
    
    Args:
        db: SQLAlchemy database session.
        observation_in: Pydantic schemas.ObservationCreate request model.
        
    Returns:
        models.Hazard: The aggregated/new hazard record.
    """
    # 1. Proximity matching to group observations under a single hazard
    hazards = db.query(models.Hazard).all()
    closest_hazard = None
    min_dist = float('inf')

    for h in hazards:
        # Ignore repaired/resolved records when aggregating new observations
        if h.status in ["REPORTED_REPAIRED", "RESOLVED"]:
            continue
        dist = haversine_distance(
            observation_in.latitude, observation_in.longitude,
            h.latitude, h.longitude
        )
        if dist < min_dist:
            min_dist = dist
            closest_hazard = h

    now_str = observation_in.timestamp

    # 2. Aggregation Logic
    if closest_hazard and min_dist <= settings.HAZARD_CLUSTER_RADIUS_METERS:
        # Fetch current observations already linked to this hazard
        linked_observations = closest_hazard.observations
        count = len(linked_observations)
        
        # Running average coordinate calculation
        new_lat = (closest_hazard.latitude * count + observation_in.latitude) / (count + 1)
        new_lng = (closest_hazard.longitude * count + observation_in.longitude) / (count + 1)
        
        # Max confidence score aggregation
        new_confidence = max(closest_hazard.confidence, observation_in.confidence)
        
        updates = {
            "latitude": new_lat,
            "longitude": new_lng,
            "confidence": new_confidence,
            "last_detected": now_str
        }
        
        # 3 & 4. Prototype Verification Heuristics
        # Transition DETECTED -> VERIFIED when count >= 3 (temporally/spatially consistent)
        if closest_hazard.status == "DETECTED":
            if (count + 1) >= 3 or observation_in.source == "manual_report":
                updates["status"] = "VERIFIED"
        # If it was already VERIFIED, we promote it to ACTIVE because it's recently sighted
        elif closest_hazard.status == "VERIFIED":
            updates["status"] = "ACTIVE"
            
        hazard = crud.update_hazard(db, closest_hazard, updates)
    else:
        # No nearby hazard - initialize a new Hazard entity
        initial_status = "DETECTED"
        if observation_in.source == "manual_report":
            initial_status = "VERIFIED"
            
        new_hazard_in = schemas.HazardCreate(
            type="pothole",
            latitude=observation_in.latitude,
            longitude=observation_in.longitude,
            severity="medium",  # Temp initial severity
            status=initial_status,
            confidence=observation_in.confidence,
            timestamp=now_str,
            source=observation_in.source,
        )
        hazard = crud.create_hazard(db, new_hazard_in)
        
    # Associate hazard_id with the observation and save
    observation_in.hazard_id = hazard.id
    db_obs = crud.create_observation(db, observation_in)
    
    # 5. Severity classification updates
    # Bbox area ratio helper mock: high confidence maps to a larger bounding box size
    area_ratio = 0.02 if db_obs.confidence < 0.8 else 0.06
    new_severity = calculate_severity(area_ratio, hazard.confidence)
    hazard = crud.update_hazard(db, hazard, {"severity": new_severity})
    
    return hazard
