from sqlalchemy.orm import Session
from app import models, schemas

# Hazard CRUDS
def get_hazard(db: Session, hazard_id: int):
    return db.query(models.Hazard).filter(models.Hazard.id == hazard_id).first()

def get_hazards(db: Session, skip: int = 0, limit: int = 100, status: str = None):
    query = db.query(models.Hazard)
    if status:
        query = query.filter(models.Hazard.status == status)
    return query.offset(skip).limit(limit).all()

def create_hazard(db: Session, hazard: schemas.HazardCreate):
    db_hazard = models.Hazard(
        type=hazard.type,
        latitude=hazard.latitude,
        longitude=hazard.longitude,
        severity=hazard.severity,
        status=hazard.status,
        confidence=hazard.confidence,
        timestamp=hazard.timestamp,
        source=hazard.source,
        # Legacy lifecycle fields support the observation pipeline.
        first_detected=hazard.timestamp.isoformat(),
        last_detected=hazard.timestamp.isoformat(),
    )
    db.add(db_hazard)
    db.commit()
    db.refresh(db_hazard)
    return db_hazard

def update_hazard(db: Session, db_hazard: models.Hazard, updates: dict):
    for key, value in updates.items():
        setattr(db_hazard, key, value)
    db.commit()
    db.refresh(db_hazard)
    return db_hazard

def delete_hazard(db: Session, hazard_id: int):
    db_hazard = db.query(models.Hazard).filter(models.Hazard.id == hazard_id).first()
    if db_hazard:
        db.delete(db_hazard)
        db.commit()
        return True
    return False


# Observation CRUDS
def get_observation(db: Session, observation_id: int):
    return db.query(models.Observation).filter(models.Observation.id == observation_id).first()

def get_observations(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Observation).offset(skip).limit(limit).all()

def create_observation(db: Session, observation: schemas.ObservationCreate):
    db_observation = models.Observation(
        hazard_id=observation.hazard_id,
        source=observation.source,
        latitude=observation.latitude,
        longitude=observation.longitude,
        confidence=observation.confidence,
        timestamp=observation.timestamp,
        sensor_evidence=observation.sensor_evidence
    )
    db.add(db_observation)
    db.commit()
    db.refresh(db_observation)
    return db_observation
