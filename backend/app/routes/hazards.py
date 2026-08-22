"""Hazard CRUD endpoints for the MVP API."""

import logging
from math import asin, cos, radians, sin, sqrt

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/hazards", tags=["hazards"])


def _get_hazard_or_404(db: Session, hazard_id: int) -> models.Hazard:
    hazard = db.get(models.Hazard, hazard_id)
    if hazard is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hazard not found")
    return hazard


def _distance_meters(latitude: float, longitude: float, hazard: models.Hazard) -> float:
    """Return great-circle distance without requiring a spatial database."""
    earth_radius_meters = 6_371_000
    lat_delta = radians(hazard.latitude - latitude)
    lon_delta = radians(hazard.longitude - longitude)
    a = sin(lat_delta / 2) ** 2 + cos(radians(latitude)) * cos(radians(hazard.latitude)) * sin(lon_delta / 2) ** 2
    return earth_radius_meters * 2 * asin(sqrt(a))


@router.post("", response_model=schemas.Hazard, status_code=status.HTTP_201_CREATED)
def create_hazard(payload: schemas.HazardCreate, db: Session = Depends(get_db)):
    hazard = models.Hazard(
        type=payload.type,
        latitude=payload.latitude,
        longitude=payload.longitude,
        confidence=payload.confidence,
        severity=payload.severity.value,
        timestamp=payload.timestamp,
        status=payload.status.value,
        source=payload.source,
        first_detected=payload.timestamp.isoformat(),
        last_detected=payload.timestamp.isoformat(),
    )
    try:
        db.add(hazard)
        db.commit()
        db.refresh(hazard)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Could not create hazard")
        raise HTTPException(status_code=500, detail="Could not create hazard")
    return hazard


@router.get("", response_model=list[schemas.Hazard])
def list_hazards(
    status_filter: schemas.HazardStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(models.Hazard)
    if status_filter:
        query = query.filter(models.Hazard.status == status_filter.value)
    return query.order_by(models.Hazard.timestamp.desc(), models.Hazard.id.desc()).offset(offset).limit(limit).all()


@router.get("/nearby", response_model=list[schemas.Hazard])
def nearby_hazards(
    latitude: float = Query(ge=-90, le=90),
    longitude: float = Query(ge=-180, le=180),
    radius_meters: float = Query(default=500, gt=0, le=50_000),
    db: Session = Depends(get_db),
):
    hazards = db.query(models.Hazard).all()
    return [hazard for hazard in hazards if _distance_meters(latitude, longitude, hazard) <= radius_meters]


@router.get("/{hazard_id}", response_model=schemas.Hazard)
def get_hazard(hazard_id: int, db: Session = Depends(get_db)):
    return _get_hazard_or_404(db, hazard_id)


@router.patch("/{hazard_id}/status", response_model=schemas.Hazard)
def update_hazard_status(hazard_id: int, payload: schemas.HazardStatusUpdate, db: Session = Depends(get_db)):
    hazard = _get_hazard_or_404(db, hazard_id)
    try:
        hazard.status = payload.status.value
        db.commit()
        db.refresh(hazard)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Could not update hazard %s", hazard_id)
        raise HTTPException(status_code=500, detail="Could not update hazard status")
    return hazard
