from typing import List
from app import models

def calculate_road_health(hazards: List[models.Hazard]) -> dict:
    """
    Calculate a prototype road health rating index based on spatial density and severity.
    
    Formula:
      Score = 100 - Sum(weights * hazard_counts)
      where:
        high_severity = 10 pts deduction
        medium_severity = 5 pts deduction
        low_severity = 2 pts deduction
        
    Args:
        hazards: List of active Hazard objects in the targeted geofence radius.
        
    Returns:
        dict: containing the prototype score, count breakdown, and engineering disclaimer.
    """
    score = 100.0
    high_count = 0
    med_count = 0
    low_count = 0

    for h in hazards:
        # Only calculate based on active states
        if h.status in ["DETECTED", "VERIFIED", "ACTIVE"]:
            if h.severity == "high":
                score -= 10.0
                high_count += 1
            elif h.severity == "medium":
                score -= 5.0
                med_count += 1
            elif h.severity == "low":
                score -= 2.0
                low_count += 1

    # Clamp the index score bounds
    score = max(0.0, score)

    return {
        "prototype_road_health_score": score,
        "active_hazards": high_count + med_count + low_count,
        "breakdown": {
            "high_severity": high_count,
            "medium_severity": med_count,
            "low_severity": low_count
        },
        "notice": "PROTOTYPE ONLY - NOT FOR ENGINEERING USE"
    }
