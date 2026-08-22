def calculate_severity(bbox_area_ratio: float, confidence: float) -> str:
    """
    Determine observation severity ('low', 'medium', 'high') based on 
    the relative bounding box area and classification confidence.
    
    Args:
        bbox_area_ratio: The relative size of the pothole (width * height / frame_resolution)
        confidence: YOLO class confidence score (0.0 to 1.0)
        
    Returns:
        str: 'low' | 'medium' | 'high'
    """
    # Heuristics: large visual areas represent major physical depressions in close proximity
    if bbox_area_ratio >= 0.05 and confidence >= 0.75:
        return "high"
    elif bbox_area_ratio >= 0.015:
        return "medium"
    else:
        return "low"
