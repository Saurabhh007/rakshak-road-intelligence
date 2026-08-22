import json
import os
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine, Base
from app import models

def seed_database():
    """
    Ensure SQLite database tables exist and seeds them with coordinates
    and default observations from seed_hazards.json.
    """
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if database is already seeded to avoid duplicates
        if db.query(models.Hazard).count() > 0:
            print("DB: Hazards already seeded. Skipping.")
            return
            
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        seed_path = os.path.join(project_root, "data", "samples", "seed_hazards.json")
        if not os.path.exists(seed_path):
            print(f"DB: Seed hazards file not found at: '{seed_path}'")
            return
            
        with open(seed_path, "r") as f:
            hazards_data = json.load(f)
            
        print(f"DB: Seeding {len(hazards_data)} hazards...")
        for h_data in hazards_data:
            hazard = models.Hazard(
                type=h_data.get("type", "pothole"),
                latitude=h_data["latitude"],
                longitude=h_data["longitude"],
                severity=h_data["severity"],
                status=h_data["status"],
                confidence=h_data["confidence"],
                timestamp=datetime.fromisoformat(h_data["first_detected"].replace("Z", "+00:00")),
                source="ai_camera",
                first_detected=h_data["first_detected"],
                last_detected=h_data["last_detected"]
            )
            db.add(hazard)
            db.flush()  # Generate auto-incremented primary key ID
            
            # Create a corresponding raw observation linked to this hazard
            observation = models.Observation(
                hazard_id=hazard.id,
                source="ai_camera" if h_data["status"] != "manual_report" else "manual_report",
                latitude=h_data["latitude"],
                longitude=h_data["longitude"],
                confidence=h_data["confidence"],
                timestamp=h_data["first_detected"],
                sensor_evidence=None
            )
            db.add(observation)
            
        db.commit()
        print("DB: Seeding completed successfully.")
    except Exception as e:
        db.rollback()
        print(f"DB: Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
