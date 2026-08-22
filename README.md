# RAKSHAK — AI-Powered Road Intelligence & Hazard Warning System

RAKSHAK is an AI-powered dynamic road intelligence and hazard warning system designed as a modular Road Intelligence Engine. The system processes video streams (real webcam or local fallback `demo_video.mp4`), runs YOLO model object detection to log pothole observations, aggregates observations into verified road hazard events, maps them in real-time, and warns approaching drivers.

---

## 🚀 How to Run the Application

### Prerequisites
*   Python 3.10+ (Tested on Python 3.14.7)
*   Node.js (v18+) & npm

### 1. Backend Setup & Run
1.  Navigate to the project root directory.
2.  Install dependencies:
    ```bash
    pip install -r backend/requirements.txt
    ```
3.  Install the local AI module in editable mode:
    ```bash
    pip install -e ./ai
    ```
4.  Seed the database (populates SQLite `rakshak.db` with sample MG Road hazards):
    ```bash
    $env:PYTHONPATH="backend"
    python backend/app/seed_db.py
    ```
5.  Start the FastAPI backend:
    ```bash
    python backend/run.py
    ```
    The backend will start on [http://localhost:8000](http://localhost:8000). The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 2. Frontend Setup & Run
1.  Open a new terminal session and navigate to the `frontend` folder:
    ```bash
    cd frontend
    ```
2.  Install packages:
    ```bash
    npm install
    ```
3.  Start the Vite development server:
    ```bash
    npm run dev
    ```
4.  Open the application in your browser at [http://localhost:5173](http://localhost:5173).

---

## 💡 Pitching Guide for Hackathon Judges

When presenting to judges, focus on these three engineering strengths:

### 1. Observation vs. Hazard (Evidence vs. Truth)
> *"An AI prediction is evidence, not truth. We first create an observation. The verification layer then aggregates observations into a hazard."*
Explain that a single frame detection is a raw Observation. Only when multiple sightings are matched within a 10m clustering radius do we consolidate them into a singular `Hazard` record.

### 2. Prototype Verification Heuristics
Explain that the verification logic (`verification.py`) computes transitions dynamically. A new hazard starts as `DETECTED`. It only promotes to `VERIFIED` when:
*   We get $\ge 3$ temporally/spatially consistent observations (e.g. crossing the same spot multiple times).
*   An administrator manually validates the hazard.

### 3. Model-Agnostic AI Loader
Show that the AI layer is fully decoupled. We don't assume `yolov8n.pt` is trained for potholes. The weights, architecture, confidence threshold, and frame intervals are completely configurable via `backend/app/config.py`. 

---

## 🛠️ Configuration Tuning (`backend/app/config.py`)

You can edit these parameters to fine-tune the prototype during the hackathon:
*   `AI_FRAME_INTERVAL`: Process 1 in every N frames (reduce this if CPU is fast, increase if experiencing lag).
*   `WARNING_DISTANCE_METERS`: Distance geofence to warn drivers (default `120.0` meters).
*   `HAZARD_CLUSTER_RADIUS_METERS`: Range to group duplicate sightings (default `10.0` meters).
