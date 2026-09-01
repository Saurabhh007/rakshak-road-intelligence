import { useState } from "react";
import MapView from "./components/MapView";
import LivePerceptionPanel from "./components/LivePerceptionPanel";
import ActiveHazardsPanel from "./components/ActiveHazardsPanel";
import StatsPanel from "./components/StatsPanel";
import AlertBanner from "./components/AlertBanner";
import useTelemetry from "./hooks/useTelemetry";
import { API_BASE_URL } from "./config";
import demoRoute from "../../data/samples/demo_route.json";

import "./App.css";

export default function App() {
    const [routeState, setRouteState] = useState("STOPPED");
    const [routeSpeed, setRouteSpeed] = useState(1);

    const {
        currentCoords,
        warning,
        currentMode,
        modeInfo,
        switchMode,
        realInferenceResult,
        realInferenceLoading,
        triggerRealInference,
        uploadedInferenceResult,
        uploadedInferenceLoading,
        uploadedInferenceError,
        runUploadedImageInference,
        systemStatus,
        roadHealth,
        hazards,
        hazardsError,
        refreshData,
        resetSimulatedRoute,
        routeProgress
    } = useTelemetry(routeState, routeSpeed, () => setRouteState("COMPLETED"));

    // Reset database to initial seed coordinates
    const handleReset = async () => {
        if (!confirm("Are you sure you want to reset and reseed all database hazards?")) {
            return;
        }
        try {
            const response = await fetch(`${API_BASE_URL}/simulation/reset`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ mode: "DEMO" })
            });
            if (response.ok) {
                alert("Database reseeded successfully!");
                refreshData();
            }
        } catch (err) {
            console.error("Database reset error:", err);
            alert("Failed to reset database.");
        }
    };

    const startRoute = () => setRouteState("RUNNING");
    const pauseRoute = () => setRouteState("PAUSED");
    const resumeRoute = () => setRouteState("RUNNING");
    const resetRoute = () => {
        resetSimulatedRoute();
        setRouteState("STOPPED");
    };

    return (
        <div className="dashboard-root">
            {/* Top Bar Header with Mode Selector */}
            <header className="dashboard-header">
                <div className="header-left">
                    <h1 className="header-title">RAKSHAK</h1>
                    <span className="header-tagline">ROAD INTELLIGENCE ENGINE</span>
                </div>

                {/* Dashboard Mode Selector: Options A, B, C, D */}
                <div className="mode-selector-container">
                    <button
                        className={`mode-btn ${currentMode === "LIVE_CAMERA" ? "mode-btn-active mode-btn-a" : ""}`}
                        onClick={() => switchMode("LIVE_CAMERA")}
                    >
                        <span className="mode-letter">A</span> LIVE WEBCAM
                    </button>
                    <button
                        className={`mode-btn ${currentMode === "SYSTEM_DEMO" ? "mode-btn-active mode-btn-b" : ""}`}
                        onClick={() => switchMode("SYSTEM_DEMO")}
                    >
                        <span className="mode-letter">B</span> SYSTEM DEMO
                    </button>
                    <button
                        className={`mode-btn ${currentMode === "IMAGE_FALLBACK" ? "mode-btn-active mode-btn-c" : ""}`}
                        onClick={() => switchMode("IMAGE_FALLBACK")}
                    >
                        <span className="mode-letter">C</span> REAL AI IMAGE
                    </button>
                    <button
                        className={`mode-btn ${currentMode === "UPLOAD_IMAGE" ? "mode-btn-active mode-btn-d" : ""}`}
                        onClick={() => switchMode("UPLOAD_IMAGE")}
                    >
                        <span className="mode-letter">D</span> UPLOAD IMAGE
                    </button>
                </div>

                <div className="status-indicator">
                    <span className="status-dot pulsing-green"></span>
                    <span className="status-text uppercase font-bold text-green">
                        {currentMode === "LIVE_CAMERA" && "LIVE WEBCAM ACTIVE"}
                        {currentMode === "SYSTEM_DEMO" && "DEMO PIPELINE ACTIVE"}
                        {currentMode === "IMAGE_FALLBACK" && "AI IMAGE VERIFICATION"}
                        {currentMode === "UPLOAD_IMAGE" && "AI UPLOAD VERIFICATION"}
                    </span>
                </div>
            </header>

            {/* Main Content Grid */}
            <main className="dashboard-grid">
                {/* Left Side Mapping Section */}
                <section className="grid-cell map-cell">
                    <MapView 
                        hazards={hazards} 
                        currentCoords={currentCoords} 
                        routeCoordinates={demoRoute} 
                        routeState={routeState}
                        routeProgress={routeProgress}
                        isSimulating={routeState === "RUNNING"}
                        onToggleSim={routeState === "RUNNING" ? pauseRoute : startRoute}
                        gpsSource="SIMULATED"
                        onToggleGps={() => {}}
                        currentMode="SYSTEM_DEMO"
                        error={hazardsError}
                    />
                </section>

                {/* Right Side UI HUD Analytics */}
                <section className="grid-cell right-cell">
                    <LivePerceptionPanel 
                        currentMode={currentMode}
                        modeInfo={modeInfo}
                        systemStatus={systemStatus} 
                        hazards={hazards}
                        realInferenceResult={realInferenceResult}
                        realInferenceLoading={realInferenceLoading}
                        onRunInference={triggerRealInference}
                        uploadedInferenceResult={uploadedInferenceResult}
                        uploadedInferenceLoading={uploadedInferenceLoading}
                        uploadedInferenceError={uploadedInferenceError}
                        onUploadInference={runUploadedImageInference}
                        routeState={routeState}
                    />
                    <ActiveHazardsPanel hazards={hazards} />
                    <StatsPanel 
                        roadHealth={roadHealth} 
                        hazards={hazards}
                        onReset={handleReset}
                        routeState={routeState}
                        routeSpeed={routeSpeed}
                        onStartRoute={startRoute}
                        onPauseRoute={pauseRoute}
                        onResumeRoute={resumeRoute}
                        onResetRoute={resetRoute}
                        onSetRouteSpeed={setRouteSpeed}
                        routeProgress={routeProgress}
                    />
                </section>
            </main>

            {/* Bottom Alert Overlay Banner */}
            <footer className="dashboard-footer">
                <AlertBanner warning={warning} />
            </footer>
        </div>
    );
}
