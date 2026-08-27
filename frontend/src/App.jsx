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
    const [isSimulating, setIsSimulating] = useState(false);
    const [gpsSource, setGpsSource] = useState("SIMULATED");

    const {
        currentCoords,
        warning,
        systemStatus,
        roadHealth,
        hazards,
        hazardsError,
        refreshData
    } = useTelemetry(isSimulating, gpsSource);

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

    const handleToggleSim = () => {
        setIsSimulating(prev => !prev);
    };

    const handleToggleGps = () => {
        setGpsSource(prev => (prev === "LIVE" ? "SIMULATED" : "LIVE"));
    };

    return (
        <div className="dashboard-root">
            {/* Top Bar Header */}
            <header className="dashboard-header">
                <h1 className="header-title">RAKSHAK</h1>
                <div className="status-indicator">
                    <span className="status-dot pulsing-green"></span>
                    <span className="status-text uppercase font-bold text-green">ROAD SCAN ACTIVE</span>
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
                        error={hazardsError}
                    />
                </section>

                {/* Right Side UI HUD Analytics */}
                <section className="grid-cell right-cell">
                    <LivePerceptionPanel systemStatus={systemStatus} hazards={hazards} />
                    <ActiveHazardsPanel hazards={hazards} />
                    <StatsPanel 
                        roadHealth={roadHealth} 
                        hazards={hazards}
                        onReset={handleReset}
                        isSimulating={isSimulating}
                        onToggleSim={handleToggleSim}
                        gpsSource={gpsSource}
                        onToggleGps={handleToggleGps}
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
