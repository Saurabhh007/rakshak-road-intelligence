export default function StatsPanel({ roadHealth, hazards = [], onReset, isSimulating, onToggleSim, gpsSource, onToggleGps }) {
    const score = roadHealth?.prototype_road_health_score ?? 100.0;
    const totalHazards = roadHealth?.active_hazards ?? 0;
    const breakdown = roadHealth?.breakdown || { high_severity: 0, medium_severity: 0, low_severity: 0 };
    
    // Compute unique damage categories
    const categories = Array.from(new Set(hazards.map(h => h.type.toUpperCase())));
    const categoriesStr = categories.length > 0 ? categories.join(", ") : "NONE";

    // Compute latest detection
    const latestDet = hazards.length > 0
        ? [...hazards].sort((a, b) => new Date(b.timestamp || b.last_detected) - new Date(a.timestamp || a.last_detected))[0]
        : null;

    const formatTime = (timeStr) => {
        if (!timeStr) return "";
        try {
            return new Date(timeStr).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch {
            return "";
        }
    };

    // Choose status indicator based on health score range
    let healthColor = "text-green";
    if (score < 60) {
        healthColor = "text-red";
    } else if (score < 85) {
        healthColor = "text-yellow";
    }

    return (
        <div className="card stats-panel">
            <h2 className="panel-title">ROAD ANALYTICS</h2>

            {/* Big Health Score visual */}
            <div className="road-health-score-container">
                <span className="road-health-label">PROTOTYPE ROAD HEALTH</span>
                <span className={`road-health-value ${healthColor}`}>
                    {score.toFixed(0)} <span className="health-total">/ 100</span>
                </span>
                <span className="engineering-disclaimer">{roadHealth?.notice || "PROTOTYPE ONLY - NOT FOR ENGINEERING USE"}</span>
            </div>

            {/* Active Hazard Statistics list */}
            <div className="hazard-stats-container">
                <div className="active-hazards-header">
                    <span>ACTIVE HAZARDS:</span>
                    <span className="hazards-total">{totalHazards}</span>
                </div>
                <div className="severity-counts">
                    <div className="severity-row">
                        <span className="severity-badge high-badge">🔴 High</span>
                        <span className="severity-count">{breakdown.high_severity}</span>
                    </div>
                    <div className="severity-row">
                        <span className="severity-badge med-badge">🟠 Medium</span>
                        <span className="severity-count">{breakdown.medium_severity}</span>
                    </div>
                    <div className="severity-row">
                        <span className="severity-badge low-badge">🟡 Low</span>
                        <span className="severity-count">{breakdown.low_severity}</span>
                    </div>
                </div>
            </div>

            {/* Detection Summary Details */}
            <div className="detection-summary-container">
                <div className="summary-row">
                    <span className="summary-label">CATEGORIES:</span>
                    <span className="summary-value">{categoriesStr}</span>
                </div>
                {latestDet && (
                    <div className="summary-row">
                        <span className="summary-label">LATEST SIGHTING:</span>
                        <span className="summary-value highlight-text">
                            {latestDet.type.toUpperCase()} ({latestDet.severity.toUpperCase()}) @ {formatTime(latestDet.timestamp || latestDet.last_detected)}
                        </span>
                    </div>
                )}
            </div>

            {/* Dashboard Control actions */}
            <div className="controls-container">
                <button 
                    onClick={onToggleSim} 
                    className={`btn ${isSimulating ? "btn-stop" : "btn-start"}`}
                >
                    {isSimulating ? "⏹️ STOP SIMULATION" : "▶️ START DRIVE SIM"}
                </button>
                <button 
                    onClick={onToggleGps} 
                    className="btn btn-secondary"
                >
                    GPS SOURCE: {gpsSource === "LIVE" ? "🟢 LIVE" : "🟡 SIMULATED"}
                </button>
                <button 
                    onClick={onReset} 
                    className="btn btn-danger"
                >
                    🔄 RESET SYSTEM DATA
                </button>
            </div>
        </div>
    );
}
