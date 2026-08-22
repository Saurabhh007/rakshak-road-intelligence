import "./ActiveHazardsPanel.css";

export default function ActiveHazardsPanel({ hazards }) {
    // Sort hazards by timestamp desc
    const sortedHazards = [...hazards].sort(
        (a, b) => new Date(b.timestamp || b.last_detected) - new Date(a.timestamp || a.last_detected)
    );

    const formatTime = (timeStr) => {
        if (!timeStr) return "N/A";
        try {
            return new Date(timeStr).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch {
            return "N/A";
        }
    };

    return (
        <div className="card active-hazards-panel">
            <h2 className="panel-title">ACTIVE ROAD HAZARDS</h2>
            <div className="hazards-table-container">
                {sortedHazards.length === 0 ? (
                    <div className="no-hazards-message">
                        <span>NO ACTIVE HAZARDS DETECTED</span>
                    </div>
                ) : (
                    <table className="hazards-table">
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Severity</th>
                                <th>Confidence</th>
                                <th>Coordinates</th>
                                <th>Status</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedHazards.map((h) => {
                                const lat = typeof h.latitude === "number" ? h.latitude.toFixed(4) : "N/A";
                                const lng = typeof h.longitude === "number" ? h.longitude.toFixed(4) : "N/A";
                                return (
                                    <tr key={h.id}>
                                        <td className="hazard-type">{h.type}</td>
                                        <td>
                                            <span className={`severity-badge-inline severity-${h.severity}`}>
                                                {h.severity}
                                            </span>
                                        </td>
                                        <td className="hazard-conf">
                                            {(h.confidence * 100).toFixed(0)}%
                                        </td>
                                        <td className="hazard-coords">
                                            {lat}, {lng}
                                        </td>
                                        <td>
                                            <span className={`status-badge-inline status-${h.status.toLowerCase()}`}>
                                                {h.status}
                                            </span>
                                        </td>
                                        <td className="hazard-time">
                                            {formatTime(h.timestamp || h.last_detected)}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}
