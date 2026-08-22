import { useState } from "react";
import { API_BASE_URL } from "../config";

export default function LivePerceptionPanel({ systemStatus, hazards = [] }) {
    const aiEngine = systemStatus?.ai_engine || "SIMULATED";
    const gpsMode = systemStatus?.gps || "SIMULATED";
    const backendOnline = systemStatus?.backend === "CONNECTED";

    const [streamStatus, setStreamStatus] = useState("CONNECTING");

    // Filter and sort hazards to find recent YOLO detections
    const recentDetections = [...hazards]
        .sort((a, b) => new Date(b.timestamp || b.last_detected) - new Date(a.timestamp || a.last_detected))
        .slice(0, 3);

    const formatTime = (timeStr) => {
        if (!timeStr) return "N/A";
        try {
            return new Date(timeStr).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch {
            return "N/A";
        }
    };

    return (
        <div className="card perception-panel">
            <div className="panel-header-row">
                <h2 className="panel-title">ROAD CAMERA FEED</h2>
                <div className="stream-status-indicators">
                    {streamStatus === "CONNECTING" && (
                        <span className="badge-stream status-connecting">CONNECTING...</span>
                    )}
                    {streamStatus === "ACTIVE" && (
                        <span className="badge-stream status-active-stream">LIVE / STREAM ACTIVE</span>
                    )}
                    {streamStatus === "OFFLINE" && (
                        <span className="badge-stream status-offline">CAMERA OFFLINE</span>
                    )}
                    {aiEngine === "REAL" && backendOnline && (
                        <span className="badge-yolo-real">REAL YOLO INFERENCE</span>
                    )}
                </div>
            </div>
            
            {/* Live OpenCV streaming JPEG output from FastAPI */}
            <div className="camera-feed-container">
                <img 
                    src={`${API_BASE_URL}/stream/video?t=${Date.now()}`} 
                    alt="Road Camera Live Feed"
                    className="camera-image"
                    style={{ display: streamStatus === "ACTIVE" ? "block" : "none" }}
                    onError={() => setStreamStatus("OFFLINE")}
                    onLoad={() => setStreamStatus("ACTIVE")}
                />
                {streamStatus === "CONNECTING" && (
                    <div className="camera-fallback">
                        <span className="pulsing-text">CONNECTING TO VIDEO STREAM...</span>
                    </div>
                )}
                {streamStatus === "OFFLINE" && (
                    <div className="camera-fallback error-fallback">
                        <span>CAMERA FEED OFFLINE</span>
                    </div>
                )}
            </div>

            {/* Hardware Sensor & System Status Flags */}
            <div className="status-grid">
                <div className="status-item">
                    <span className="status-label">AI ENGINE:</span>
                    <span className={`status-value ${aiEngine === "REAL" ? "text-green" : "text-yellow"}`}>
                        {aiEngine === "REAL" ? "🟢 REAL" : "🟡 SIMULATED"}
                    </span>
                </div>
                <div className="status-item">
                    <span className="status-label">GPS:</span>
                    <span className={`status-value ${gpsMode === "LIVE" ? "text-green" : "text-yellow"}`}>
                        {gpsMode === "LIVE" ? "🟢 LIVE" : "🟡 SIMULATED"}
                    </span>
                </div>
                <div className="status-item">
                    <span className="status-label">BACKEND:</span>
                    <span className={`status-value ${backendOnline ? "text-green" : "text-red"}`}>
                        {backendOnline ? "🟢 CONNECTED" : "🔴 OFFLINE"}
                    </span>
                </div>
            </div>

            {/* Real-Time Detection Feed list */}
            <div className="recent-detections-section">
                <h3 className="section-subtitle">RECENT AI DETECTIONS</h3>
                <div className="recent-detections-list">
                    {recentDetections.length === 0 ? (
                        <div className="no-detections-placeholder">
                            <span>SURFACE SENSING IN PROGRESS...</span>
                        </div>
                    ) : (
                        recentDetections.map((det) => (
                            <div key={det.id} className="recent-detection-item">
                                <div className="det-time-type">
                                    <span className="det-time">[{formatTime(det.timestamp || det.last_detected)}]</span>
                                    <span className="det-type">{det.type}</span>
                                </div>
                                <div className="det-meta">
                                    <span className="det-conf">{(det.confidence * 100).toFixed(0)}%</span>
                                    <span className={`det-severity severity-badge-inline severity-${det.severity}`}>
                                        {det.severity}
                                    </span>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}
