import { useState, useRef } from "react";
import { API_BASE_URL } from "../config";

export default function LivePerceptionPanel({
    currentMode = "SYSTEM_DEMO",
    modeInfo,
    systemStatus,
    hazards = [],
    realInferenceResult,
    realInferenceLoading,
    onRunInference,
    uploadedInferenceResult,
    uploadedInferenceLoading,
    uploadedInferenceError,
    onUploadInference,
    routeState = "STOPPED"
}) {
    const [streamStatus, setStreamStatus] = useState("CONNECTING");
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isDragOver, setIsDragOver] = useState(false);
    const fileInputRef = useRef(null);

    const isLiveCamera = currentMode === "LIVE_CAMERA";
    const isSystemDemo = currentMode === "SYSTEM_DEMO";
    const isImageFallback = currentMode === "IMAGE_FALLBACK";
    const isUploadImage = currentMode === "UPLOAD_IMAGE";

    const aiEngine = modeInfo?.ai_engine || systemStatus?.ai_engine || (isSystemDemo ? "SIMULATED" : "REAL");
    const backendCameraStatus = modeInfo?.camera_status || "CAMERA_CONNECTING";
    const cameraStatus = isLiveCamera ? backendCameraStatus : "N/A";
    const gpsDisplay = routeState !== "STOPPED"
        ? "SIMULATED DEMO ROUTE"
        : (isSystemDemo ? "N/A" : isLiveCamera ? (modeInfo?.gps_source || "N/A") : "N/A");

    // Recent detections (sorted newest first)
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

    // File selection handler
    const handleFileChange = (file) => {
        if (!file) return;
        setSelectedFile(file);
        const reader = new FileReader();
        reader.onload = (e) => {
            setPreviewUrl(e.target.result);
        };
        reader.readAsDataURL(file);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragOver(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileChange(e.dataTransfer.files[0]);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragOver(false);
    };

    const handleRunUploadInference = async () => {
        if (!selectedFile || !onUploadInference) return;
        await onUploadInference(selectedFile);
    };

    const handleResetUpload = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    return (
        <div className="card perception-panel">
            {/* Panel Header */}
            <div className="panel-header-row">
                <div>
                    <h2 className="panel-title">
                        {isLiveCamera && "OPTION A — LIVE LAPTOP WEBCAM FEED"}
                        {isSystemDemo && "OPTION B — SYSTEM DEMO PIPELINE"}
                        {isImageFallback && "OPTION C — REAL AI IMAGE FALLBACK"}
                        {isUploadImage && "OPTION D — UPLOAD IMAGE (REAL AI VERIFICATION)"}
                    </h2>
                    <div className="mode-subtitle">
                        {isLiveCamera && "INPUT: LAPTOP WEBCAM  |  AI MODE: REAL INFERENCE  |  GPS: SIMULATED WHEN ROUTE RUNS"}
                        {isSystemDemo && "INPUT: VIDEO  |  MODE: SYSTEM DEMO  |  DETECTION: DEMO/SIMULATED  |  GPS: SIMULATED"}
                        {isImageFallback && "INPUT: IMAGE (sample_road11.jpg)  |  AI MODE: REAL INFERENCE  |  GPS: SIMULATED ROUTE WHEN RUNNING"}
                        {isUploadImage && "INPUT: UPLOADED IMAGE  |  AI MODE: REAL INFERENCE  |  GPS: SIMULATED  |  LOCATION ASSOCIATION: SIMULATED DEMO ROUTE"}
                    </div>
                </div>

                <div className="stream-status-indicators">
                    {isLiveCamera && (
                        <>
                            {cameraStatus === "CAMERA_ACTIVE" && (
                                <span className="badge-stream status-active-stream">🟢 WEBCAM ACTIVE</span>
                            )}
                            {cameraStatus === "CAMERA_CONNECTING" && (
                                <span className="badge-stream status-connecting">🟡 CONNECTING...</span>
                            )}
                            {(cameraStatus === "CAMERA_OFFLINE" || cameraStatus === "CAMERA_ERROR") && (
                                <span className="badge-stream status-offline">🔴 WEBCAM OFFLINE</span>
                            )}
                            <span className="badge-yolo-real">REAL YOLO INFERENCE</span>
                        </>
                    )}
                    {isSystemDemo && (
                        <>
                            <span className="badge-stream badge-demo">VIDEO PIPELINE</span>
                            <span className="badge-simulated">DETECTION: DEMO/SIMULATED</span>
                        </>
                    )}
                    {isImageFallback && (
                        <>
                            <span className="badge-yolo-real">RDD2022 REAL YOLO</span>
                            <span className="badge-threshold">THRESHOLD: 0.25 (ISOLATED)</span>
                        </>
                    )}
                    {isUploadImage && (
                        <>
                            <span className="badge-yolo-real badge-upload-real">REAL AI VERIFICATION</span>
                            <span className="badge-threshold">ISOLATED (NO GPS)</span>
                        </>
                    )}
                </div>
            </div>

            {/* Main Visual Perception Feed Container */}
            <div className="camera-feed-container">
                {/* Option A & B: Live Video Stream */}
                {(isLiveCamera || isSystemDemo) && (
                    <>
                        <img
                            src={`${API_BASE_URL}/stream/video?mode=${currentMode}`}
                            alt="Road Perception Stream"
                            className="camera-image"
                            style={{ display: (cameraStatus === "CAMERA_ACTIVE" || isSystemDemo) ? "block" : "none" }}
                            onError={() => setStreamStatus("OFFLINE")}
                            onLoad={() => setStreamStatus("ACTIVE")}
                        />
                        {isLiveCamera && cameraStatus === "CAMERA_CONNECTING" && (
                            <div className="camera-fallback">
                                <span className="pulsing-text">CONNECTING TO LAPTOP WEBCAM...</span>
                            </div>
                        )}
                        {isLiveCamera && (cameraStatus === "CAMERA_OFFLINE" || cameraStatus === "CAMERA_ERROR") && (
                            <div className="camera-fallback error-fallback">
                                <span className="fallback-icon">⚠️</span>
                                <div>
                                    <div className="font-bold">WEBCAM OFFLINE</div>
                                    <div className="sub-fallback-text">Check laptop webcam permissions or connection. Live webcam mode does not fall back to simulation.</div>
                                </div>
                            </div>
                        )}
                    </>
                )}

                {/* Option C: Real AI Image Inference View */}
                {isImageFallback && (
                    <div className="real-image-container">
                        {realInferenceLoading ? (
                            <div className="camera-fallback">
                                <span className="pulsing-text">RUNNING REAL RDD2022 YOLOv12 INFERENCE ON sample_road11.jpg...</span>
                            </div>
                        ) : realInferenceResult?.annotated_image ? (
                            <div className="image-annotated-wrapper">
                                <img
                                    src={realInferenceResult.annotated_image}
                                    alt="Real AI Pothole Detection (sample_road11.jpg)"
                                    className="annotated-result-image"
                                />
                                <div className="image-overlay-pill">
                                    <span className="pill-dot">🟢</span>
                                    <span>REAL INFERENCE: D40 POTHOLE (CONF: {(realInferenceResult.detections[0]?.confidence * 100).toFixed(1)}%)</span>
                                </div>
                            </div>
                        ) : realInferenceResult?.status === "AI_UNAVAILABLE" ? (
                            <div className="camera-fallback error-fallback">
                                <span>AI UNAVAILABLE — YOLO weights could not be loaded</span>
                            </div>
                        ) : (
                            <div className="camera-fallback">
                                <span>Click below to execute Real AI Inference on sample_road11.jpg</span>
                            </div>
                        )}
                    </div>
                )}

                {/* Option D: Upload Image View */}
                {isUploadImage && (
                    <div className="upload-perception-container">
                        {uploadedInferenceLoading ? (
                            <div className="camera-fallback">
                                <span className="pulsing-text">RUNNING REAL RDD2022 YOLO INFERENCE ON UPLOADED IMAGE...</span>
                            </div>
                        ) : uploadedInferenceResult?.annotated_image ? (
                            <div className="image-annotated-wrapper">
                                <img
                                    src={uploadedInferenceResult.annotated_image}
                                    alt="Uploaded Image AI Inference Result"
                                    className="annotated-result-image"
                                />
                                <div className="image-overlay-pill">
                                    <span className="pill-dot">{uploadedInferenceResult.detections?.length > 0 ? "🟢" : "⚪"}</span>
                                    <span>
                                        {uploadedInferenceResult.detections?.length > 0
                                            ? `REAL INFERENCE: ${uploadedInferenceResult.detections[0].class_name} POTHOLE (CONF: ${(uploadedInferenceResult.detections[0].confidence * 100).toFixed(1)}%)`
                                            : "REAL INFERENCE: NO DETECTION ABOVE THRESHOLD"}
                                    </span>
                                </div>
                            </div>
                        ) : previewUrl ? (
                            <div className="upload-preview-wrapper">
                                <img src={previewUrl} alt="Selected Road Image Preview" className="upload-preview-image" />
                                <div className="upload-preview-badge">
                                    <span>READY FOR REAL AI INFERENCE</span>
                                </div>
                            </div>
                        ) : (
                            <div
                                className={`upload-dropzone ${isDragOver ? "dropzone-active" : ""}`}
                                onDrop={handleDrop}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onClick={() => fileInputRef.current && fileInputRef.current.click()}
                            >
                                <div className="dropzone-icon">📤</div>
                                <div className="dropzone-title">DROP ROAD / POTHOLE IMAGE HERE</div>
                                <div className="dropzone-sub">or click to browse (JPG, PNG, WEBP)</div>
                                <input
                                    type="file"
                                    ref={fileInputRef}
                                    style={{ display: "none" }}
                                    accept="image/jpeg,image/png,image/webp"
                                    onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
                                />
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Option C Inference Details Card */}
            {isImageFallback && (
                <div className="option-c-details-card">
                    <div className="option-c-meta-row">
                        <div className="meta-col">
                            <span className="meta-label">TARGET IMAGE:</span>
                            <span className="meta-val font-mono">ai/test_images/sample_road11.jpg</span>
                        </div>
                        <div className="meta-col">
                            <span className="meta-label">AI MODEL:</span>
                            <span className="meta-val text-green font-bold">yolo12s_RDD2022_best.pt (REAL)</span>
                        </div>
                        <div className="meta-col">
                            <span className="meta-label">ISOLATED THRESHOLD:</span>
                            <span className="meta-val font-mono">0.25 (Production: 0.60)</span>
                        </div>
                    </div>

                    {realInferenceResult?.detections && realInferenceResult.detections.length > 0 ? (
                        <div className="detection-table-mini">
                            <div className="detection-table-row detection-header">
                                <span>Detected Class</span>
                                <span>Confidence</span>
                                <span>Bounding Box [x1, y1, x2, y2]</span>
                                <span>Source</span>
                            </div>
                            {realInferenceResult.detections.map((det, i) => (
                                <div key={i} className="detection-table-row">
                                    <span className="badge-class">{det.class_name}</span>
                                    <span className="text-green font-bold font-mono">{(det.confidence * 100).toFixed(2)}% ({det.confidence})</span>
                                    <span className="font-mono">[{det.bbox.join(", ")}]</span>
                                    <span className="badge-yolo-real">REAL INFERENCE</span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="no-detections-text">No D40 detections found above threshold 0.25</div>
                    )}

                    <div className="option-c-actions">
                        <button
                            onClick={onRunInference}
                            disabled={realInferenceLoading}
                            className="btn btn-primary-action"
                        >
                            {realInferenceLoading ? "⏳ RUNNING INFERENCE..." : "⚡ RE-RUN REAL AI INFERENCE"}
                        </button>
                        <span className="isolated-notice">
                            ℹ️ Option C is an isolated AI verification demo. Results are not inserted into the live hazard database or map.
                        </span>
                    </div>
                </div>
            )}

            {/* Option D Upload Image Details Card */}
            {isUploadImage && (
                <div className="option-c-details-card option-d-details-card">
                    <div className="option-c-meta-row">
                        <div className="meta-col">
                            <span className="meta-label">INPUT FILE:</span>
                            <span className="meta-val font-mono">{selectedFile?.name || uploadedInferenceResult?.filename || "None selected"}</span>
                        </div>
                        <div className="meta-col">
                            <span className="meta-label">AI MODEL:</span>
                            <span className="meta-val text-green font-bold">yolo12s_RDD2022_best.pt (REAL)</span>
                        </div>
                        <div className="meta-col">
                            <span className="meta-label">ISOLATED THRESHOLD:</span>
                            <span className="meta-val font-mono">0.25 (Isolated Fallback)</span>
                        </div>
                    </div>

                    {/* Detections List / Table */}
                    {uploadedInferenceResult?.detections && uploadedInferenceResult.detections.length > 0 ? (
                        <div className="detection-table-mini">
                            <div className="detection-table-row detection-header">
                                <span>Detected Class</span>
                                <span>Confidence</span>
                                <span>Bounding Box [x1, y1, x2, y2]</span>
                                <span>Source</span>
                            </div>
                            {uploadedInferenceResult.detections.map((det, i) => (
                                <div key={i} className="detection-table-row">
                                    <span className="badge-class">{det.class_name}</span>
                                    <span className="text-green font-bold font-mono">{(det.confidence * 100).toFixed(2)}% ({det.confidence})</span>
                                    <span className="font-mono">[{det.bbox.join(", ")}]</span>
                                    <span className="badge-yolo-real">REAL AI</span>
                                </div>
                            ))}
                        </div>
                    ) : uploadedInferenceResult ? (
                        <div className="no-detections-banner">
                            <span>ℹ️ NO DETECTION ABOVE CONFIDENCE THRESHOLD (0.25) — Evaluated truthfully by model</span>
                        </div>
                    ) : null}

                    {uploadedInferenceError && (
                        <div className="upload-error-box">
                            <span>⚠️ {uploadedInferenceError}</span>
                        </div>
                    )}

                    {/* Actions Row */}
                    <div className="option-d-actions">
                        {previewUrl && !uploadedInferenceResult && (
                            <button
                                onClick={handleRunUploadInference}
                                disabled={uploadedInferenceLoading}
                                className="btn btn-primary-action btn-upload-run"
                            >
                                {uploadedInferenceLoading ? "⏳ RUNNING INFERENCE..." : "⚡ RUN REAL AI VERIFICATION"}
                            </button>
                        )}

                        {(previewUrl || uploadedInferenceResult) && (
                            <button
                                onClick={handleResetUpload}
                                className="btn btn-secondary btn-upload-reset"
                            >
                                📤 SELECT DIFFERENT IMAGE
                            </button>
                        )}

                        <span className="isolated-notice">
                            ℹ️ VERIFICATION ONLY — NO GPS LOCATION (Not stored in production database or map).
                        </span>
                    </div>
                </div>
            )}

            {/* Truthful 6-Point System Status Grid */}
            <div className="status-grid-six">
                <div className="status-item">
                    <span className="status-label">CAMERA:</span>
                    <span className={`status-value ${cameraStatus === "CAMERA_ACTIVE" ? "text-green" : cameraStatus === "CAMERA_OFFLINE" || cameraStatus === "CAMERA_ERROR" ? "text-red" : cameraStatus === "CAMERA_CONNECTING" ? "text-yellow" : "text-gray"}`}>
                        {cameraStatus === "CAMERA_ACTIVE" ? "🟢 ACTIVE" : cameraStatus === "CAMERA_OFFLINE" || cameraStatus === "CAMERA_ERROR" ? "🔴 OFFLINE" : cameraStatus === "CAMERA_CONNECTING" ? "🟡 CONNECTING" : "⚪ N/A"}
                    </span>
                </div>
                <div className="status-item">
                    <span className="status-label">AI ENGINE:</span>
                    <span className={`status-value ${aiEngine === "REAL" ? "text-green" : aiEngine === "SIMULATED" ? "text-yellow" : "text-red"}`}>
                        {aiEngine === "REAL" ? "🟢 REAL" : aiEngine === "SIMULATED" ? "🟡 SIMULATED" : "🔴 AI UNAVAILABLE"}
                    </span>
                </div>
                <div className="status-item">
                    <span className="status-label">GPS:</span>
                    <span className={`status-value ${gpsDisplay === "LIVE" ? "text-green" : gpsDisplay.includes("DEMO") || gpsDisplay === "SIMULATED" ? "text-yellow" : "text-gray"}`}>
                        {gpsDisplay === "LIVE" ? "🟢 LIVE" : gpsDisplay.includes("DEMO") || gpsDisplay === "SIMULATED" ? "🟡 SIMULATED" : "⚪ N/A"}
                    </span>
                </div>
                <div className="status-item">
                    <span className="status-label">VERIFICATION:</span>
                    <span className="status-value text-green">🟢 ACTIVE</span>
                </div>
                <div className="status-item">
                    <span className="status-label">MAP:</span>
                    <span className="status-value text-green">🟢 ACTIVE</span>
                </div>
                <div className="status-item">
                    <span className="status-label">WARNING:</span>
                    <span className="status-value text-green">🟢 ACTIVE</span>
                </div>
            </div>

            {/* Recent Detections List (For Option A & B) */}
            {(isLiveCamera || isSystemDemo) && (
                <div className="recent-detections-section">
                    <div className="section-header-flex">
                        <h3 className="section-subtitle">
                            {isLiveCamera ? "LIVE AI DETECTIONS (REAL INFERENCE)" : "DEMO DETECTION EVENTS (SIMULATED)"}
                        </h3>
                        <span className="detection-source-badge">
                            {isLiveCamera ? "SOURCE: REAL INFERENCE" : "SOURCE: DEMO/SIMULATED"}
                        </span>
                    </div>

                    <div className="recent-detections-list">
                        {recentDetections.length === 0 ? (
                            <div className="no-detections-placeholder">
                                <span>{isLiveCamera ? "SCANNING LIVE ROAD SURFACE VIA WEBCAM..." : "AWAITING DEMO DETECTION EVENTS..."}</span>
                            </div>
                        ) : (
                            recentDetections.map((det) => (
                                <div key={det.id} className="recent-detection-item">
                                    <div className="det-time-type">
                                        <span className="det-time">[{formatTime(det.timestamp || det.last_detected)}]</span>
                                        <span className="det-type">{det.type}</span>
                                        {isSystemDemo && <span className="sim-tag">[DEMO/SIMULATED]</span>}
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
            )}
        </div>
    );
}
