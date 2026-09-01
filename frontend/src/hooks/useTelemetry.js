import { useState, useEffect, useRef } from "react";
import { API_BASE_URL } from "../config";
import demoRoute from "../../../data/samples/demo_route.json";

export default function useTelemetry(routeState, routeSpeed, onRouteCompleted) {
    const [currentCoords, setCurrentCoords] = useState(demoRoute[0]);
    const [warning, setWarning] = useState(null);
    const [currentMode, setCurrentMode] = useState("SYSTEM_DEMO");
    const [modeInfo, setModeInfo] = useState({
        mode: "SYSTEM_DEMO",
        camera_status: "N/A",
        connected: false,
        source_type: "file",
        ai_engine: "SIMULATED",
        input_type: "VIDEO",
        detection_source: "DEMO/SIMULATED",
        gps_source: "DEMO TELEMETRY",
        backend: "CONNECTED"
    });
    const [realInferenceResult, setRealInferenceResult] = useState(null);
    const [realInferenceLoading, setRealInferenceLoading] = useState(false);
    const [systemStatus, setSystemStatus] = useState({
        ai_engine: "SIMULATED",
        gps: "SIMULATED",
        backend: "OFFLINE",
        camera: "N/A",
        mode: "SYSTEM_DEMO",
        verification: "ACTIVE",
        map: "ACTIVE",
        warning_status: "ACTIVE"
    });
    const [roadHealth, setRoadHealth] = useState(null);
    const [hazards, setHazards] = useState([]);
    const [hazardsError, setHazardsError] = useState(null);
    const [liveGpsAvailable, setLiveGpsAvailable] = useState(false);

    const routeIndexRef = useRef(0);
    const audioContextRef = useRef(null);

    // Play alert alarm chime
    const playAlertSound = () => {
        try {
            if (!audioContextRef.current) {
                audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
            }
            const ctx = audioContextRef.current;
            if (ctx.state === "suspended") {
                ctx.resume();
            }
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.5);
        } catch (e) {
            console.error("Audio playback error:", e);
        }
    };

    // 0. Fetch initial mode from backend
    useEffect(() => {
        const fetchInitialMode = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/mode`);
                if (res.ok) {
                    const data = await res.json();
                    setCurrentMode(data.mode);
                    setModeInfo(data);
                }
            } catch (err) {
                console.error("Failed to fetch initial mode:", err);
            }
        };
        fetchInitialMode();
    }, []);

    // Poll backend status so a stream image load cannot falsely imply that the
    // smartphone camera is active (an MJPEG connection may stay open offline).
    useEffect(() => {
        const refreshModeStatus = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/camera/status`);
                if (res.ok) setModeInfo(await res.json());
            } catch { /* the telemetry status handles backend connectivity */ }
        };
        refreshModeStatus();
        const interval = setInterval(refreshModeStatus, 1500);
        return () => clearInterval(interval);
    }, []);

    // Switch mode API handler
    const switchMode = async (targetMode) => {
        try {
            const res = await fetch(`${API_BASE_URL}/mode`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: targetMode })
            });
            if (res.ok) {
                const data = await res.json();
                setCurrentMode(data.mode);
                setModeInfo(prev => ({
                    ...prev,
                    mode: data.mode,
                    camera_status: data.camera_status,
                    ai_engine: data.ai_mode,
                    input_type: data.input_type,
                    detection_source: data.detection_source,
                    gps_source: data.gps_mode
                }));
                // If switching to Option C, trigger real image inference automatically
                if (data.mode === "IMAGE_FALLBACK") {
                    triggerRealInference();
                }
            }
        } catch (err) {
            console.error("Failed to switch mode:", err);
        }
    };

    // Option C: Trigger real RDD2022 inference on sample_road11.jpg
    const triggerRealInference = async () => {
        setRealInferenceLoading(true);
        try {
            const association = routeIsRunning
                ? `?associate_demo_route=true&latitude=${currentCoords.latitude}&longitude=${currentCoords.longitude}`
                : "";
            const res = await fetch(`${API_BASE_URL}/demo/real-inference${association}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" }
            });
            if (res.ok) {
                const data = await res.json();
                setRealInferenceResult(data);
            }
        } catch (err) {
            console.error("Failed to run real image inference:", err);
        } finally {
            setRealInferenceLoading(false);
        }
    };

    const [uploadedInferenceResult, setUploadedInferenceResult] = useState(null);
    const [uploadedInferenceLoading, setUploadedInferenceLoading] = useState(false);
    const [uploadedInferenceError, setUploadedInferenceError] = useState(null);

    // Option D: Run real RDD2022 inference on uploaded image
    const runUploadedImageInference = async (file, threshold = 0.25) => {
        setUploadedInferenceLoading(true);
        setUploadedInferenceError(null);
        try {
            const formData = new FormData();
            formData.append("file", file);
            const association = routeIsRunning
                ? `&associate_demo_route=true&latitude=${currentCoords.latitude}&longitude=${currentCoords.longitude}`
                : "";
            const res = await fetch(`${API_BASE_URL}/upload/inference?threshold=${threshold}${association}`, {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            if (res.ok) {
                setUploadedInferenceResult(data);
                return data;
            } else {
                const errorMsg = data.detail || "Failed to process uploaded image.";
                setUploadedInferenceError(errorMsg);
                return null;
            }
        } catch (err) {
            console.error("Upload inference API error:", err);
            setUploadedInferenceError("Network error while connecting to inference server.");
            return null;
        } finally {
            setUploadedInferenceLoading(false);
        }
    };

    const routeGpsEnabled = routeState !== "STOPPED";
    const routeIsRunning = routeState === "RUNNING";
    const resetSimulatedRoute = () => {
        routeIndexRef.current = 0;
        setCurrentCoords(demoRoute[0]);
    };

    // Global route engine: independent of A/B/C/D and the detection source.
    useEffect(() => {
        if (!routeIsRunning) return;

        const interval = setInterval(() => {
            if (routeIndexRef.current >= demoRoute.length - 1) {
                onRouteCompleted?.();
                return;
            }
            routeIndexRef.current += 1;
            const nextCoords = demoRoute[routeIndexRef.current];
            setCurrentCoords(nextCoords);
        }, Math.max(250, 1000 / routeSpeed));

        return () => clearInterval(interval);
    }, [routeIsRunning, routeSpeed, onRouteCompleted]);

    // 2. Geolocation Watch Position (For GPS: LIVE mode)
    useEffect(() => {
        if (currentMode !== "LIVE_CAMERA" || routeGpsEnabled) {
            setLiveGpsAvailable(false);
            return;
        }

        let watchId = null;
        if (navigator.geolocation) {
            watchId = navigator.geolocation.watchPosition(
                (position) => {
                    setCurrentCoords({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude
                    });
                    setLiveGpsAvailable(true);
                },
                (err) => {
                    console.error("GPS LIVE error:", err);
                    setLiveGpsAvailable(false);
                },
                { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
            );
        }

        return () => {
            if (watchId !== null) {
                navigator.geolocation.clearWatch(watchId);
            }
        };
    }, [currentMode, routeGpsEnabled]);

    // 3. Telemetry Post & Geofence warning loop
    useEffect(() => {
        if (!currentCoords) return;

        const syncTelemetry = async () => {
            try {
                const gpsSource = routeIsRunning ? "DEMO_SIMULATED" : (liveGpsAvailable ? "LIVE" : "TEST_FALLBACK");
                const response = await fetch(`${API_BASE_URL}/telemetry`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        latitude: currentCoords.latitude,
                        longitude: currentCoords.longitude,
                        speed_kmh: routeIsRunning ? 40.0 * routeSpeed : 0.0,
                        imu_accel_z: 9.8,
                        gps_source: gpsSource
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    if (data.warning_active && !warning) {
                        playAlertSound();
                    }
                    
                    setWarning(data.warning);
                    setSystemStatus(data.system_status);
                }
            } catch (err) {
                console.error("Telemetry API synchronization error:", err);
                setSystemStatus(prev => ({ ...prev, backend: "OFFLINE" }));
            }
        };

        syncTelemetry();
        const interval = setInterval(syncTelemetry, 2000);
        return () => clearInterval(interval);
    }, [currentCoords, routeGpsEnabled, routeIsRunning, routeSpeed, currentMode, liveGpsAvailable]);

    // 4. Fetch Hazards list & Road Health index
    const refreshData = async () => {
        if (!currentCoords) return;
        try {
            const hazardsRes = await fetch(`${API_BASE_URL}/hazards`);
            if (hazardsRes.ok) {
                const hazardsData = await hazardsRes.json();
                console.log(`[MAP DATA RECEIVED] Fetched ${hazardsData.length} hazard(s) from backend`);
                setHazards(hazardsData);
                setHazardsError(null);
            } else {
                setHazardsError("Unable to load hazards");
            }

            const healthRes = await fetch(
                `${API_BASE_URL}/road-health?latitude=${currentCoords.latitude}&longitude=${currentCoords.longitude}&radius_meters=500`
            );
            if (healthRes.ok) {
                const healthData = await healthRes.json();
                setRoadHealth(healthData);
            }
        } catch (err) {
            console.error("Error fetching map hazards metadata:", err);
            setHazardsError("Unable to load hazards");
        }
    };

    // Refresh hazards list periodically
    useEffect(() => {
        refreshData();
        const interval = setInterval(refreshData, 1000);
        return () => clearInterval(interval);
    }, [currentCoords]);

    return {
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
        setUploadedInferenceResult,
        systemStatus,
        roadHealth,
        hazards,
        hazardsError,
        refreshData,
        resetSimulatedRoute,
        routeProgress: Math.round((routeIndexRef.current / Math.max(1, demoRoute.length - 1)) * 100)
    };
}
