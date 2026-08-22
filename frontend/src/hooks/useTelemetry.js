import { useState, useEffect, useRef } from "react";
import { API_BASE_URL } from "../config";
import demoRoute from "../../../data/samples/demo_route.json";

export default function useTelemetry(isSimulating, gpsSource) {
    const [currentCoords, setCurrentCoords] = useState(demoRoute[0]);
    const [warning, setWarning] = useState(null);
    const [systemStatus, setSystemStatus] = useState({
        ai_engine: "SIMULATED",
        gps: "SIMULATED",
        backend: "OFFLINE"
    });
    const [roadHealth, setRoadHealth] = useState(null);
    const [hazards, setHazards] = useState([]);

    const routeIndexRef = useRef(0);
    const audioContextRef = useRef(null);

    // Play alert alarm chime
    const playAlertSound = () => {
        try {
            // Lazy-init audio context on user interaction
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
            osc.frequency.setValueAtTime(880, ctx.currentTime); // High pitch alarm tone
            
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5); // Decay over 500ms
            
            osc.connect(gain);
            gain.connect(ctx.destination);
            
            osc.start();
            osc.stop(ctx.currentTime + 0.5);
        } catch (e) {
            console.error("Audio playback error:", e);
        }
    };

    // 1. Simulation Coordinates Loop
    useEffect(() => {
        if (!isSimulating || gpsSource === "LIVE") return;

        const interval = setInterval(() => {
            routeIndexRef.current = (routeIndexRef.current + 1) % demoRoute.length;
            const nextCoords = demoRoute[routeIndexRef.current];
            setCurrentCoords(nextCoords);
        }, 1000); // Step every 1 second

        return () => clearInterval(interval);
    }, [isSimulating, gpsSource]);

    // 2. Geolocation Watch Position (For GPS: LIVE mode)
    useEffect(() => {
        if (gpsSource !== "LIVE") return;

        let watchId = null;
        if (navigator.geolocation) {
            watchId = navigator.geolocation.watchPosition(
                (position) => {
                    setCurrentCoords({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude
                    });
                },
                (err) => {
                    console.error("GPS LIVE error:", err);
                },
                { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
            );
        }

        return () => {
            if (watchId !== null) {
                navigator.geolocation.clearWatch(watchId);
            }
        };
    }, [gpsSource]);

    // 3. Telemetry Post & Geofence warning loop
    useEffect(() => {
        if (!currentCoords) return;

        const syncTelemetry = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/telemetry`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        latitude: currentCoords.latitude,
                        longitude: currentCoords.longitude,
                        speed_kmh: isSimulating ? 40.0 : 0.0,
                        imu_accel_z: 9.8,
                        gps_source: gpsSource
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    
                    // Trigger sound chime if alert is activated for the first time
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
    }, [currentCoords, gpsSource, isSimulating]);

    // 4. Fetch Hazards list & Road Health index
    const refreshData = async () => {
        if (!currentCoords) return;
        try {
            // Fetch database hazards markers
            const hazardsRes = await fetch(`${API_BASE_URL}/hazards`);
            if (hazardsRes.ok) {
                const hazardsData = await hazardsRes.json();
                setHazards(hazardsData);
            }

            // Fetch road health score calculations
            const healthRes = await fetch(
                `${API_BASE_URL}/road-health?latitude=${currentCoords.latitude}&longitude=${currentCoords.longitude}&radius_meters=500`
            );
            if (healthRes.ok) {
                const healthData = await healthRes.json();
                setRoadHealth(healthData);
            }
        } catch (err) {
            console.error("Error fetching map hazards metadata:", err);
        }
    };

    // Refresh hazards list periodically (every 1 second to update dynamically)
    useEffect(() => {
        refreshData();
        const interval = setInterval(refreshData, 1000);
        return () => clearInterval(interval);
    }, [currentCoords]);

    return {
        currentCoords,
        warning,
        systemStatus,
        roadHealth,
        hazards,
        refreshData
    };
}
