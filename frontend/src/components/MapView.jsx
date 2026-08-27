import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export default function MapView({ hazards, currentCoords, routeCoordinates, error }) {
    const mapRef = useRef(null);
    const mapInstance = useRef(null);
    const vehicleMarker = useRef(null);
    const hazardLayerGroup = useRef(null);

    // Initialize Leaflet Map
    useEffect(() => {
        if (mapRef.current && !mapInstance.current) {
            // MG Road Bengaluru center
            const centerLat = currentCoords?.latitude || 12.9716;
            const centerLng = currentCoords?.longitude || 77.5946;

            const map = L.map(mapRef.current, {
                center: [centerLat, centerLng],
                zoom: 16,
                zoomControl: true
            });

            // Clean high-contrast style tiles
            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                attribution: '&copy; <a href="https://osm.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);

            mapInstance.current = map;
            hazardLayerGroup.current = L.layerGroup().addTo(map);

            // Draw route path line
            if (routeCoordinates && routeCoordinates.length > 0) {
                const latlngs = routeCoordinates.map(pt => [pt.latitude, pt.longitude]);
                L.polyline(latlngs, {
                    color: "#0078FF",
                    weight: 4,
                    opacity: 0.6,
                    dashArray: "8, 8"
                }).addTo(map);
            }
        }

        // Cleanup on unmount
        return () => {
            if (mapInstance.current) {
                mapInstance.current.remove();
                mapInstance.current = null;
            }
        };
    }, []);

    // Update Vehicle Location Marker
    useEffect(() => {
        if (!mapInstance.current || !currentCoords) return;

        const { latitude, longitude } = currentCoords;
        const latlng = [latitude, longitude];

        if (!vehicleMarker.current) {
            // Pulsing blue circle representing vehicle position
            vehicleMarker.current = L.circleMarker(latlng, {
                radius: 8,
                color: "#FFFFFF",
                fillColor: "#0078FF",
                fillOpacity: 0.9,
                weight: 2
            }).addTo(mapInstance.current);
            vehicleMarker.current.bindPopup("<b>My Vehicle</b>").openPopup();
        } else {
            vehicleMarker.current.setLatLng(latlng);
        }

        // Center map on vehicle as it moves
        mapInstance.current.panTo(latlng);
    }, [currentCoords]);

    // Render Hazard Circle Markers on Map
    useEffect(() => {
        if (!mapInstance.current || !hazardLayerGroup.current || !hazards) return;

        // Clear existing markers from group
        hazardLayerGroup.current.clearLayers();

        hazards.forEach(h => {
            // Assign color scheme based on severity level (aligning with RAKSHAK Core theme variables)
            let color = "#f56565"; // Red for high
            if (h.severity === "medium") {
                color = "#ed8936"; // Orange
            } else if (h.severity === "low") {
                color = "#48bb78"; // Green
            }

            // Draw hazard circle marker
            const marker = L.circleMarker([h.latitude, h.longitude], {
                radius: 12,
                color: "#FFFFFF",
                fillColor: color,
                fillOpacity: 0.8,
                weight: 2
            });

            // Format coordinates, date, and ID from actual backend data schema
            const formattedLat = typeof h.latitude === "number" ? h.latitude.toFixed(6) : "N/A";
            const formattedLng = typeof h.longitude === "number" ? h.longitude.toFixed(6) : "N/A";
            const formattedTime = h.timestamp ? new Date(h.timestamp).toLocaleTimeString() : "N/A";
            const formattedDate = h.timestamp ? new Date(h.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : "N/A";
            const hazardId = `PTH-${String(h.id).padStart(3, '0')}`;

            // Bind click popup with stats metadata details structured as a dark HUD card
            marker.bindPopup(`
                <div class="custom-map-popup">
                    <div class="popup-title" style="color: ${color};">
                        ${h.type.toUpperCase()} #${hazardId}
                    </div>
                    <div class="popup-field">
                        <span class="popup-label">Severity:</span> 
                        <span class="popup-value" style="color: ${color}; font-weight: 800;">${h.severity.toUpperCase()}</span>
                    </div>
                    <div class="popup-field">
                        <span class="popup-label">Confidence:</span> 
                        <span class="popup-value highlight-blue">${(h.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div class="popup-field">
                        <span class="popup-label">Location:</span> 
                        <span class="popup-value font-mono">${formattedLat}, ${formattedLng}</span>
                    </div>
                    <div class="popup-field">
                        <span class="popup-label">Status:</span> 
                        <span class="popup-value status-active">${h.status}</span>
                    </div>
                    <div class="popup-field">
                        <span class="popup-label">Detected:</span> 
                        <span class="popup-value">${formattedDate} ${formattedTime}</span>
                    </div>
                </div>
            `);

            hazardLayerGroup.current.addLayer(marker);
        });
    }, [hazards]);

    return (
        <div style={{ position: "relative", width: "100%", height: "100%" }}>
            <div 
                ref={mapRef} 
                id="map" 
                style={{ 
                    width: "100%", 
                    height: "100%", 
                    backgroundColor: "#111" 
                }} 
            />
            {error && (
                <div className="map-error-overlay">
                    <span className="overlay-icon">⚠️</span>
                    <span>{error}</span>
                </div>
            )}
            {!error && hazards && hazards.length === 0 && (
                <div className="map-empty-overlay">
                    <span className="overlay-icon">ℹ</span>
                    <span>No active hazards detected</span>
                </div>
            )}
        </div>
    );
}
