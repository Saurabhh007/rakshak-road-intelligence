import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

export default function MapView({ hazards, currentCoords, routeCoordinates }) {
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
            // Assign color scheme based on severity level
            let color = "#E53E3E"; // Red for high
            if (h.severity === "medium") {
                color = "#DD6B20"; // Amber/Orange
            } else if (h.severity === "low") {
                color = "#D69E2E"; // Yellow
            }

            // Draw hazard circle marker
            const marker = L.circleMarker([h.latitude, h.longitude], {
                radius: 12,
                color: "#FFFFFF",
                fillColor: color,
                fillOpacity: 0.8,
                weight: 2
            });

            // Bind click popup with stats metadata details
            marker.bindPopup(`
                <div style="font-family: sans-serif; font-size: 13px;">
                    <b style="text-transform: uppercase; color: ${color};">${h.type} (${h.severity})</b><br/>
                    <b>Status:</b> ${h.status}<br/>
                    <b>Confidence:</b> ${(h.confidence * 100).toFixed(0)}%<br/>
                    <b>Last Sighted:</b> ${new Date(h.last_detected).toLocaleTimeString()}
                </div>
            `);

            hazardLayerGroup.current.addLayer(marker);
        });
    }, [hazards]);

    return (
        <div 
            ref={mapRef} 
            id="map" 
            style={{ 
                width: "100%", 
                height: "100%", 
                backgroundColor: "#111" 
            }} 
        />
    );
}
