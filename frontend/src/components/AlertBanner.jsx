export default function AlertBanner({ warning }) {
    if (warning) {
        // Warning flashing banner (High, Medium, Low severity styling)
        return (
            <div className={`alert-banner alert-active alert-${warning.severity}`}>
                <span className="alert-icon">⚠️</span>
                <span className="alert-text">
                    {warning.type.toUpperCase()} {warning.distance_meters}m AHEAD — REDUCE SPEED
                </span>
            </div>
        );
    }

    // Default monitoring status banner
    return (
        <div className="alert-banner alert-clear">
            <span className="alert-icon">🟢</span>
            <span className="alert-text">
                ROAD SCAN ACTIVE — SURFACE CLEAR IN GEOPROXIMITY
            </span>
        </div>
    );
}
