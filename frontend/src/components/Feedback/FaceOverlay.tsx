import './FaceOverlay.css';

interface FaceOverlayProps {
    faceDetected?: boolean;
    // Landmarks will be added here later
}

export default function FaceOverlay({ faceDetected = false }: FaceOverlayProps) {
    if (!faceDetected) return null;

    return (
        <div className="face-overlay">
            {/* Placeholder for dynamic bounding box */}
            <div className="face-box">
                <span className="face-label">FACE DETECTED</span>
                <div className="scanning-line" />
            </div>

            {/* Future: Render SVG landmarks here */}
        </div>
    );
}
