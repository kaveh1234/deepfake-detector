import { useEffect, useState } from 'react';
import './Traps.css';

interface DiscoFlashProps {
    isActive: boolean;
    phase: 'red' | 'green' | 'blue'; // From backend
    progress?: number;
}

export default function DiscoFlash({ isActive, phase, progress }: DiscoFlashProps) {
    // If not active, don't render or render transparent
    if (!isActive) return null;

    // Map phase to color
    const colors: Record<string, string> = {
        red: 'rgba(255, 0, 0, 0.4)',
        green: 'rgba(0, 255, 0, 0.4)',
        blue: 'rgba(0, 0, 255, 0.4)',
    };

    const color = colors[phase] || 'transparent';

    return (
        <div
            className="disco-flash-overlay"
            style={{ backgroundColor: color }}
        >
            <div className="disco-message">
                HOLD STILL
            </div>
            {/* Optional progress bar for current color */}
            {progress !== undefined && (
                <div className="disco-progress-container">
                    <div
                        className="disco-progress-bar"
                        style={{ width: `${progress}%`, backgroundColor: phase }} // use vivid color for bar
                    />
                </div>
            )}
        </div>
    );
}
