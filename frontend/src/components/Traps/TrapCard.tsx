import { TrapType, TrapStatus } from '../../types';
import './Traps.css';

interface TrapCardProps {
    type: TrapType;
    status: TrapStatus;
    isActive: boolean;
}

const TRAP_INSTRUCTIONS: Record<TrapType, string> = {
    fisheye: "Move your face VERY CLOSE to the camera until the bar fills up.",
    squint: "SCRUNCH your nose and maintain the expression.",
    disco: "Hold STILL while the screen flashes colors.",
};

export default function TrapCard({ type, status, isActive }: TrapCardProps) {
    const instructions = TRAP_INSTRUCTIONS[type];

    // Determine class based on status/active
    let className = "trap-card";
    if (isActive) className += " active";
    if (status === 'passed') className += " passed";
    if (status === 'failed') className += " failed";

    return (
        <div className={className}>
            <h3 className="trap-title">
                {type} {status === 'passed' && '✓'} {status === 'failed' && '✗'}
            </h3>
            <p className="trap-instructions">
                {instructions}
            </p>
            {isActive && (
                <div className="animate-pulse text-xs text-blue-400 mt-2">
                    IN PROGRESS...
                </div>
            )}
        </div>
    );
}
