import { TrapType, Metrics } from '../../types';
import TrapCard from './TrapCard';
import MetricChart from './MetricChart';
import './Traps.css';

interface TrapManagerProps {
    currentTrap: TrapType | null;
    trapStatus: 'pending' | 'running' | 'passed' | 'failed';
    metrics: Metrics | null;
    onStartTrap: (trap: TrapType) => void;
}

export default function TrapManager({ currentTrap, trapStatus, metrics }: TrapManagerProps) {
    if (!currentTrap) return (
        <div className="trap-manager-waiting">
            Waiting for session start...
        </div>
    );

    const isRunning = trapStatus === 'running';

    return (
        <div className="trap-manager-container">
            <TrapCard
                type={currentTrap}
                status={trapStatus}
                isActive={isRunning}
            />

            {isRunning && metrics && (
                <div className="trap-visuals">
                    {/* Visuals specific to trap */}
                    {currentTrap === 'fisheye' && (
                        <MetricChart
                            data={[metrics.nose_face_ratio || 0]}
                            label="Nose/Face Ratio"
                            maxValue={1.0}
                            threshold={0.4}
                        />
                    )}

                    {currentTrap === 'squint' && (
                        <MetricChart
                            data={[metrics.edge_count || 0]}
                            label="Edge Density"
                            maxValue={500}
                            threshold={100}
                        />
                    )}

                    {currentTrap === 'disco' && (metrics as any).disco_match_score !== undefined && (
                        <div className="metric-chart-container">
                            <span className="metric-label">Color Match: {(metrics as any).disco_match_score}%</span>
                            <div className="disco-bar-bg">
                                <div
                                    className="disco-bar-fill"
                                    style={{ width: `${(metrics as any).disco_match_score}%` }}
                                />
                            </div>
                        </div>
                    )}

                    {/* Signal Hygiene - Always Visible during trap */}
                    <div className="mt-4 border-t border-gray-700 pt-2">
                        <MetricChart
                            data={[metrics.laplacian_variance || 0]}
                            label="Signal Hygiene (Blur)"
                            maxValue={1000}
                            threshold={100}
                            color="#aa44ff"
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
