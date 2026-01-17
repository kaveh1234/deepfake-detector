import './Traps.css';

interface MetricChartProps {
    data: number[]; // Array of values
    label: string;
    threshold?: number;
    maxValue?: number;
    color?: string;
}

export default function MetricChart({ data, label, threshold, maxValue = 100, color = '#4488ff' }: MetricChartProps) {
    if (!data || data.length === 0) return null;

    const width = 200;
    const height = 60;
    const padding = 5;

    // Normalize data to fit height
    const normalize = (val: number) => {
        const y = height - padding - ((val / maxValue) * (height - 2 * padding));
        return Math.max(padding, Math.min(height - padding, y));
    };

    const points = data.map((val, idx) => {
        const x = (idx / (Math.max(data.length - 1, 1))) * width;
        const y = normalize(val);
        return `${x},${y}`;
    }).join(' ');

    const areaPoints = `0,${height} ${points} ${width},${height}`;

    const thresholdY = threshold ? normalize(threshold) : null;
    const activeValue = data[data.length - 1];

    return (
        <div className="metric-chart-container">
            <span className="metric-label">{label}: {activeValue.toFixed(2)}</span>
            <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`}>
                {/* Area fill */}
                <polygon points={areaPoints} className="chart-area" style={{ fill: color, fillOpacity: 0.2 }} />

                {/* Line */}
                <polyline points={points} className="chart-line" style={{ stroke: color }} />

                {/* Threshold Line */}
                {thresholdY !== null && (
                    <line x1="0" y1={thresholdY} x2={width} y2={thresholdY} className="threshold-line" />
                )}

                {/* Current Value Dot */}
                <circle
                    cx={width}
                    cy={normalize(activeValue)}
                    r="3"
                    fill={color}
                />
            </svg>
        </div>
    );
}
