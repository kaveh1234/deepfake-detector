import './Camera.css';

interface QualityBannerProps {
  fps: number;
  resolution: { width: number; height: number } | null;
}

export default function QualityBanner({ fps, resolution }: QualityBannerProps) {
  const isFpsOk = fps >= 15;
  const isResOk = resolution ? resolution.height >= 720 : false;
  const isQualityOk = isFpsOk && isResOk;

  if (!resolution) return null;

  return (
    <div className={`quality-banner ${isQualityOk ? 'ok' : 'warning'}`}>
      <div className="metrics-group">
        <span className={!isFpsOk ? 'metric-bad' : ''}>
          FPS: {fps}
        </span>
        <span className={!isResOk ? 'metric-bad' : ''}>
          RES: {resolution.width}x{resolution.height}
        </span>
      </div>

      {!isQualityOk && (
        <div className="status-group">
          <span className="icon">⚠️</span>
          <span>Low Quality</span>
        </div>
      )}

      {isQualityOk && (
        <div className="status-group">
          <span className="icon">✓</span>
          <span>HD Signal</span>
        </div>
      )}
    </div>
  );
}
