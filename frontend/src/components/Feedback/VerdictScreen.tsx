import './Feedback.css';
import { TrapResult } from '../../types';

interface VerdictScreenProps {
  verdict: string | null; // 'LIKELY_REAL' | 'LIKELY_FAKE'
  trapResults: TrapResult[];
  onReset: () => void;
}

export default function VerdictScreen({ verdict, trapResults, onReset }: VerdictScreenProps) {
  if (!verdict) return null;

  // Handle backend verdict format: LIKELY_REAL or LIKELY_FAKE
  const isPass = verdict.toUpperCase().includes('REAL');
  const failedTraps = trapResults.filter(t => t.status === 'failed');
  const passedTraps = trapResults.filter(t => t.status === 'passed');

  return (
    <div className="verdict-screen-overlay">
      <div className={`verdict-badge ${isPass ? 'verdict-pass' : 'verdict-fail'}`}>
        {isPass ? 'LIKELY REAL' : 'LIKELY FAKE'}
      </div>

      {/* Summary */}
      <div className="verdict-summary">
        <span className="summary-item passed">{passedTraps.length} Passed</span>
        <span className="summary-item failed">{failedTraps.length} Failed</span>
      </div>

      {/* Show all trap results */}
      <div className="trap-results-log">
        {trapResults.map((trap, idx) => (
          <div key={idx} className={`trap-result-item ${trap.status}`}>
            <span className="trap-icon">
              {trap.status === 'passed' ? '✓' : '✗'}
            </span>
            <span className="trap-name">{trap.trap_type.toUpperCase()}</span>
            <span className="trap-score">
              {trap.score.toFixed(2)}x
            </span>
            <span className="trap-message">{trap.message}</span>
          </div>
        ))}
      </div>

      {!isPass && failedTraps.length > 0 && (
        <div className="failure-log">
          <h4 className="failure-title">Failure Details</h4>
          {failedTraps.map((trap, idx) => (
            <div key={idx} className="failure-item">
              <span className="failure-icon">⚠</span>
              <span>{trap.trap_type}: -{trap.penalty} life(s)</span>
            </div>
          ))}
        </div>
      )}

      <button className="btn btn-primary verdict-reset-btn" onClick={onReset}>
        Start New Session
      </button>
    </div>
  );
}
