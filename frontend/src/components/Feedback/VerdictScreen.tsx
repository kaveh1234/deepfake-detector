import './Feedback.css';
import { TrapResult } from '../../types';

interface VerdictScreenProps {
  verdict: string | null; // 'pass' | 'fail'
  trapResults: TrapResult[];
  onReset: () => void;
}

export default function VerdictScreen({ verdict, trapResults, onReset }: VerdictScreenProps) {
  if (!verdict) return null;

  const isPass = verdict.toLowerCase() === 'pass';
  const failedTraps = trapResults.filter(t => t.status === 'failed');

  return (
    <div className="verdict-screen-overlay">
      <div className={`verdict-badge ${isPass ? 'verdict-pass' : 'verdict-fail'}`}>
        {isPass ? 'PASS' : 'FAIL'}
      </div>

      {!isPass && failedTraps.length > 0 && (
        <div className="failure-log">
          <h4 className="failure-title">Failure Reason</h4>
          {failedTraps.map((trap, idx) => (
            <div key={idx} className="failure-item">
              <span className="failure-icon">⚠️</span>
              <span>{trap.trap_type}: {trap.message || 'Verification failed'}</span>
            </div>
          ))}
        </div>
      )}

      <button className="btn btn-secondary" onClick={onReset}>
        Start New Session
      </button>
    </div>
  );
}
