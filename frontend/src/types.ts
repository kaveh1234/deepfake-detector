// WebSocket Message Types
export type WSMessageType =
    | 'frame'
    | 'snapshot'
    | 'control'
    | 'metrics'
    | 'trap_result'
    | 'session_update'
    | 'error'
    | 'calibration_complete'
    | 'disco_phase';

export type ControlAction =
    | 'start_session'
    | 'reset_session'
    | 'start_calibration'
    | 'run_trap'
    | 'cancel_trap';

export type TrapType = 'fisheye' | 'squint' | 'disco';
export type TrapStatus = 'pending' | 'running' | 'passed' | 'failed';
export type SessionState = 'idle' | 'calibrating' | 'ready' | 'running_trap' | 'completed';
export type DiscoPhase = 'red' | 'green' | 'blue';

export interface TrapResult {
    trap_type: TrapType;
    status: TrapStatus;
    score: number;
    baseline: number;
    threshold: number;
    penalty: number;
    message: string;
    edge_map_b64?: string;  // Edge visualization for squint trap
}

export interface CalibrationData {
    laplacian_variance: number;
    nose_face_ratio: number;
    edge_count: number;
}

export interface Metrics {
    laplacian_variance: number;
    nose_face_ratio?: number;
    edge_count?: number;
    disco_match_score?: number;
    face_detected: boolean;
    quality_ok: boolean;
    quality_message?: string;
    fps: number;
    resolution: [number, number];
    lives: number;
    state: SessionState;
}

export interface SessionInfo {
    session_id: string;
    state: SessionState;
    lives: number;
    calibrated: boolean;
    current_trap: TrapType | null;
    trap_results: TrapResult[];
    verdict: string | null;
    calibration: CalibrationData | null;
}
