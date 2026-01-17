"""Session management for deepfake detection."""

import uuid
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from models import SessionState, TrapType, TrapResult, DiscoPhase
from face_processor import FaceProcessor, FaceROI
from traps import TrapRunner, CalibrationData


@dataclass
class SessionMetrics:
    """Real-time session metrics."""
    laplacian_variance: float = 0.0
    nose_face_ratio: float = 0.0
    edge_count: float = 0.0
    disco_match_score: float = 0.0
    face_detected: bool = False
    fps: float = 0.0
    resolution: tuple[int, int] = (0, 0)


@dataclass
class Session:
    """Detection session state."""
    session_id: str
    state: SessionState = SessionState.IDLE
    lives: int = 3
    calibration: Optional[CalibrationData] = None
    current_trap: Optional[TrapType] = None
    trap_results: list[TrapResult] = field(default_factory=list)
    verdict: Optional[str] = None
    metrics: SessionMetrics = field(default_factory=SessionMetrics)

    # Timing
    created_at: float = field(default_factory=time.time)
    calibration_start: Optional[float] = None
    trap_start: Optional[float] = None

    # Calibration samples
    calibration_samples: list[FaceROI] = field(default_factory=list)

    # Frame timing for FPS calculation
    frame_times: deque = field(default_factory=lambda: deque(maxlen=30))

    # Current disco phase
    disco_phase: Optional[DiscoPhase] = None


class SessionManager:
    """Manages detection sessions and coordinates processing."""

    CALIBRATION_DURATION = 0.3  # seconds (Insta-snap)
    TRAP_DURATION = 1.5  # seconds per trap (Ultra fast)
    DISCO_PHASE_DURATION = 0.2  # seconds per color (Strobe)

    # Quality gate thresholds
    MIN_WIDTH = 1280
    MIN_HEIGHT = 720
    MIN_FPS = 15

    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.face_processor = FaceProcessor()
        self.trap_runners: dict[str, TrapRunner] = {}

        # Keep last 3 sessions for history
        self.session_history: deque[Session] = deque(maxlen=3)

    def create_session(self) -> Session:
        """Create a new detection session."""
        session_id = str(uuid.uuid4())[:8]
        session = Session(session_id=session_id)

        self.sessions[session_id] = session
        self.trap_runners[session_id] = TrapRunner()

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def reset_session(self, session_id: str) -> Session:
        """Reset a session to initial state."""
        if session_id in self.sessions:
            old_session = self.sessions[session_id]
            self.session_history.append(old_session)

        # Create fresh session with same ID
        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        self.trap_runners[session_id] = TrapRunner()

        return session

    def start_calibration(self, session_id: str) -> bool:
        """Start the calibration phase."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.state = SessionState.CALIBRATING
        session.calibration_start = time.time()
        session.calibration_samples = []

        return True

    def process_frame(
        self,
        session_id: str,
        frame_data: bytes,
        width: int,
        height: int
    ) -> Optional[dict]:
        """
        Process an incoming video frame.

        Returns metrics dict or None if session not found.
        """
        import cv2
        import numpy as np

        session = self.sessions.get(session_id)
        runner = self.trap_runners.get(session_id)

        if not session or not runner:
            return None

        # Update frame timing for FPS calculation
        now = time.time()
        session.frame_times.append(now)
        if len(session.frame_times) > 1:
            time_span = session.frame_times[-1] - session.frame_times[0]
            if time_span > 0:
                session.metrics.fps = (len(session.frame_times) - 1) / time_span

        # Decode frame
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            session.metrics.face_detected = False
            return self._get_metrics_dict(session)

        session.metrics.resolution = (width, height)

        # Process face
        roi_data = self.face_processor.process_frame(frame)
        session.metrics.face_detected = roi_data is not None

        if roi_data is None:
            return self._get_metrics_dict(session)

        # Update metrics based on current state
        if session.state == SessionState.CALIBRATING:
            self._process_calibration_frame(session, runner, roi_data)
        elif session.state == SessionState.RUNNING_TRAP:
            self._process_trap_frame(session, runner, roi_data, frame)

        # Always compute signal hygiene metrics
        variance, _, _ = runner.run_signal_hygiene(roi_data)
        session.metrics.laplacian_variance = variance

        if roi_data.face_width > 0:
            session.metrics.nose_face_ratio = roi_data.nose_width / roi_data.face_width

        return self._get_metrics_dict(session)

    def process_snapshot(
        self,
        session_id: str,
        frame_data: bytes,
        width: int,
        height: int
    ) -> Optional[dict]:
        """
        Process a high-resolution snapshot during traps.

        Same as process_frame but for full-res images.
        """
        # Snapshots are processed the same way as frames
        # but are expected to be higher quality
        return self.process_frame(session_id, frame_data, width, height)

    def _process_calibration_frame(
        self,
        session: Session,
        runner: TrapRunner,
        roi_data: FaceROI
    ):
        """Process a frame during calibration."""
        session.calibration_samples.append(roi_data)

        # Check if calibration is complete
        elapsed = time.time() - (session.calibration_start or 0)
        if elapsed >= self.CALIBRATION_DURATION:
            # Perform calibration
            try:
                calibration = runner.calibrate(session.calibration_samples)
                session.calibration = calibration
                session.state = SessionState.READY
                session.lives = runner.lives
            except Exception:
                session.state = SessionState.IDLE

    def _process_trap_frame(
        self,
        session: Session,
        runner: TrapRunner,
        roi_data: FaceROI,
        frame
    ):
        """Process a frame during trap execution."""
        import cv2

        if session.current_trap == TrapType.FISHEYE:
            ratio = runner.process_fisheye_frame(roi_data)
            session.metrics.nose_face_ratio = ratio

        elif session.current_trap == TrapType.SQUINT:
            edge_count = runner.process_squint_frame(roi_data)
            session.metrics.edge_count = edge_count

        elif session.current_trap == TrapType.DISCO:
            if session.disco_phase:
                face_color = self.face_processor.get_face_color(frame)
                if face_color:
                    score = runner.process_disco_phase(
                        session.disco_phase,
                        face_color
                    )
                    session.metrics.disco_match_score = score

    def start_trap(self, session_id: str, trap_type: TrapType) -> bool:
        """Start a trap execution."""
        session = self.sessions.get(session_id)
        runner = self.trap_runners.get(session_id)

        if not session or not runner:
            return False

        if session.state not in [SessionState.READY, SessionState.RUNNING_TRAP]:
            return False

        session.state = SessionState.RUNNING_TRAP
        session.current_trap = trap_type
        session.trap_start = time.time()

        # Reset the specific trap
        if trap_type == TrapType.FISHEYE:
            runner.fisheye_trap.reset()
        elif trap_type == TrapType.SQUINT:
            runner.squint_trap.reset()
        elif trap_type == TrapType.DISCO:
            runner.disco_trap.reset()
            # Set baseline color from current face
            if session.calibration_samples:
                import cv2
                # Use first calibration sample for baseline color
                sample = session.calibration_samples[0]
                if sample.cheek_roi.size > 0:
                    rgb_roi = cv2.cvtColor(sample.cheek_roi, cv2.COLOR_BGR2RGB)
                    mean_color = tuple(rgb_roi.mean(axis=(0, 1)) / 255.0)
                    runner.set_disco_baseline(mean_color)

        return True

    def set_disco_phase(self, session_id: str, phase: DiscoPhase):
        """Set the current disco phase."""
        session = self.sessions.get(session_id)
        if session:
            session.disco_phase = phase

    def evaluate_trap(self, session_id: str) -> Optional[TrapResult]:
        """Evaluate the current trap and return result."""
        session = self.sessions.get(session_id)
        runner = self.trap_runners.get(session_id)

        if not session or not runner or not session.current_trap:
            return None

        result = None
        if session.current_trap == TrapType.FISHEYE:
            result = runner.evaluate_fisheye()
        elif session.current_trap == TrapType.SQUINT:
            result = runner.evaluate_squint()
        elif session.current_trap == TrapType.DISCO:
            result = runner.evaluate_disco()

        if result:
            session.trap_results.append(result)
            session.lives = runner.lives

            # Check for completion
            if runner.lives <= 0:
                session.state = SessionState.COMPLETED
                session.verdict = "LIKELY_FAKE"
            elif len(session.trap_results) >= 3:
                session.state = SessionState.COMPLETED
                session.verdict = runner.get_verdict()
            else:
                session.state = SessionState.READY

            session.current_trap = None

        return result

    def check_quality_gate(
        self,
        session_id: str,
        width: int,
        height: int,
        fps: float
    ) -> tuple[bool, str]:
        """
        Check if video quality meets requirements.

        Returns (passed, message)
        """
        issues = []

        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            issues.append(f"Resolution too low: {width}x{height} (need {self.MIN_WIDTH}x{self.MIN_HEIGHT})")

        if fps < self.MIN_FPS:
            issues.append(f"FPS too low: {fps:.1f} (need {self.MIN_FPS})")

        if issues:
            return False, "; ".join(issues)

        return True, "Quality OK"

    def _get_metrics_dict(self, session: Session) -> dict:
        """Convert session metrics to dict."""
        return {
            "laplacian_variance": session.metrics.laplacian_variance,
            "nose_face_ratio": session.metrics.nose_face_ratio,
            "edge_count": session.metrics.edge_count,
            "disco_match_score": session.metrics.disco_match_score,
            "face_detected": session.metrics.face_detected,
            "fps": session.metrics.fps,
            "resolution": session.metrics.resolution,
            "lives": session.lives,
            "state": session.state.value,
        }

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get full session info."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "lives": session.lives,
            "calibrated": session.calibration is not None,
            "current_trap": session.current_trap.value if session.current_trap else None,
            "trap_results": [
                {
                    "trap_type": r.trap_type.value,
                    "status": r.status.value,
                    "score": r.score,
                    "baseline": r.baseline,
                    "threshold": r.threshold,
                    "penalty": r.penalty,
                    "message": r.message,
                }
                for r in session.trap_results
            ],
            "verdict": session.verdict,
            "calibration": {
                "laplacian_variance": session.calibration.laplacian_variance,
                "nose_face_ratio": session.calibration.nose_face_ratio,
                "edge_count": session.calibration.edge_count,
            } if session.calibration else None,
        }

    def cleanup(self):
        """Clean up resources."""
        self.face_processor.close()
