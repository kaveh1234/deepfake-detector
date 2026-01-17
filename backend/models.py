"""Pydantic models for the Deepfake KYC Detector API."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class TrapType(str, Enum):
    """Types of traps in the detection system."""
    FISHEYE = "fisheye"
    SQUINT = "squint"
    DISCO = "disco"


class TrapStatus(str, Enum):
    """Status of a trap execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"


class SessionState(str, Enum):
    """State of a detection session."""
    IDLE = "idle"
    CALIBRATING = "calibrating"
    READY = "ready"
    RUNNING_TRAP = "running_trap"
    COMPLETED = "completed"


class DiscoPhase(str, Enum):
    """Phases of the Disco trap."""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


# WebSocket Message Models
class CalibrationBaseline(BaseModel):
    """Baseline metrics captured during calibration."""
    laplacian_variance: float  # lap_base
    nose_face_ratio: float  # R0
    edge_count: float  # E0
    timestamp: float


class FaceLandmarks(BaseModel):
    """Face landmark data from MediaPipe."""
    nose_width: float
    face_width: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    landmarks: list[tuple[float, float, float]]  # normalized x, y, z


class TrapResult(BaseModel):
    """Result of a single trap execution."""
    trap_type: TrapType
    status: TrapStatus
    score: float
    baseline: float
    threshold: float
    penalty: int
    message: str


class LiveMetrics(BaseModel):
    """Real-time metrics sent to frontend."""
    laplacian_variance: float
    nose_face_ratio: Optional[float] = None
    edge_count: Optional[float] = None
    disco_match_score: Optional[float] = None
    face_detected: bool
    quality_ok: bool
    fps: Optional[float] = None
    resolution: Optional[tuple[int, int]] = None


class SessionInfo(BaseModel):
    """Current session state information."""
    session_id: str
    state: SessionState
    lives: int
    calibrated: bool
    current_trap: Optional[TrapType] = None
    trap_results: list[TrapResult] = []
    verdict: Optional[str] = None


# WebSocket Protocol Messages
class WSMessageType(str, Enum):
    """Types of WebSocket messages."""
    # Client -> Server
    FRAME = "frame"  # Low-res continuous stream
    SNAPSHOT = "snapshot"  # Full-res trap sampling
    CONTROL = "control"  # Start/reset/run trap

    # Server -> Client
    METRICS = "metrics"  # Live metrics update
    TRAP_RESULT = "trap_result"  # Trap pass/fail
    SESSION_UPDATE = "session_update"  # Session state change
    ERROR = "error"  # Error message
    CALIBRATION_COMPLETE = "calibration_complete"
    DISCO_PHASE = "disco_phase"  # Current disco color phase


class ControlAction(str, Enum):
    """Control actions from client."""
    START_SESSION = "start_session"
    RESET_SESSION = "reset_session"
    START_CALIBRATION = "start_calibration"
    RUN_TRAP = "run_trap"
    CANCEL_TRAP = "cancel_trap"


class WSMessage(BaseModel):
    """Generic WebSocket message wrapper."""
    type: WSMessageType
    data: dict
