"""Trap implementations for deepfake detection."""

import cv2
import numpy as np
import base64
from dataclasses import dataclass
from typing import Optional
from collections import deque

from face_processor import FaceROI
from models import TrapType, TrapStatus, TrapResult, DiscoPhase


@dataclass
class CalibrationData:
    """Baseline data from calibration phase."""
    laplacian_variance: float  # lap_base
    nose_face_ratio: float  # R0
    edge_count: float  # E0
    samples: int


class SignalHygiene:
    """Signal Hygiene checks - Laplacian Variance (Blur/Smoothness Gate)."""

    CONSECUTIVE_FAIL_THRESHOLD = 10  # Frames before failing
    SMOOTHNESS_RATIO = 0.55  # Fail if < baseline * 0.55

    def __init__(self):
        self.fail_count = 0
        self.variance_history = deque(maxlen=30)

    def compute_laplacian_variance(self, face_roi: np.ndarray) -> float:
        """
        Compute Laplacian variance for texture analysis.

        High values = sharp/detailed texture (real face)
        Low values = smooth/plastic texture (potential deepfake)
        """
        if face_roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())

        self.variance_history.append(variance)
        return variance

    def check_smoothness(
        self,
        current_variance: float,
        baseline: float
    ) -> tuple[bool, str]:
        """
        Check if face texture is suspiciously smooth.

        Returns:
            (passed, message)
        """
        threshold = baseline * self.SMOOTHNESS_RATIO

        if current_variance < threshold:
            self.fail_count += 1
            if self.fail_count >= self.CONSECUTIVE_FAIL_THRESHOLD:
                return False, f"Smoothness drop detected ({current_variance:.1f} < {threshold:.1f})"
        else:
            self.fail_count = max(0, self.fail_count - 1)

        return True, "Texture OK"

    def reset(self):
        """Reset signal hygiene state."""
        self.fail_count = 0
        self.variance_history.clear()


class FisheyeTrap:
    """
    Fisheye Trap - Macro Lens Distortion Detection.

    Real cameras exhibit barrel distortion when objects are close,
    causing the nose to appear proportionally larger relative to face width.
    Virtual cameras/deepfakes don't exhibit this physical distortion.
    """

    # Thresholds
    RATIO_INCREASE_MIN = 0.12  # Minimum absolute increase in R
    RATIO_MULTIPLIER_PASS = 1.8  # R_close / R0 for strong pass
    RATIO_MULTIPLIER_FAIL = 1.4  # Below this = fail
    PENALTY = 2  # Lives lost on failure

    def __init__(self):
        self.ratio_history = deque(maxlen=30)
        self.max_ratio = 0.0

    def compute_ratio(self, roi_data: FaceROI) -> float:
        """Compute nose-to-face width ratio."""
        if roi_data.face_width == 0:
            return 0.0

        ratio = roi_data.nose_width / roi_data.face_width
        self.ratio_history.append(ratio)
        self.max_ratio = max(self.max_ratio, ratio)
        return ratio

    def evaluate(self, baseline_ratio: float) -> TrapResult:
        """
        Evaluate fisheye trap results.

        Pass conditions (either):
        - R_close >= R0 + 0.12 (absolute increase)
        - R_close / R0 >= 1.8 (relative increase)

        Fail condition:
        - R_close / R0 < 1.4
        """
        if baseline_ratio == 0:
            return TrapResult(
                trap_type=TrapType.FISHEYE,
                status=TrapStatus.FAILED,
                score=0.0,
                baseline=0.0,
                threshold=0.0,
                penalty=self.PENALTY,
                message="No baseline ratio available"
            )

        r_close = self.max_ratio
        ratio_multiplier = r_close / baseline_ratio
        absolute_increase = r_close - baseline_ratio

        # Check pass conditions
        passed = (
            absolute_increase >= self.RATIO_INCREASE_MIN or
            ratio_multiplier >= self.RATIO_MULTIPLIER_PASS
        )

        # Override to fail if ratio multiplier is too low
        if ratio_multiplier < self.RATIO_MULTIPLIER_FAIL:
            passed = False

        return TrapResult(
            trap_type=TrapType.FISHEYE,
            status=TrapStatus.PASSED if passed else TrapStatus.FAILED,
            score=ratio_multiplier,
            baseline=baseline_ratio,
            threshold=self.RATIO_MULTIPLIER_FAIL,
            penalty=0 if passed else self.PENALTY,
            message=(
                f"Ratio increase: {ratio_multiplier:.2f}x "
                f"(R0={baseline_ratio:.3f} -> R={r_close:.3f})"
            )
        )

    def reset(self):
        """Reset trap state."""
        self.ratio_history.clear()
        self.max_ratio = 0.0


class SquintTrap:
    """
    Squint Trap - Surface Texture Entropy Detection.

    When a real person scrunches their nose, edge count increases
    due to skin wrinkles and texture changes. Deepfakes often fail
    to generate realistic micro-expressions and texture changes.
    """

    # Thresholds
    EDGE_SPIKE_PASS = 1.35  # E_squint / E0 for pass
    EDGE_SPIKE_FAIL = 1.15  # Below this = fail
    PENALTY = 1  # Lives lost on failure

    # Canny edge detection parameters
    CANNY_LOW = 50
    CANNY_HIGH = 150

    def __init__(self):
        self.edge_history = deque(maxlen=30)
        self.max_edge_count = 0
        self.last_nose_roi: Optional[np.ndarray] = None  # Store for edge map

    def compute_edge_count(self, nose_roi: np.ndarray) -> float:
        """Compute edge count using Canny edge detection."""
        if nose_roi.size == 0:
            return 0.0

        self.last_nose_roi = nose_roi.copy()  # Store for visualization

        gray = cv2.cvtColor(nose_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.CANNY_LOW, self.CANNY_HIGH)
        edge_count = float(np.count_nonzero(edges))

        self.edge_history.append(edge_count)
        self.max_edge_count = max(self.max_edge_count, edge_count)
        return edge_count

    def get_edge_visualization(self, nose_roi: np.ndarray) -> np.ndarray:
        """Get edge detection visualization for UI."""
        if nose_roi.size == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        gray = cv2.cvtColor(nose_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.CANNY_LOW, self.CANNY_HIGH)
        return edges

    def evaluate(self, baseline_edge_count: float) -> TrapResult:
        """
        Evaluate squint trap results.

        Pass: E_squint >= E0 * 1.35
        Fail: E_squint < E0 * 1.15
        """
        # Generate edge map visualization
        edge_map_b64 = None
        if self.last_nose_roi is not None and self.last_nose_roi.size > 0:
            edges = self.get_edge_visualization(self.last_nose_roi)
            _, buffer = cv2.imencode('.png', edges)
            edge_map_b64 = base64.b64encode(buffer).decode('utf-8')

        if baseline_edge_count == 0:
            return TrapResult(
                trap_type=TrapType.SQUINT,
                status=TrapStatus.FAILED,
                score=0.0,
                baseline=0.0,
                threshold=0.0,
                penalty=self.PENALTY,
                message="No baseline edge count available",
                edge_map_b64=edge_map_b64
            )

        e_squint = self.max_edge_count
        edge_multiplier = e_squint / baseline_edge_count

        passed = edge_multiplier >= self.EDGE_SPIKE_PASS

        # Explicit fail condition
        if edge_multiplier < self.EDGE_SPIKE_FAIL:
            passed = False

        return TrapResult(
            trap_type=TrapType.SQUINT,
            status=TrapStatus.PASSED if passed else TrapStatus.FAILED,
            score=edge_multiplier,
            baseline=baseline_edge_count,
            threshold=self.EDGE_SPIKE_FAIL,
            penalty=0 if passed else self.PENALTY,
            message=(
                f"Edge increase: {edge_multiplier:.2f}x "
                f"(E0={baseline_edge_count:.0f} -> E={e_squint:.0f})"
            ),
            edge_map_b64=edge_map_b64
        )

    def reset(self):
        """Reset trap state."""
        self.edge_history.clear()
        self.max_edge_count = 0
        self.last_nose_roi = None


class DiscoTrap:
    """
    Disco Trap - Photometric Causality Detection (FATAL).

    Flash RED -> GREEN -> BLUE on screen and verify that face color
    changes correspondingly. Pre-recorded or synthetic feeds won't
    show proper color reflection from the screen light.
    """

    # Thresholds
    ERROR_THRESHOLD = 0.25  # Normalized error threshold per color
    PENALTY = 3  # Lives lost on failure (FATAL)

    # Ground truth colors (normalized RGB)
    COLORS = {
        DiscoPhase.RED: np.array([1.0, 0.0, 0.0]),
        DiscoPhase.GREEN: np.array([0.0, 1.0, 0.0]),
        DiscoPhase.BLUE: np.array([0.0, 0.0, 1.0]),
    }

    def __init__(self):
        self.phase_results: dict[DiscoPhase, Optional[float]] = {
            DiscoPhase.RED: None,
            DiscoPhase.GREEN: None,
            DiscoPhase.BLUE: None,
        }
        self.baseline_color: Optional[np.ndarray] = None

    def set_baseline_color(self, color: tuple[float, float, float]):
        """Set the baseline neutral face color."""
        self.baseline_color = np.array(color)

    def compute_color_match(
        self,
        face_color: tuple[float, float, float],
        phase: DiscoPhase
    ) -> float:
        """
        Compute how well the face color matches the expected flash.

        Returns match score 0-1 (higher = better match)
        """
        face_rgb = np.array(face_color)
        ground_truth = self.COLORS[phase]

        # Compute color shift direction from baseline
        if self.baseline_color is not None:
            color_shift = face_rgb - self.baseline_color
            # Normalize the shift
            shift_norm = np.linalg.norm(color_shift)
            if shift_norm > 0.01:
                color_shift_normalized = color_shift / shift_norm
            else:
                color_shift_normalized = np.zeros(3)
        else:
            color_shift_normalized = face_rgb

        # Check if the dominant color channel matches expected
        dominant_channel = np.argmax(np.abs(color_shift_normalized))
        expected_channel = np.argmax(ground_truth)

        # Compute correlation/match score
        if dominant_channel == expected_channel and color_shift_normalized[dominant_channel] > 0:
            # Color shift is in the right direction
            match_score = abs(color_shift_normalized[dominant_channel])
        else:
            # Wrong direction or wrong channel
            match_score = 0.0

        return float(match_score)

    def record_phase(
        self,
        phase: DiscoPhase,
        face_color: tuple[float, float, float]
    ) -> float:
        """Record color match for a disco phase."""
        match_score = self.compute_color_match(face_color, phase)
        self.phase_results[phase] = match_score
        return match_score

    def evaluate(self) -> TrapResult:
        """
        Evaluate disco trap results.

        All three phases must show reasonable color correlation.
        """
        # Check if all phases were recorded
        missing_phases = [
            p for p, score in self.phase_results.items()
            if score is None
        ]

        if missing_phases:
            return TrapResult(
                trap_type=TrapType.DISCO,
                status=TrapStatus.FAILED,
                score=0.0,
                baseline=0.0,
                threshold=self.ERROR_THRESHOLD,
                penalty=self.PENALTY,
                message=f"Missing phases: {[p.value for p in missing_phases]}"
            )

        # Calculate average match score
        scores = list(self.phase_results.values())
        avg_score = sum(s for s in scores if s is not None) / len(scores)

        # Check individual phase thresholds
        # For disco, we want at least some response to each color
        min_response = 0.1  # Minimum detectable response
        phases_passed = sum(
            1 for s in scores
            if s is not None and s >= min_response
        )

        # Pass if at least 2/3 phases show response and avg is reasonable
        passed = phases_passed >= 2 and avg_score >= min_response

        phase_details = ", ".join(
            f"{p.value}={s:.2f}" if s else f"{p.value}=N/A"
            for p, s in self.phase_results.items()
        )

        return TrapResult(
            trap_type=TrapType.DISCO,
            status=TrapStatus.PASSED if passed else TrapStatus.FAILED,
            score=avg_score,
            baseline=0.0,
            threshold=min_response,
            penalty=0 if passed else self.PENALTY,
            message=f"Color response: {phase_details}"
        )

    def reset(self):
        """Reset trap state."""
        self.phase_results = {
            DiscoPhase.RED: None,
            DiscoPhase.GREEN: None,
            DiscoPhase.BLUE: None,
        }
        self.baseline_color = None


class TrapRunner:
    """Orchestrates trap execution and scoring."""

    INITIAL_LIVES = 3

    def __init__(self):
        self.signal_hygiene = SignalHygiene()
        self.fisheye_trap = FisheyeTrap()
        self.squint_trap = SquintTrap()
        self.disco_trap = DiscoTrap()

        self.lives = self.INITIAL_LIVES
        self.calibration: Optional[CalibrationData] = None
        self.results: list[TrapResult] = []

    def calibrate(self, samples: list[FaceROI]) -> CalibrationData:
        """
        Perform calibration from neutral face samples.

        Computes baseline metrics over the calibration period.
        """
        if not samples:
            raise ValueError("No samples provided for calibration")

        # Compute averages
        lap_variances = []
        ratios = []
        edge_counts = []

        for roi in samples:
            # Laplacian variance
            gray = cv2.cvtColor(roi.face_roi, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_variances.append(float(lap.var()))

            # Nose/face ratio
            if roi.face_width > 0:
                ratios.append(roi.nose_width / roi.face_width)

            # Edge count
            if roi.nose_roi.size > 0:
                gray_nose = cv2.cvtColor(roi.nose_roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_nose, 50, 150)
                edge_counts.append(float(np.count_nonzero(edges)))

        self.calibration = CalibrationData(
            laplacian_variance=np.mean(lap_variances) if lap_variances else 0,
            nose_face_ratio=np.mean(ratios) if ratios else 0,
            edge_count=np.mean(edge_counts) if edge_counts else 0,
            samples=len(samples)
        )

        return self.calibration

    def run_signal_hygiene(self, roi: FaceROI) -> tuple[float, bool, str]:
        """
        Run signal hygiene check on a frame.

        Returns: (variance, passed, message)
        """
        variance = self.signal_hygiene.compute_laplacian_variance(roi.face_roi)

        if self.calibration:
            passed, msg = self.signal_hygiene.check_smoothness(
                variance,
                self.calibration.laplacian_variance
            )
            # Apply -1 life penalty when smoothness gate fails
            if not passed:
                self.lives -= 1
                self.signal_hygiene.reset()  # Reset counter after penalty
                msg = f"Signal hygiene FAILED: -1 life ({msg})"
        else:
            passed, msg = True, "No calibration baseline"

        return variance, passed, msg

    def process_fisheye_frame(self, roi: FaceROI) -> float:
        """Process a frame during fisheye trap."""
        return self.fisheye_trap.compute_ratio(roi)

    def evaluate_fisheye(self) -> TrapResult:
        """Evaluate fisheye trap and apply penalty."""
        if not self.calibration:
            return TrapResult(
                trap_type=TrapType.FISHEYE,
                status=TrapStatus.FAILED,
                score=0,
                baseline=0,
                threshold=0,
                penalty=2,
                message="Not calibrated"
            )

        result = self.fisheye_trap.evaluate(self.calibration.nose_face_ratio)
        self.lives -= result.penalty
        self.results.append(result)
        return result

    def process_squint_frame(self, roi: FaceROI) -> float:
        """Process a frame during squint trap."""
        return self.squint_trap.compute_edge_count(roi.nose_roi)

    def evaluate_squint(self) -> TrapResult:
        """Evaluate squint trap and apply penalty."""
        if not self.calibration:
            return TrapResult(
                trap_type=TrapType.SQUINT,
                status=TrapStatus.FAILED,
                score=0,
                baseline=0,
                threshold=0,
                penalty=1,
                message="Not calibrated"
            )

        result = self.squint_trap.evaluate(self.calibration.edge_count)
        self.lives -= result.penalty
        self.results.append(result)
        return result

    def set_disco_baseline(self, face_color: tuple[float, float, float]):
        """Set baseline color for disco trap."""
        self.disco_trap.set_baseline_color(face_color)

    def process_disco_phase(
        self,
        phase: DiscoPhase,
        face_color: tuple[float, float, float]
    ) -> float:
        """Process a disco phase."""
        return self.disco_trap.record_phase(phase, face_color)

    def evaluate_disco(self) -> TrapResult:
        """Evaluate disco trap and apply penalty (FATAL if fails)."""
        result = self.disco_trap.evaluate()
        self.lives -= result.penalty
        self.results.append(result)
        return result

    def get_verdict(self) -> str:
        """Get final verdict based on lives remaining."""
        if self.lives <= 0:
            return "LIKELY_FAKE"
        elif self.lives == self.INITIAL_LIVES:
            return "LIKELY_REAL"
        else:
            return "LIKELY_REAL"  # Passed with some issues

    def reset(self):
        """Reset all trap state for a new session."""
        self.signal_hygiene.reset()
        self.fisheye_trap.reset()
        self.squint_trap.reset()
        self.disco_trap.reset()

        self.lives = self.INITIAL_LIVES
        self.calibration = None
        self.results = []
