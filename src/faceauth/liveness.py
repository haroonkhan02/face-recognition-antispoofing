"""Active liveness: a randomised head-turn challenge checked from YuNet's 5 landmarks.

Passive anti-spoofing (the CNN) judges a single frame. An *active* challenge asks the user to do
something an attack can't easily reproduce: here, turn the head left and right in a random order
within a time limit. A printed photo can't turn; a pre-recorded video doesn't know the order.

Yaw proxy (no 3D model needed): ``(nose_x - eye_midpoint_x) / inter-eye distance``. It is ~0 for
a frontal face and grows in magnitude as the head turns. Computed on the *mirrored* preview, so
"left" means the user's left as they see themselves on screen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

TURN_THRESHOLD = 0.25  # |yaw proxy| needed to count as a turn
CENTER_THRESHOLD = 0.10  # must come back below this between turns


def yaw_proxy(landmarks: np.ndarray) -> float:
    """Signed horizontal nose offset in units of inter-eye distance (negative = screen-left)."""
    lm = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    eye_a, eye_b, nose = lm[0], lm[1], lm[2]
    dist = np.linalg.norm(eye_b - eye_a)
    if dist < 1e-6:
        return 0.0
    return float((nose[0] - (eye_a[0] + eye_b[0]) / 2) / dist)


@dataclass
class HeadTurnChallenge:
    """State machine: centre -> turn #1 -> centre -> turn #2 -> passed (or timed out)."""

    timeout_s: float = 8.0
    seed: int | None = None
    steps: list[str] = field(default_factory=list)
    state: str = "waiting"  # waiting / center / turn / passed / failed
    _index: int = 0
    _start: float | None = None

    def __post_init__(self):
        if not self.steps:
            self.steps = random.Random(self.seed).sample(["left", "right"], 2)

    @property
    def prompt(self) -> str:
        if self.state == "passed":
            return "liveness passed"
        if self.state == "failed":
            return "liveness failed: try again"
        if self.state in ("waiting", "center"):
            return "look straight at the camera"
        return f"turn your head {self.steps[self._index]}"

    def update(self, landmarks: np.ndarray | None, t: float) -> str:
        """Feed one frame's landmarks (None = no face) and its timestamp in seconds."""
        if self.state in ("passed", "failed"):
            return self.state
        if self._start is None:
            self._start = t
        if t - self._start > self.timeout_s:
            self.state = "failed"
            return self.state
        if landmarks is None:
            return self.state
        yaw = yaw_proxy(landmarks)
        if self.state in ("waiting", "center"):
            if abs(yaw) < CENTER_THRESHOLD:
                self.state = "turn"
        elif self.state == "turn":
            want = self.steps[self._index]
            if (want == "left" and yaw < -TURN_THRESHOLD) or (
                    want == "right" and yaw > TURN_THRESHOLD):
                self._index += 1
                self.state = "passed" if self._index == len(self.steps) else "center"
        return self.state
