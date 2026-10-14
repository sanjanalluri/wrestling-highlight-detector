"""Near-duplicate frame detection via SSIM + optical flow.

Ported from the original Groot ``frame_deduplicator.py`` (same thresholds and logic).
SSIM is computed with OpenCV Gaussian filtering so scikit-image is not required.
Useful for spotting static stretches (stoppages, replays of a still scoreboard).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

SSIM_THRESHOLD = 0.985
OPTICAL_FLOW_THRESHOLD = 0.5  # mean Farneback flow magnitude in pixels


def _gray(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean structural similarity of two RGB frames (Wang et al., 7x7 Gaussian window)."""
    a, b = _gray(frame_a), _gray(frame_b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (7, 7), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def optical_flow_magnitude(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean dense (Farneback) optical-flow magnitude between two RGB frames."""
    a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.mean(np.linalg.norm(flow, axis=-1)))


def is_duplicate(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    ssim_threshold: float = SSIM_THRESHOLD,
    flow_threshold: float = OPTICAL_FLOW_THRESHOLD,
) -> Tuple[bool, float, float]:
    """Return ``(duplicate, ssim, flow)``; duplicate means very similar *and* almost no motion."""
    s = ssim(frame_a, frame_b)
    f = optical_flow_magnitude(frame_a, frame_b)
    return (s > ssim_threshold and f < flow_threshold), s, f


def deduplicate(
    frames: Sequence[np.ndarray],
    ssim_threshold: float = SSIM_THRESHOLD,
    flow_threshold: float = OPTICAL_FLOW_THRESHOLD,
) -> List[int]:
    """Return indices of frames to keep, comparing each frame to the last kept frame."""
    if len(frames) == 0:
        return []
    keep = [0]
    for i in range(1, len(frames)):
        dup, _, _ = is_duplicate(frames[keep[-1]], frames[i], ssim_threshold, flow_threshold)
        if not dup:
            keep.append(i)
    return keep
