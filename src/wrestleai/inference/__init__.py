"""Video-level inference and highlight range detection."""

from wrestleai.inference.predict import DetectedRange, infer_video, merge_positive_windows

__all__ = ["infer_video", "merge_positive_windows", "DetectedRange"]
