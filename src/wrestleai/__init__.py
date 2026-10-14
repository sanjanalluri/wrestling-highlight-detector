"""WrestleAI: the Groot highlight-detection pipeline for wrestling match footage.

Stages: data processing -> optional DeepLabV3+ silhouette masking -> per-frame
EfficientNet features -> Bi-LSTM temporal model -> dense classifier.
"""

__version__ = "0.1.0"
