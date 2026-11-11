"""
EEG processing and neural encoding layer.
"""
from src.eeg.loader import N_CHANNELS, SFREQ, load_and_filter_edf
from src.eeg.models import EMBED_DIM, EEGEncoder
from src.eeg.pipeline import DUMMY_EDF_PATH, EEGPipeline
from src.eeg.preprocessor import (
    STRIDE,
    WINDOW_SEC,
    WINDOW_SIZE,
    make_dummy_eeg,
    normalize_windows,
    segment_windows,
)

__all__ = [
    "SFREQ",
    "N_CHANNELS",
    "WINDOW_SEC",
    "WINDOW_SIZE",
    "STRIDE",
    "EMBED_DIM",
    "DUMMY_EDF_PATH",
    "load_and_filter_edf",
    "segment_windows",
    "normalize_windows",
    "make_dummy_eeg",
    "EEGEncoder",
    "EEGPipeline",
]
