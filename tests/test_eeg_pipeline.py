"""
Unit tests for EEG signal preprocessing and encoder pipeline.
"""
import unittest
import numpy as np
import torch

from src.eeg.pipeline import EEGPipeline
from src.eeg.preprocessor import make_dummy_eeg, normalize_windows, segment_windows


class TestEEGPipeline(unittest.TestCase):
    def test_dummy_signal_generation(self):
        eeg = make_dummy_eeg(duration_sec=10, sfreq=256, n_channels=23)
        self.assertEqual(eeg.shape, (23, 2560))

    def test_window_segmentation(self):
        data = np.zeros((23, 2560), dtype=np.float32)
        windows = segment_windows(data, window_size=1024, stride=512)
        # (2560 - 1024) // 512 + 1 = 4 windows
        self.assertEqual(windows.shape, (4, 23, 1024))

    def test_eeg_pipeline_execution(self):
        pipeline = EEGPipeline()
        out = pipeline.analyze(None)  # Uses dummy eeg internally
        self.assertIn("seizure_prob", out)
        self.assertIn("z_patient", out)
        self.assertEqual(out["z_patient"].shape, (1, 128))


if __name__ == "__main__":
    unittest.main()
