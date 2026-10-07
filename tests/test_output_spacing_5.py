"""Small interface/known-answer checks, not a replacement for real-data parity."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/output-spacing-5'))
import frozen_support_safe_finalize as helper
import grow_track_graph as producer


class OutputSpacingFiveTests(unittest.TestCase):
    def test_wrong_pair_rejected_before_growth(self):
        args = SimpleNamespace(coarse_mask_mode='support', resample_spacing=5., output_spacing=10.)
        result = producer.process_seed(0, Path('must-not-be-created'), args, None, None)
        self.assertFalse(result['ok'])
        self.assertIn('resample-spacing 5 and output-spacing 5', result['error'])

    def test_near_identity_full_plane_keeps_supported_rim(self):
        y, x = np.indices((11, 13))
        fine = np.stack((x + 100., y + 100., np.full_like(x, 100.)), axis=-1)
        factor = 1.0005
        raw = producer.resample_grid(fine, factor)
        result, certificate = helper.finalize(raw, fine[..., 0] >= 0, factor)
        np.testing.assert_array_equal(result, raw)
        self.assertEqual(certificate['initial_bad_footprint_quads'], 0)
        self.assertTrue(certificate['independent_brute_support']['passed'])
        self.assertGreater((result[..., 0] >= 0).sum(),
                           (producer.finalize_coarse_grid(raw)[..., 0] >= 0).sum())


if __name__ == '__main__':
    unittest.main()
