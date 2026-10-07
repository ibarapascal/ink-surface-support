"""Portable known-answer tests; run: python spiral-fitting/test_support_safe_finalize.py.

Requires only Python, NumPy, SciPy and the adjacent producer/helper sources.
No frozen implementation, local experiment paths, reference meshes or downloads.
"""
import unittest
from types import SimpleNamespace

import numpy as np

import grow_track_graph as producer
import support_safe_finalize as support


def plane(shape):
    rows, columns = np.indices(shape)
    return np.stack((columns + 100., rows + 100., np.full(shape, 100.)), axis=-1)


def masked_plane(mask, spacing=1.):
    grid = plane(mask.shape) * spacing
    grid[~mask] = -1.
    return grid


class SupportFinalizationTests(unittest.TestCase):
    def test_complete_plane_is_unchanged(self):
        fine = plane((13, 13))
        coarse = producer.resample_grid(fine, 4.)
        result, certificate = support.finalize(coarse, fine[..., 0] >= 0, 4.)
        np.testing.assert_array_equal(result, coarse)
        self.assertEqual(certificate['final_valid_vertices'], 16)
        self.assertEqual(certificate['final_complete_quads'], 9)
        self.assertEqual(certificate['deleted_greedily'], [])

    def test_fractional_lattice_full_planes(self):
        fine = plane((15, 17))
        for factor in (2.5, 4.00000000001, 4.000000002, .75):
            with self.subTest(factor=factor):
                coarse = producer.resample_grid(fine, factor)
                result, _ = support.finalize(coarse, fine[..., 0] >= 0, factor)
                np.testing.assert_array_equal(result, coarse)

    def test_endpoint_epsilon_changes_support_footprint(self):
        valid = np.ones((13, 13), dtype=bool)
        valid[5, 1] = False
        # End4+1e-11 snaps support to row4. End4+2e-9 includes row5.
        self.assertTrue(support.full_support(valid, 4.00000000001)[0, 0])
        self.assertFalse(support.full_support(valid, 4.000000002)[0, 0])
        # An exact endpoint excludes row5 as well.
        self.assertTrue(support.full_support(valid, 4.)[0, 0])

    def test_interior_hole_with_valid_coarse_corners(self):
        valid = np.ones((13, 13), dtype=bool)
        valid[2, 2] = False
        coarse = producer.resample_grid(masked_plane(valid), 4.)
        self.assertTrue((coarse[..., 0] >= 0).all())
        result, certificate = support.finalize(coarse, valid, 4.)
        # Only the top-left rectangle includes the hole. Its top-left vertex
        # touches no supported neighbor, so deletion has unique zero loss.
        expected = coarse.copy()
        expected[0, 0] = -1.
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(certificate['initial_bad_footprint_quads'], 1)
        self.assertEqual(certificate['deleted_greedily'], [{
            'vertex': [0, 0], 'destroyed_adjacent_safe_quad_area_mm2': 0.,
        }])

    def test_equal_components_keep_first_scan_order(self):
        valid = np.zeros((13, 21), dtype=bool)
        valid[:5, :5] = True
        valid[8:13, 16:21] = True
        coarse = producer.resample_grid(masked_plane(valid), 2.)
        result, certificate = support.finalize(coarse, valid, 2.)
        expected = np.full(coarse.shape, -1.)
        expected[:3, :3] = coarse[:3, :3]
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(certificate['components_before_largest'], 2)
        self.assertEqual(certificate['vertices_removed_by_component_rule'], 9)
        self.assertEqual(certificate['deleted_greedily'], [])

    def test_four_connectivity_excludes_diagonal_only_neighbor(self):
        coarse = np.full((3, 3, 3), -1.)
        coarse[0, 0] = (100., 100., 100.)
        coarse[1, 1] = (104., 104., 100.)
        result, certificate = support.finalize(coarse, np.ones((9, 9), bool), 4.)
        expected = np.full(coarse.shape, -1.)
        expected[0, 0] = coarse[0, 0]
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(certificate['components_before_largest'], 2)

    def test_no_restore_no_coordinate_change_no_input_mutation(self):
        valid = np.ones((13, 13), dtype=bool)
        valid[2, 2] = False
        coarse = producer.resample_grid(masked_plane(valid), 4.)
        coarse[3, 3] = -1.
        before = coarse.copy()
        fine_before = valid.copy()
        result, _ = support.finalize(coarse, valid, 4.)
        keep = result[..., 0] >= 0
        self.assertFalse(np.any(keep & (before[..., 0] < 0)))
        np.testing.assert_array_equal(result[keep], before[keep])
        np.testing.assert_array_equal(coarse, before)
        np.testing.assert_array_equal(valid, fine_before)

    def test_hole_reachable_after_original_fine40_clean(self):
        raw = np.ones((64, 64), dtype=bool)
        raw[25, 26] = False
        clean = producer.clean_valid_mask(raw, erode_px=8, min_component_px=320)
        coarse = producer.resample_grid(masked_plane(clean, spacing=5.), 4.)
        self.assertTrue((coarse[8:10, 6:8, 0] >= 0).all())
        self.assertFalse(clean[32:37, 24:29].all())
        result, certificate = support.finalize(coarse, clean, 4.)
        self.assertGreater(certificate['initial_bad_footprint_quads'], 0)
        self.assertFalse((result[8:10, 6:8, 0] >= 0).all())
        # The original finalizer also removes this cell. This is a known
        # no-erosion hazard, NOT evidence that the original algorithm is broken.
        original = producer.finalize_coarse_grid(coarse)
        self.assertFalse((original[8:10, 6:8, 0] >= 0).all())

    def test_invalid_factor_and_lattice_rejected(self):
        for factor in (0., -1., float('nan'), float('inf')):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                support.finalize(plane((4, 4)), np.ones((13, 13), bool), factor)
        with self.assertRaises(ValueError):
            support.finalize(plane((3, 4)), np.ones((13, 13), bool), 4.)


    def test_exact_two_factor_hole_is_not_globally_safe(self):
        valid = np.ones((7, 7), dtype=bool)
        valid[1, 1] = False
        coarse = producer.resample_grid(masked_plane(valid), 2.)
        self.assertTrue((coarse[..., 0] >= 0).all())
        result, certificate = support.finalize(coarse, valid, 2.)
        expected = coarse.copy()
        expected[0, 0] = -1.
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(certificate['initial_bad_footprint_quads'], 1)
        self.assertTrue(certificate['independent_brute_support']['passed'])

    def test_producer_rejects_unqualified_spacing_before_growth(self):
        for working, output in ((5., 20.), (5., 5.), (4., 10.)):
            with self.subTest(working=working, output=output):
                args = SimpleNamespace(coarse_mask_mode='support',
                    resample_spacing=working, output_spacing=output)
                result = producer.process_seed(0, None, args, None, None)
                self.assertFalse(result['ok'])
                self.assertEqual(result['error'],
                    'ValueError: experimental support mode is qualified only for '
                    'resample-spacing 5 and output-spacing 10')


if __name__ == '__main__':
    unittest.main(verbosity=2)
