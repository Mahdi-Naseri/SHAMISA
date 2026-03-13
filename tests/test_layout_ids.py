import unittest

import torch

from models.associations import SampleLayout


class SampleLayoutTests(unittest.TestCase):
    def test_index_mapping_consistency(self):
        z_ref = torch.zeros(3, 2, 4)
        z_dist = torch.zeros(3, 2, 5, 6, 4)
        h_ref = torch.zeros(3, 2, 8)
        h_dist = torch.zeros(3, 2, 5, 6, 8)
        z_ref_compact = torch.zeros(6, 4)
        z_dist_compact = torch.zeros(3 * 2 * 5 * 6, 4)
        h_ref_compact = torch.zeros(6, 8)
        h_dist_compact = torch.zeros(3 * 2 * 5 * 6, 8)

        index_map = SampleLayout(
            z_ref,
            z_dist,
            h_ref,
            h_dist,
            z_ref_compact,
            z_dist_compact,
            h_ref_compact,
            h_dist_compact,
        )

        self.assertEqual(index_map.ref_offset(2, 1), 5)
        self.assertEqual(index_map.ref_node(2, 1), 5)

        distorted_flat = index_map.dist_offset(1, 1, 4, 5)
        manual = 1 * (2 * 5 * 6) + 1 * (5 * 6) + 4 * 6 + 5
        self.assertEqual(distorted_flat, manual)
        self.assertEqual(index_map.dist_node(1, 1, 4, 5), z_ref_compact.shape[0] + manual)


if __name__ == "__main__":
    unittest.main()
