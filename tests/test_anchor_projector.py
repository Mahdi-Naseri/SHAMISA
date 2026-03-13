import unittest

import torch
from dotmap import DotMap

from models.associations import AnchorProjector


class AnchorProjectorTests(unittest.TestCase):
    def test_forward_shape(self):
        torch.manual_seed(0)
        cfg = DotMap(
            {
                "n_prototypes": 7,
                "prototype_norm": False,
                "weight_norm": False,
            }
        )
        model = AnchorProjector(cfg, feature_dim=5)
        x = torch.randn(4, 5)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4, 7))

    def test_row_normalization(self):
        torch.manual_seed(0)
        cfg = DotMap(
            {
                "n_prototypes": 3,
                "prototype_norm": True,
                "weight_norm": False,
            }
        )
        model = AnchorProjector(cfg, feature_dim=4)
        _ = model(torch.randn(2, 4))
        weights = model.head.weight.data
        row_norms = torch.linalg.norm(weights, dim=1)
        self.assertTrue(torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
