import unittest

from dotmap import DotMap

from models.associations import finalize_association_config


class AssociationConfigTests(unittest.TestCase):
    def test_zero_coeff_disables_branch(self):
        args = DotMap(
            {
                "model": {
                    "relations": {
                        "branches": {
                            "metadata_a": {"active": True, "coeff": 1.0},
                            "metadata_b": {"active": True, "coeff": 0.0},
                        },
                        "regularizer": {"active": True, "coeff": 0.2},
                        "weighting": {"active": True, "mode": "mlp"},
                    }
                }
            }
        )
        out = finalize_association_config(args)
        self.assertTrue(out.model.relations.branches.metadata_a.active)
        self.assertFalse(out.model.relations.branches.metadata_b.active)

    def test_zero_regularizer_coeff_disables_regularizer(self):
        args = DotMap(
            {
                "model": {
                    "relations": {
                        "branches": {"metadata_a": {"active": True, "coeff": 1.0}},
                        "regularizer": {"active": True, "coeff": 0.0},
                        "weighting": {"active": True, "mode": "mlp"},
                    }
                }
            }
        )
        out = finalize_association_config(args)
        self.assertFalse(out.model.relations.regularizer.active)

    def test_disabled_weighting_switches_mode(self):
        args = DotMap(
            {
                "model": {
                    "relations": {
                        "branches": {"metadata_a": {"active": True, "coeff": 1.0}},
                        "regularizer": {"active": True, "coeff": 0.1},
                        "weighting": {"active": False, "mode": "mlp"},
                    }
                }
            }
        )
        out = finalize_association_config(args)
        self.assertEqual(out.model.relations.weighting.mode, "disabled")


if __name__ == "__main__":
    unittest.main()
