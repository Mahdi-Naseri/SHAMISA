import tempfile
import unittest
from pathlib import Path

from data.dataset_kadis700 import KADIS700Dataset
from data.dataset_kadis700_structured import KADIS700StructuredDataset


class KadisPathResolutionTests(unittest.TestCase):
    def test_plain_existing_path_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            img_path = tmp_path / "img.png"
            img_path.write_bytes(b"x")
            resolved = KADIS700Dataset._resolve_ref_image_path(img_path, tmp_path)
            self.assertEqual(resolved, img_path)

    def test_placeholder_path_maps_to_base_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            mapped = base / "KADIS700" / "ref_imgs" / "a.png"
            mapped.parent.mkdir(parents=True, exist_ok=True)
            mapped.write_bytes(b"x")
            placeholder = Path("data_base_path/KADIS700/ref_imgs/a.png")

            out_a = KADIS700Dataset._resolve_ref_image_path(placeholder, base)
            out_b = KADIS700StructuredDataset._resolve_ref_image_path(placeholder, base)
            self.assertEqual(out_a, mapped)
            self.assertEqual(out_b, mapped)


if __name__ == "__main__":
    unittest.main()
