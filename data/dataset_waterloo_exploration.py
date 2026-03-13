"""
Waterloo Exploration Dataset (unlabeled) loader for gMAD scoring.

Usage:
    from data.dataset_waterloo_exploration import WaterlooExplorationDataset

    dataset = WaterlooExplorationDataset(
        root_dir="/path/to/WaterlooExploration",
        return_relpath=True,
        crop_size=224,
        subset="distorted",
    )
"""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils_data import center_corners_crop, resize_crop


class WaterlooExplorationDataset(Dataset):
    """
    Dataset for unlabeled Waterloo Exploration images.

    Args:
        root_dir (str or Path): root directory containing images.
        transform (callable, optional): transform applied to each crop (should not include normalization).
        return_relpath (bool): if True, return paths relative to root_dir.
        crop_size (int): crop size for the 5-crop evaluation.
        subset (str, optional): "pristine" or "distorted" (default: "distorted").

    Returns:
        dict with keys:
            img (Tensor): 5 x 3 x crop_size x crop_size crops
            img_ds (Tensor): downsampled 5-crop tensor
            path (str): identifier string
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[callable] = None,
        return_relpath: bool = True,
        crop_size: int = 224,
        subset: Optional[str] = "distorted",
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Waterloo root not found: {self.root_dir}. "
                "Please provide the dataset root directory."
            )

        self.transform = transform
        self.return_relpath = return_relpath
        self.crop_size = crop_size
        self.subset = subset
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.data_root, self.subset_resolved = self._resolve_subset_root()
        self.images = self._scan_images(self.data_root)
        if not self.images:
            raise FileNotFoundError(
                f"No images found under {self.data_root}. "
                "Expected .bmp/.png/.jpg/.jpeg files in nested folders."
            )

    def _resolve_subset_root(self) -> Tuple[Path, str]:
        requested = self.subset.lower() if isinstance(self.subset, str) else None
        if requested not in (None, "pristine", "distorted"):
            raise ValueError(
                f"Unsupported subset '{self.subset}'. "
                "Expected 'pristine', 'distorted', or None."
            )

        subdirs = {p.name: p for p in self.root_dir.iterdir() if p.is_dir()}
        subdir_names = sorted(subdirs.keys())
        pristine_name = "pristine_images"
        distorted_candidates = (
            "distorted_images",
            "degraded_images",
            "distorted",
            "degraded",
            "distortions",
            "degradations",
        )
        distortion_keywords = (
            "jpeg",
            "jp2",
            "jp2k",
            "blur",
            "noise",
            "fastfading",
            "fading",
            "white",
            "contrast",
            "satur",
            "color",
        )

        def _warn(message: str) -> None:
            print(f"[WaterlooExplorationDataset] WARNING: {message}")

        if requested == "pristine":
            if self.root_dir.name == pristine_name:
                return self.root_dir, "pristine"
            if pristine_name in subdirs:
                return subdirs[pristine_name], "pristine"
            _warn(
                "subset='pristine' but no 'pristine_images' folder found; "
                "falling back to root_dir."
            )
            return self.root_dir, "pristine"

        if requested == "distorted":
            if self.root_dir.name in distorted_candidates:
                return self.root_dir, "distorted"
            for name in distorted_candidates:
                if name in subdirs:
                    return subdirs[name], "distorted"
            has_distortion_subdirs = any(
                any(keyword in name.lower() for keyword in distortion_keywords)
                for name in subdirs
            )
            if pristine_name in subdirs and not has_distortion_subdirs:
                raise FileNotFoundError(
                    "subset='distorted' requested, but only 'pristine_images' "
                    "was found under the Waterloo root. Provide a distorted "
                    "image directory or generate the distorted pool. "
                    f"Detected subfolders: {subdir_names}"
                )
            if not subdirs:
                raise FileNotFoundError(
                    "subset='distorted' requested, but no subfolders were found "
                    f"under {self.root_dir}."
                )
            return self.root_dir, "distorted"

        if pristine_name in subdirs:
            _warn(
                "subset not provided; defaulting to 'pristine_images'. "
                "Set subset='distorted' to enforce the distorted pool."
            )
            return subdirs[pristine_name], "pristine"
        for name in distorted_candidates:
            if name in subdirs:
                _warn(
                    "subset not provided; defaulting to distorted candidate "
                    f"folder '{name}'."
                )
                return subdirs[name], "distorted"
        _warn(
            "subset not provided and no standard subfolders found; "
            "using root_dir as dataset root."
        )
        return self.root_dir, "unknown"

    def _scan_images(self, root_dir: Path):
        exts = {".bmp", ".png", ".jpg", ".jpeg"}
        images = [
            path
            for path in root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in exts
        ]
        return sorted(images, key=lambda p: str(p))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        path = self.images[index]
        img = Image.open(path).convert("RGB")
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        crops = center_corners_crop(img, crop_size=self.crop_size)
        crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)

        if self.transform is None:
            to_tensor = transforms.ToTensor()
            crops = [to_tensor(crop) for crop in crops]
            crops_ds = [to_tensor(crop) for crop in crops_ds]
        else:
            crops = [self.transform(crop) for crop in crops]
            crops_ds = [self.transform(crop) for crop in crops_ds]

        img = torch.stack(crops, dim=0)
        img_ds = torch.stack(crops_ds, dim=0)
        img = self.normalize(img)
        img_ds = self.normalize(img_ds)

        if self.return_relpath:
            identifier = str(path.relative_to(self.root_dir))
        else:
            identifier = str(path)

        return {"img": img, "img_ds": img_ds, "path": identifier}
