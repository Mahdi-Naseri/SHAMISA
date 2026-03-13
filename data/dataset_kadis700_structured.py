import os
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
import pandas as pd

from utils.utils_data import resize_crop, get_distortions_composition
from utils.utils import PROJECT_ROOT


class KADIS700StructuredDataset(Dataset):
    """
    KADIS700 dataset class used for pre-training the encoders for IQA.

    Args:
        root (string): root directory of the dataset
        patch_size (int): size of the patches to extract from the images
        max_distortions (int): maximum number of distortions to apply to the images
        num_levels (int): number of levels of distortion to apply to the images
        pristine_prob (float): probability of not distorting the images

    Returns:
        dictionary with keys:
            img_A_orig (Tensor): first view of the image pair
            img_A_ds (Tensor): downsampled version of the first view of the image pair (scale factor 2)
            img_B_orig (Tensor): second view of the image pair
            img_B_ds (Tensor): downsampled version of the second view of the image pair (scale factor 2)
            img_A_name (string): name of the image of the first view of the image pair
            img_B_name (string): name of the image of the second view of the image pair
            distortion_functions (list): list of the names of the distortion functions applied to the images
            distortion_values (list): list of the values of the distortion functions applied to the images
    """

    def __init__(
        self,
        root: str,
        patch_size: int = 224,
        max_distortions: int = 4,
        num_levels: int = 5,
        n_refs=3,
        n_dist_comps=4,
        n_dist_comp_levels=6,  # number of levels for each distortion composition
        extended_int_distortions=False,
        severity_discrete=False,
        severity_dist="gaussian",
        fixed_order=False,
        cache_path: str = None,
        cache_mode: str = "na",
        epoch: int = 0,
        cache_load_max_avail_epoch=None,
        cache_load_mode: str = "mmap",
        no_cache_opt_variant: str = "variant_b",
    ):
        root = Path(root)
        base_root = root.parent
        filenames_csv_path = PROJECT_ROOT / "data" / "synthetic_filenames.csv"
        if not filenames_csv_path.exists():
            self._generate_filenames_csv(root, filenames_csv_path)
        df = pd.read_csv(filenames_csv_path)
        self.ref_images = df["Filename"].tolist()
        self.ref_images = [
            self._resolve_ref_image_path(Path(img), base_root) for img in self.ref_images
        ]

        self.patch_size = patch_size
        self.max_distortions = max_distortions
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.num_levels = num_levels
        self.n_refs = n_refs
        self.n_dist_comps = n_dist_comps
        self.n_dist_comp_levels = n_dist_comp_levels
        self.extended_int_distortions = extended_int_distortions
        self.severity_discrete = severity_discrete
        self.severity_dist = severity_dist
        self.fixed_order = fixed_order
        self.cache_path = cache_path
        self.cache_mode = cache_mode
        self.epoch = epoch
        self.cache_load_max_avail_epoch = cache_load_max_avail_epoch
        if self.cache_load_max_avail_epoch is not None:
            self.cache_load_max_avail_epoch = int(self.cache_load_max_avail_epoch)
        self.cache_load_mode = str(cache_load_mode).lower()
        if self.cache_load_mode not in {"default", "mmap"}:
            raise ValueError("cache_load_mode must be one of: default, mmap")
        self.no_cache_opt_variant = str(no_cache_opt_variant).lower()
        if self.no_cache_opt_variant not in {
            "baseline",
            "variant_a",
            "variant_b",
            "variant_d",
            "variant_e",
        }:
            raise ValueError(
                "no_cache_opt_variant must be one of: baseline, variant_a, variant_b, variant_d, variant_e"
            )

        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
            3, 1, 1
        )
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
            3, 1, 1
        )

        assert (
            0 <= self.max_distortions <= 7
        ), "The parameter max_distortions must be in the range [0, 7]"
        assert (
            1 <= self.num_levels <= 5
        ), "The parameter num_levels must be in the range [1, 5]"

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_cache_filename(self, index):
        """
        Generate a sharded cache filename to avoid storing too many files in a single directory.
        Uses 2-level sharding based on the zero-padded index.

        Example:
            index = 123 -> cache_path/00/01/epoch_0_index_123.pt
        """
        index_str = f"{index:06d}"
        subdir1 = index_str[:2]
        subdir2 = index_str[2:4]

        cur_epoch = self.epoch
        if self.cache_mode == "load" and self.cache_load_max_avail_epoch is not None:
            cur_epoch %= self.cache_load_max_avail_epoch + 1

        filename = f"epoch_{cur_epoch}_index_{index}.pt"
        full_path = os.path.join(self.cache_path, subdir1, subdir2, filename)

        if self.cache_mode in {"save", "load"}:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

        return full_path

    def _safe_cache_save(self, output_item: dict, cache_file: str, max_retries: int = 3) -> None:
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        pid = os.getpid()
        for attempt in range(max_retries):
            tmp_file = f"{cache_file}.tmp.{pid}.{attempt}"
            try:
                torch.save(output_item, tmp_file)
                os.replace(tmp_file, cache_file)
                return
            except Exception:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass
                if attempt + 1 == max_retries:
                    raise
                time.sleep(0.2 * (attempt + 1))

    def _to_float_tensor(self, img: Image.Image) -> torch.Tensor:
        img_t = transforms.functional.pil_to_tensor(img)
        return img_t.to(dtype=torch.float32).div_(255.0)

    def _normalize_inplace(self, img: torch.Tensor) -> torch.Tensor:
        img.sub_(self._norm_mean).div_(self._norm_std)
        return img

    def _normalize_batch_inplace(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs.sub_(self._norm_mean).div_(self._norm_std)
        return imgs

    @staticmethod
    def _apply_distortion_chain(
        img: torch.Tensor,
        distort_functions,
        distort_values,
        clone_input: bool = True,
    ) -> torch.Tensor:
        out = img.clone() if clone_input else img
        for distortion, value in zip(distort_functions, distort_values):
            if distortion:
                out = distortion(out, value)
                if out.dtype != torch.float32:
                    out = out.to(torch.float32)
                out = torch.clamp(out, 0, 1)
        return out

    @staticmethod
    def _apply_distortion_pair(
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        distort_functions,
        distort_values,
        clone_input: bool = True,
    ):
        out_a = img_a.clone() if clone_input else img_a
        out_b = img_b.clone() if clone_input else img_b
        for distortion, value in zip(distort_functions, distort_values):
            if distortion:
                out_a = distortion(out_a, value)
                if out_a.dtype != torch.float32:
                    out_a = out_a.to(torch.float32)
                out_a = torch.clamp(out_a, 0, 1)

                out_b = distortion(out_b, value)
                if out_b.dtype != torch.float32:
                    out_b = out_b.to(torch.float32)
                out_b = torch.clamp(out_b, 0, 1)
        return out_a, out_b

    def _sample_distortion_compositions(self):
        return [
            get_distortions_composition(
                self.max_distortions,
                self.num_levels,
                self.n_dist_comp_levels,
                self.extended_int_distortions,
                self.severity_discrete,
                self.severity_dist,
                self.fixed_order,
            )
            for _ in range(self.n_dist_comps)
        ]

    def _build_metadata_tensors(self, dist_comps):
        composition_indices = torch.from_numpy(np.stack([entry[1] for entry in dist_comps]))
        composition_values = torch.from_numpy(np.stack([entry[2] for entry in dist_comps]))
        num_distortions = torch.from_numpy(
            np.asarray([entry[3] for entry in dist_comps], dtype=np.int64)
        )
        variable_dist = torch.from_numpy(
            np.asarray([entry[4] for entry in dist_comps], dtype=np.int64)
        )
        return composition_indices, composition_values, num_distortions, variable_dist

    def _build_output_item(self, imgs_names, imgs, imgs_downsampled, refs, refs_downsampled, dist_comps):
        (
            composition_indices,
            composition_values,
            num_distortions,
            variable_dist,
        ) = self._build_metadata_tensors(dist_comps)

        return {
            "imgs_names": imgs_names,
            "dist_imgs": torch.cat((imgs, imgs_downsampled), dim=0),
            "ref_imgs": torch.cat((refs, refs_downsampled), dim=0),
            "composition_indices": composition_indices,
            "composition_values": composition_values,
            "num_distortions": num_distortions,
            "variable_dist": variable_dist,
        }

    def _build_item_baseline(self, index: int) -> dict:
        dist_comps = self._sample_distortion_compositions()
        refs = torch.zeros(
            (
                self.n_refs,
                3,
                self.patch_size,
                self.patch_size,
            )
        )
        refs_downsampled = torch.zeros_like(refs)

        imgs = torch.zeros(
            (
                self.n_refs,
                self.n_dist_comps,
                self.n_dist_comp_levels,
                3,
                self.patch_size,
                self.patch_size,
            )
        )
        imgs_downsampled = torch.zeros_like(imgs)
        imgs_names = []
        for i in range(self.n_refs):
            img_idx = index if i == 0 else random.randint(0, len(self.ref_images) - 1)
            img_pil = Image.open(self.ref_images[img_idx]).convert("RGB")
            img_name = self.ref_images[img_idx].stem
            imgs_names.append(img_name)
            img_orig_pil = resize_crop(img_pil, self.patch_size)
            img_downsampled_pil = resize_crop(
                img_pil, self.patch_size, downscale_factor=2
            )

            img_orig = transforms.ToTensor()(img_orig_pil)
            img_downsampled = transforms.ToTensor()(img_downsampled_pil)

            for j, dist_comp in enumerate(dist_comps):
                distort_functions, _, composition_values, _, _ = dist_comp
                for k, distort_values in enumerate(composition_values):
                    img_orig_distorted = self._apply_distortion_chain(
                        img_orig,
                        distort_functions=distort_functions,
                        distort_values=distort_values,
                    )
                    img_downsampled_distorted = self._apply_distortion_chain(
                        img_downsampled,
                        distort_functions=distort_functions,
                        distort_values=distort_values,
                    )
                    img_orig_distorted = self.normalize(img_orig_distorted)
                    img_downsampled_distorted = self.normalize(
                        img_downsampled_distorted
                    )

                    imgs[i, j, k] = img_orig_distorted
                    imgs_downsampled[i, j, k] = img_downsampled_distorted

            refs[i] = self.normalize(img_orig)
            refs_downsampled[i] = self.normalize(img_downsampled)

        return self._build_output_item(
            imgs_names=imgs_names,
            imgs=imgs,
            imgs_downsampled=imgs_downsampled,
            refs=refs,
            refs_downsampled=refs_downsampled,
            dist_comps=dist_comps,
        )

    def _build_item_variant_a(self, index: int) -> dict:
        dist_comps = self._sample_distortion_compositions()
        refs = torch.empty(
            (
                self.n_refs,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )
        refs_downsampled = torch.empty_like(refs)
        imgs = torch.empty(
            (
                self.n_refs,
                self.n_dist_comps,
                self.n_dist_comp_levels,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )
        imgs_downsampled = torch.empty_like(imgs)
        imgs_names = []

        n_ref_images = len(self.ref_images)
        for i in range(self.n_refs):
            img_idx = index if i == 0 else random.randint(0, n_ref_images - 1)
            img_path = self.ref_images[img_idx]
            imgs_names.append(img_path.stem)
            with Image.open(img_path) as img_pil:
                img_pil = img_pil.convert("RGB")
                img_orig = self._to_float_tensor(resize_crop(img_pil, self.patch_size))
                img_downsampled = self._to_float_tensor(
                    resize_crop(img_pil, self.patch_size, downscale_factor=2)
                )

            refs[i] = self._normalize_inplace(img_orig.clone())
            refs_downsampled[i] = self._normalize_inplace(img_downsampled.clone())

            for j, dist_comp in enumerate(dist_comps):
                distort_functions, _, composition_values, _, _ = dist_comp
                for k, distort_values in enumerate(composition_values):
                    img_orig_distorted = self._apply_distortion_chain(
                        img_orig,
                        distort_functions=distort_functions,
                        distort_values=distort_values,
                    )
                    img_downsampled_distorted = self._apply_distortion_chain(
                        img_downsampled,
                        distort_functions=distort_functions,
                        distort_values=distort_values,
                    )
                    imgs[i, j, k] = self._normalize_inplace(img_orig_distorted)
                    imgs_downsampled[i, j, k] = self._normalize_inplace(
                        img_downsampled_distorted
                    )

        return self._build_output_item(
            imgs_names=imgs_names,
            imgs=imgs,
            imgs_downsampled=imgs_downsampled,
            refs=refs,
            refs_downsampled=refs_downsampled,
            dist_comps=dist_comps,
        )

    def _build_levels_with_prefix_reuse(
        self,
        img: torch.Tensor,
        distort_functions,
        composition_values,
        num_distortions: int,
        variable_dist: int,
    ) -> torch.Tensor:
        levels_out = torch.empty(
            (
                self.n_dist_comp_levels,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )

        active_n = int(num_distortions)
        if active_n <= 0:
            normalized = self._normalize_inplace(img.clone())
            levels_out[:] = normalized
            return levels_out

        var_idx = int(variable_dist)
        if var_idx < 0 or var_idx >= active_n:
            for level_idx, distort_values in enumerate(composition_values):
                distorted = self._apply_distortion_chain(
                    img,
                    distort_functions=distort_functions,
                    distort_values=distort_values,
                )
                levels_out[level_idx] = self._normalize_inplace(distorted)
            return levels_out

        active_functions = distort_functions[:active_n]
        base_values = composition_values[0][:active_n]

        prefix_functions = active_functions[:var_idx]
        prefix_values = base_values[:var_idx]
        variable_function = active_functions[var_idx]
        suffix_functions = active_functions[var_idx + 1 :]
        suffix_values = base_values[var_idx + 1 :]

        prefix_img = self._apply_distortion_chain(
            img,
            distort_functions=prefix_functions,
            distort_values=prefix_values,
        )

        for level_idx in range(self.n_dist_comp_levels):
            level_img = prefix_img.clone()
            if variable_function:
                level_img = variable_function(level_img, composition_values[level_idx][var_idx])
                if level_img.dtype != torch.float32:
                    level_img = level_img.to(torch.float32)
                level_img = torch.clamp(level_img, 0, 1)
            level_img = self._apply_distortion_chain(
                level_img,
                distort_functions=suffix_functions,
                distort_values=suffix_values,
                clone_input=False,
            )
            levels_out[level_idx] = self._normalize_inplace(level_img)

        return levels_out

    def _build_item_variant_b(self, index: int) -> dict:
        dist_comps = self._sample_distortion_compositions()
        refs = torch.empty(
            (
                self.n_refs,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )
        refs_downsampled = torch.empty_like(refs)
        imgs = torch.empty(
            (
                self.n_refs,
                self.n_dist_comps,
                self.n_dist_comp_levels,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )
        imgs_downsampled = torch.empty_like(imgs)
        imgs_names = []

        n_ref_images = len(self.ref_images)
        for i in range(self.n_refs):
            img_idx = index if i == 0 else random.randint(0, n_ref_images - 1)
            img_path = self.ref_images[img_idx]
            imgs_names.append(img_path.stem)
            with Image.open(img_path) as img_pil:
                img_pil = img_pil.convert("RGB")
                img_orig = self._to_float_tensor(resize_crop(img_pil, self.patch_size))
                img_downsampled = self._to_float_tensor(
                    resize_crop(img_pil, self.patch_size, downscale_factor=2)
                )

            refs[i] = self._normalize_inplace(img_orig.clone())
            refs_downsampled[i] = self._normalize_inplace(img_downsampled.clone())

            for j, dist_comp in enumerate(dist_comps):
                (
                    distort_functions,
                    _,
                    composition_values,
                    num_distortions,
                    variable_dist,
                ) = dist_comp
                imgs[i, j] = self._build_levels_with_prefix_reuse(
                    img_orig,
                    distort_functions=distort_functions,
                    composition_values=composition_values,
                    num_distortions=num_distortions,
                    variable_dist=variable_dist,
                )
                imgs_downsampled[i, j] = self._build_levels_with_prefix_reuse(
                    img_downsampled,
                    distort_functions=distort_functions,
                    composition_values=composition_values,
                    num_distortions=num_distortions,
                    variable_dist=variable_dist,
                )

        return self._build_output_item(
            imgs_names=imgs_names,
            imgs=imgs,
            imgs_downsampled=imgs_downsampled,
            refs=refs,
            refs_downsampled=refs_downsampled,
            dist_comps=dist_comps,
        )

    def _prepare_dist_comp_plan(self, dist_comp):
        (
            distort_functions,
            _,
            composition_values,
            num_distortions,
            variable_dist,
        ) = dist_comp
        active_n = int(num_distortions)
        var_idx = int(variable_dist)

        plan = {
            "mode": "prefix",
            "distort_functions": distort_functions,
            "composition_values": composition_values,
            "active_n": active_n,
            "var_idx": var_idx,
            "prefix_functions": None,
            "prefix_values": None,
            "variable_function": None,
            "variable_values": None,
            "suffix_functions": None,
            "suffix_values": None,
        }

        if active_n <= 0:
            plan["mode"] = "none"
            return plan

        if var_idx < 0 or var_idx >= active_n:
            plan["mode"] = "fallback"
            return plan

        active_functions = distort_functions[:active_n]
        base_values = composition_values[0][:active_n]
        plan["prefix_functions"] = active_functions[:var_idx]
        plan["prefix_values"] = base_values[:var_idx]
        plan["variable_function"] = active_functions[var_idx]
        plan["variable_values"] = composition_values[:, var_idx]
        plan["suffix_functions"] = active_functions[var_idx + 1 :]
        plan["suffix_values"] = base_values[var_idx + 1 :]
        return plan

    def _build_levels_from_plan_pair(
        self,
        img_orig: torch.Tensor,
        img_downsampled: torch.Tensor,
        plan,
    ):
        levels_orig = torch.empty(
            (
                self.n_dist_comp_levels,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )
        levels_downsampled = torch.empty_like(levels_orig)

        mode = plan["mode"]
        if mode == "none":
            levels_orig[:] = img_orig
            levels_downsampled[:] = img_downsampled
            self._normalize_batch_inplace(levels_orig)
            self._normalize_batch_inplace(levels_downsampled)
            return levels_orig, levels_downsampled

        if mode == "fallback":
            distort_functions = plan["distort_functions"]
            composition_values = plan["composition_values"]
            for level_idx, distort_values in enumerate(composition_values):
                level_orig, level_downsampled = self._apply_distortion_pair(
                    img_orig,
                    img_downsampled,
                    distort_functions=distort_functions,
                    distort_values=distort_values,
                )
                levels_orig[level_idx] = level_orig
                levels_downsampled[level_idx] = level_downsampled

            self._normalize_batch_inplace(levels_orig)
            self._normalize_batch_inplace(levels_downsampled)
            return levels_orig, levels_downsampled

        prefix_orig, prefix_downsampled = self._apply_distortion_pair(
            img_orig,
            img_downsampled,
            distort_functions=plan["prefix_functions"],
            distort_values=plan["prefix_values"],
        )

        variable_function = plan["variable_function"]
        variable_values = plan["variable_values"]
        suffix_functions = plan["suffix_functions"]
        suffix_values = plan["suffix_values"]

        for level_idx in range(self.n_dist_comp_levels):
            level_orig = prefix_orig.clone()
            level_downsampled = prefix_downsampled.clone()

            if variable_function:
                variable_value = variable_values[level_idx]
                level_orig = variable_function(level_orig, variable_value)
                if level_orig.dtype != torch.float32:
                    level_orig = level_orig.to(torch.float32)
                level_orig = torch.clamp(level_orig, 0, 1)

                level_downsampled = variable_function(level_downsampled, variable_value)
                if level_downsampled.dtype != torch.float32:
                    level_downsampled = level_downsampled.to(torch.float32)
                level_downsampled = torch.clamp(level_downsampled, 0, 1)

            level_orig, level_downsampled = self._apply_distortion_pair(
                level_orig,
                level_downsampled,
                distort_functions=suffix_functions,
                distort_values=suffix_values,
                clone_input=False,
            )
            levels_orig[level_idx] = level_orig
            levels_downsampled[level_idx] = level_downsampled

        self._normalize_batch_inplace(levels_orig)
        self._normalize_batch_inplace(levels_downsampled)
        return levels_orig, levels_downsampled

    def _build_item_variant_d(self, index: int, precompute_plans: bool = False) -> dict:
        dist_comps = self._sample_distortion_compositions()
        plans = (
            [self._prepare_dist_comp_plan(dist_comp) for dist_comp in dist_comps]
            if precompute_plans
            else None
        )

        refs = torch.empty(
            (
                self.n_refs,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )
        refs_downsampled = torch.empty_like(refs)
        imgs = torch.empty(
            (
                self.n_refs,
                self.n_dist_comps,
                self.n_dist_comp_levels,
                3,
                self.patch_size,
                self.patch_size,
            ),
            dtype=torch.float32,
        )
        imgs_downsampled = torch.empty_like(imgs)
        imgs_names = []

        n_ref_images = len(self.ref_images)
        for i in range(self.n_refs):
            img_idx = index if i == 0 else random.randint(0, n_ref_images - 1)
            img_path = self.ref_images[img_idx]
            imgs_names.append(img_path.stem)
            with Image.open(img_path) as img_pil:
                img_pil = img_pil.convert("RGB")
                img_orig = self._to_float_tensor(resize_crop(img_pil, self.patch_size))
                img_downsampled = self._to_float_tensor(
                    resize_crop(img_pil, self.patch_size, downscale_factor=2)
                )

            refs[i] = self._normalize_inplace(img_orig.clone())
            refs_downsampled[i] = self._normalize_inplace(img_downsampled.clone())

            for j in range(self.n_dist_comps):
                plan = plans[j] if plans is not None else self._prepare_dist_comp_plan(dist_comps[j])
                levels_orig, levels_downsampled = self._build_levels_from_plan_pair(
                    img_orig=img_orig,
                    img_downsampled=img_downsampled,
                    plan=plan,
                )
                imgs[i, j] = levels_orig
                imgs_downsampled[i, j] = levels_downsampled

        return self._build_output_item(
            imgs_names=imgs_names,
            imgs=imgs,
            imgs_downsampled=imgs_downsampled,
            refs=refs,
            refs_downsampled=refs_downsampled,
            dist_comps=dist_comps,
        )

    def _build_item_variant_e(self, index: int) -> dict:
        return self._build_item_variant_d(index=index, precompute_plans=True)

    def __getitem__(self, index: int) -> dict:
        cache_file = None
        should_save = False
        if self.cache_mode in {"save", "load"}:
            cache_file = self._get_cache_filename(index)
            should_save = self.cache_mode == "save"
            if self.cache_mode == "load":
                if os.path.exists(cache_file):
                    load_kwargs = {"weights_only": False}
                    if self.cache_load_mode == "mmap":
                        load_kwargs["mmap"] = True
                    return torch.load(cache_file, **load_kwargs)
                print(f"WARNING: cache_file does not exist: {cache_file}")
                should_save = True

        if self.no_cache_opt_variant == "baseline":
            output_item = self._build_item_baseline(index)
        elif self.no_cache_opt_variant == "variant_a":
            output_item = self._build_item_variant_a(index)
        elif self.no_cache_opt_variant == "variant_b":
            output_item = self._build_item_variant_b(index)
        elif self.no_cache_opt_variant == "variant_d":
            output_item = self._build_item_variant_d(index, precompute_plans=False)
        elif self.no_cache_opt_variant == "variant_e":
            output_item = self._build_item_variant_e(index)
        else:
            raise ValueError(
                f"Unsupported no_cache_opt_variant: {self.no_cache_opt_variant}"
            )

        if should_save and cache_file is not None:
            self._safe_cache_save(output_item, cache_file)

        return output_item

    def __len__(self) -> int:
        return len(self.ref_images)

    def _generate_filenames_csv(self, root: Path, csv_path: Path) -> None:
        """
        Generates a CSV file with the filenames of the images for faster preprocessing.
        """
        images = list((root / "ref_imgs").glob("*.png"))
        df = pd.DataFrame(images, columns=["Filename"])
        df.to_csv(csv_path, index=False)

    @staticmethod
    def _resolve_ref_image_path(path: Path, data_base_root: Path) -> Path:
        if path.exists():
            return path
        if path.parts and path.parts[0] == "data_base_path":
            remapped = data_base_root.joinpath(*path.parts[1:])
            if remapped.exists():
                return remapped
        return path


def structured_kadis_collate(batch):
    dist_imgs = torch.stack([item["dist_imgs"] for item in batch])
    ref_imgs = torch.stack([item["ref_imgs"] for item in batch])

    imgs_names = [item["imgs_names"] for item in batch]

    def _to_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x)

    dist_comps = dict()
    dist_comps["indices"] = torch.stack(
        [_to_tensor(item["composition_indices"]) for item in batch]
    )
    dist_comps["vals"] = torch.stack(
        [_to_tensor(item["composition_values"]) for item in batch]
    )
    dist_comps["n_dist"] = torch.stack(
        [_to_tensor(item["num_distortions"]) for item in batch]
    )
    dist_comps["var_dist"] = torch.stack(
        [_to_tensor(item["variable_dist"]) for item in batch]
    )

    return {
        "imgs_names": imgs_names,
        "dist_imgs": dist_imgs,
        "dist_comps": dist_comps,
        "ref_imgs": ref_imgs,
    }
