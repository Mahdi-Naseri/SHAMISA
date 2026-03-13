import random
from random import randrange
import torchvision.transforms.functional as TF
from typing import List, Callable, Union
from PIL.Image import Image as PILImage
import numpy as np

from utils.distortions import *
import torch


distortion_groups = {
    "blur": ["gaublur", "lensblur", "motionblur"],
    "color_distortion": ["colordiff", "colorshift", "colorsat1", "colorsat2"],
    "jpeg": ["jpeg2000", "jpeg"],
    "noise": ["whitenoise", "whitenoiseCC", "impulsenoise", "multnoise"],
    "brightness_change": ["brighten", "darken", "meanshiftneg", "meanshiftpos"],
    "spatial_distortion": [
        "jitter",
        "noneccpatch",
        "pixelate",
        "quantization",
        "colorblock",
    ],
    "sharpness_contrast": [
        "highsharpen",
        "lincontrchangeneg",
        "lincontrchangepos",
        "nonlincontrchange",
    ],
}

distortion_groups_mapping = {
    "gaublur": "blur",
    "lensblur": "blur",
    "motionblur": "blur",
    "colordiff": "color_distortion",
    "colorshift": "color_distortion",
    "colorsat1": "color_distortion",
    "colorsat2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "whitenoise": "noise",
    "whitenoiseCC": "noise",
    "impulsenoise": "noise",
    "multnoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "meanshiftneg": "brightness_change",
    "meanshiftpos": "brightness_change",
    "jitter": "spatial_distortion",
    "noneccpatch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "colorblock": "spatial_distortion",
    "highsharpen": "sharpness_contrast",
    "lincontrchangeneg": "sharpness_contrast",
    "lincontrchangepos": "sharpness_contrast",
    "nonlincontrchangepos": "sharpness_contrast",
}

distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "motionblur": [1, 2, 4, 6, 10],
    "colordiff": [1, 3, 6, 8, 12],
    "colorshift": [1, 3, 6, 8, 12],
    "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    "colorsat2": [1, 2, 3, 6, 9],
    "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],
    "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
    "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],
    "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    "meanshiftneg": [0, -0.08, -0.115, -0.15, -0.185],
    "meanshiftpos": [0, 0.08, 0.115, 0.15, 0.185],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],
    "noneccpatch": [20, 40, 60, 80, 100],
    "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],
    "quantization": [20, 16, 13, 10, 7],
    "colorblock": [2, 4, 6, 8, 10],
    "highsharpen": [1, 2, 3, 6, 12],
    "lincontrchangeneg": [0.0, -0.15, -0.3, -0.4, -0.6],
    "lincontrchangepos": [0.0, 0.15, 0.3, 0.4, 0.6],
    "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
}

forced_int_distortions = [
    "lensblur",
    "motionblur",
    "colordiff",
    "colorshift",
    "colorsat2",
    "jpeg2000",
    "jpeg",
    "noneccpatch",
    "quantization",
    "highsharpen",
    "colorblock",
]
main_int_distortions = ["colorblock", "lensblur", "jpeg", "quantization", "noneccpatch"]

distortion_functions = {
    "gaublur": gaussian_blur,
    "lensblur": lens_blur,
    "motionblur": motion_blur,
    "colordiff": color_diffusion,
    "colorshift": color_shift,
    "colorsat1": color_saturation1,
    "colorsat2": color_saturation2,
    "jpeg2000": jpeg2000,
    "jpeg": jpeg,
    "whitenoise": white_noise,
    "whitenoiseCC": white_noise_cc,
    "impulsenoise": impulse_noise,
    "multnoise": multiplicative_noise,
    "brighten": brighten,
    "darken": darken,
    "meanshiftneg": mean_shift,
    "meanshiftpos": mean_shift,
    "jitter": jitter,
    "noneccpatch": non_eccentricity_patch,
    "pixelate": pixelate,
    "quantization": quantization,
    "colorblock": color_block,
    "highsharpen": high_sharpen,
    "lincontrchangeneg": linear_contrast_change,
    "lincontrchangepos": linear_contrast_change,
    "nonlincontrchange": non_linear_contrast_change,
}


def distort_images(
    image: torch.Tensor,
    distort_functions: list = None,
    distort_values: list = None,
    # max_distortions: int = 4,
    # num_levels: int = 5,
) -> torch.Tensor:
    """
    Distorts an image using the distortion composition obtained with the image degradation model proposed in the paper
    https://arxiv.org/abs/2310.14918.

    Args:
        image (Tensor): image to distort
        distort_functions (list): list of the distortion functions to apply to the image. If None, the functions are randomly chosen.
        distort_values (list): list of the values of the distortion functions to apply to the image. If None, the values are randomly chosen.
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        image (Tensor): distorted image
        distort_functions (list): list of the distortion functions applied to the image
        distort_values (list): list of the values of the distortion functions applied to the image
    """
    # if distort_functions is None or distort_values is None:
    #     distort_functions, distort_values = get_distortions_composition(
    #         max_distortions, num_levels
    #     )
    image = image.clone()
    for distortion, value in zip(distort_functions, distort_values):
        if distortion:
            image = distortion(image, value)
            if image.dtype != torch.float32:
                image = image.to(torch.float32)
            image = image.clamp_(0, 1)

    return image, distort_functions, distort_values


def sample_distortion(
    dist,
    extended_int_distortions,
    severity_discrete=False,
    severity_dist="gaussian",
    num_levels=None,
):
    if dist is None:
        return -1, -1

    MEAN = 0
    STD = 2.5
    values = distortion_range[dist]
    if severity_discrete and num_levels is not None:
        values = values[: int(num_levels)]

    # while True:
    #     index = np.abs(np.random.normal(MEAN, STD))
    #     if index <= len(values) - 1:
    #         break

    if isinstance(severity_dist, str):
        severity_dist = severity_dist.lower()
    if severity_dist == "uniform":
        clamped_idx = np.random.uniform(0, len(values) - 1)
    elif severity_dist == "gaussian":
        clamped_idx = np.clip(np.abs(np.random.normal(MEAN, STD)), 0, len(values) - 1)
    else:
        raise ValueError(f"Unsupported severity_dist: {severity_dist}")

    if severity_discrete:
        rounded_idx = int(np.round(clamped_idx))
        rounded_idx = int(np.clip(rounded_idx, 0, len(values) - 1))
        return float(rounded_idx), values[rounded_idx]

    # Get the lower and upper indices for interpolation
    lower_idx = int(np.floor(clamped_idx))
    upper_idx = int(np.ceil(clamped_idx))

    sampled_idx = clamped_idx

    if lower_idx == upper_idx:
        # If the index is an integer, no interpolation is needed
        sampled_val = values[lower_idx]
    else:
        # Interpolate between the two closest values
        lower_idx_val = values[lower_idx]
        upper_idx_val = values[upper_idx]  # could be lower than lower value
        fraction = clamped_idx - lower_idx
        sampled_val = lower_idx_val * (1 - fraction) + upper_idx_val * fraction

        run_int_distortion = (
            forced_int_distortions if extended_int_distortions else main_int_distortions
        )
        if dist in run_int_distortion:
            sampled_val = round(sampled_val)

            fraction = (sampled_val - lower_idx_val) / (upper_idx_val - lower_idx_val)
            sampled_idx = lower_idx * (1 - fraction) + upper_idx * fraction

    return sampled_idx, sampled_val


def get_distortions_composition(
    max_distortions: int = 7,
    num_levels: int = 5,
    n_dist_comp_levels=6,
    extended_int_distortions=False,
    severity_discrete=False,
    severity_dist="gaussian",
    fixed_order=False,
) -> (List[Callable], List[Union[int, float]]):
    """
    Image Degradation model proposed in the paper https://arxiv.org/abs/2310.14918. Returns a randomly assembled ordered
    sequence of distortion functions and their values.

    Args:
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        distort_functions (list): list of the distortion functions to apply to the image
        distort_values (list): list of the values of the distortion functions to apply to the image
    """
    # MEAN = 0
    # STD = 2.5

    num_distortions = random.randint(1, max_distortions)
    if fixed_order:
        ordered_groups = sorted(distortion_groups.keys())
        groups = ordered_groups[:num_distortions]
        ordered_distortions = {
            group: sorted(distortion_groups[group]) for group in ordered_groups
        }
        distortions = [
            ordered_distortions[groups[i]][0] if i < num_distortions else None
            for i in range(max_distortions)
        ]
    else:
        groups = random.sample(list(distortion_groups.keys()), num_distortions)
        distortions = [
            random.choice(distortion_groups[groups[i]]) if i < num_distortions else None
            for i in range(max_distortions)
        ]
    distort_functions = [
        distortion_functions[dist] if dist else None for dist in distortions
    ]

    # probabilities = [
    #     1 / (STD * np.sqrt(2 * np.pi)) * np.exp(-((i - MEAN) ** 2) / (2 * STD**2))
    #     for i in range(num_levels)
    # ]  # probabilities according to a gaussian distribution;
    # these probabilites are used to select a distortione level for each distortion
    # normalized_probabilities = [
    # prob / sum(probabilities) for prob in probabilities
    # ]  # normalize probabilities

    # Select the variable distortion index & value efficiently
    variable_dist = np.random.randint(num_distortions)
    variable_dist_name = distortions[variable_dist]

    # Precompute all necessary variable_dist samples at once (batch processing)
    variable_dist_samples = np.array(
        [
            sample_distortion(
                variable_dist_name,
                extended_int_distortions,
                severity_discrete=severity_discrete,
                severity_dist=severity_dist,
                num_levels=num_levels if severity_discrete else None,
            )
            for _ in range(n_dist_comp_levels)
        ],
        dtype=np.float32,
    )

    # Baseline sampling for all distortions (one-time computation)
    dist_samples_0 = np.array(
        [
            sample_distortion(
                dist,
                extended_int_distortions,
                severity_discrete=severity_discrete,
                severity_dist=severity_dist,
                num_levels=num_levels if severity_discrete else None,
            )
            for dist in distortions
        ],
        dtype=np.float32,
    )

    # Expand dist_samples_0 to match the shape of `composition_indices`
    composition_indices = np.tile(dist_samples_0[:, 0], (n_dist_comp_levels, 1))
    composition_values = np.tile(dist_samples_0[:, 1], (n_dist_comp_levels, 1))

    # Replace only the column corresponding to `variable_dist_index`
    composition_indices[:, variable_dist] = variable_dist_samples[:, 0]
    composition_values[:, variable_dist] = variable_dist_samples[:, 1]

    return [
        distort_functions,
        composition_indices,
        composition_values,
        num_distortions,
        variable_dist,
    ]


def resize_crop(
    img: PILImage, crop_size: int = 224, downscale_factor: int = 1
) -> PILImage:
    """
    Resize the image with the desired downscale factor and optionally crop it to the desired size. The crop is randomly
    sampled from the image. If crop_size is None, no crop is applied. If the crop is out of bounds, the image is
    automatically padded with zeros.

    Args:
        img (PIL Image): image to resize and crop
        crop_size (int): size of the crop. If None, no crop is applied
        downscale_factor (int): downscale factor to apply to the image

    Returns:
        img (PIL Image): resized and/or cropped image
    """
    w, h = img.size
    if downscale_factor > 1:
        img = img.resize((w // downscale_factor, h // downscale_factor))
        w, h = img.size

    if crop_size is not None:
        top = randrange(0, max(1, h - crop_size))
        left = randrange(0, max(1, w - crop_size))
        img = TF.crop(
            img, top, left, crop_size, crop_size
        )  # Automatically pad with zeros if the crop is out of bounds

    return img


def center_corners_crop(img: PILImage, crop_size: int = 224) -> List[PILImage]:
    """
    Return the center crop and the four corners of the image.

    Args:
        img (PIL.Image): image to crop
        crop_size (int): size of each crop

    Returns:
        crops (List[PIL.Image]): list of the five crops
    """
    width, height = img.size

    # Calculate the coordinates for the center crop and the four corners
    cx = width // 2
    cy = height // 2
    crops = [
        TF.crop(
            img, cy - crop_size // 2, cx - crop_size // 2, crop_size, crop_size
        ),  # Center
        TF.crop(img, 0, 0, crop_size, crop_size),  # Top-left corner
        TF.crop(img, height - crop_size, 0, crop_size, crop_size),  # Bottom-left corner
        TF.crop(img, 0, width - crop_size, crop_size, crop_size),  # Top-right corner
        TF.crop(
            img, height - crop_size, width - crop_size, crop_size, crop_size
        ),  # Bottom-right corner
    ]

    return crops
