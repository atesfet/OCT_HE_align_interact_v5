from __future__ import annotations

"""Self-contained 2D HE-to-OCT registration for batch use and the app.

The preprocessing/stain-standardization utilities are source-based and readable,
while the registration search/scoring follows the recovered 2D_v3 behavior more
closely: partial-boundary matching, overlap feasibility checks, and compact
scale/rotation/translation candidate refinement.
"""

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage, optimize
from skimage import color, exposure, filters, measure, morphology, transform

import align_he_to_oct_2D_v3 as _v3_reference


Image.MAX_IMAGE_PIXELS = None

_V3 = _v3_reference._impl

V5_CACHE_ROOT = Path(os.environ.get("OCTHE_V5_2D_CACHE", Path.home() / ".cache" / "octhe_v5_2d"))
REMBG_CACHE_ROOT = V5_CACHE_ROOT / "rembg"
NUMBA_CACHE_ROOT = V5_CACHE_ROOT / "numba"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "samples"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "coregistration_outputs" / "align_he_to_oct_2D_v5"

MACENKO_TARGET_STAINS = np.array(
    [
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ],
    dtype=np.float32,
)
MACENKO_TARGET_MAX_CONC = np.array([1.9705, 1.0308], dtype=np.float32)


@dataclass
class PairPaths:
    case_id: str
    oct_path: Path
    he_path: Path
    output_dir: Path


@dataclass
class RegistrationResult:
    scale: float
    rotation_deg: float
    translation_y: float
    translation_x: float
    score: float
    details: dict[str, float]


def _prepare_rembg_env() -> None:
    NUMBA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    REMBG_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_ROOT)
    os.environ["U2NET_HOME"] = str(REMBG_CACHE_ROOT)


@lru_cache(maxsize=1)
def _get_rembg_session():
    _prepare_rembg_env()
    from rembg import new_session

    return new_session("u2net")


def _read_tiff(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(str(path)))


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        rgb = array[..., :3].astype(np.float32)
        alpha = array[..., 3:].astype(np.float32)
        if alpha.max() > 1.5:
            alpha = alpha / 255.0
        white = np.full_like(rgb, 255.0 if rgb.max() > 1.5 else 1.0)
        return np.clip(rgb * alpha + white * (1.0 - alpha), 0, white.max()).astype(array.dtype)
    return array[..., :3]


def _oct_to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    rgb = _ensure_rgb(array).astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _robust_rescale(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0] if np.any(finite > 0) else finite
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _remove_row_col_bias(image: np.ndarray) -> np.ndarray:
    residual = image - filters.gaussian(image, sigma=max(8, min(image.shape) / 90), preserve_range=True)
    row_bias = np.median(residual, axis=1)
    col_bias = np.median(residual, axis=0)
    row_bias = filters.gaussian(row_bias, sigma=max(3, image.shape[0] / 300), preserve_range=True)
    col_bias = filters.gaussian(col_bias, sigma=max(3, image.shape[1] / 300), preserve_range=True)
    corrected = image - row_bias[:, None] - col_bias[None, :]
    return _robust_rescale(corrected, 0.5, 99.7)


def preprocess_oct_2d(gray: np.ndarray) -> dict[str, np.ndarray]:
    raw = _robust_rescale(gray, 0.5, 99.7)
    sigma = max(24, min(raw.shape) / 18)
    shading = filters.gaussian(raw, sigma=sigma, preserve_range=True)
    shading = np.maximum(shading, np.percentile(shading, 5) + 1e-4)
    flat = raw / shading * float(np.median(shading))
    flat = _robust_rescale(flat, 0.5, 99.7)
    destriped = _remove_row_col_bias(flat)
    clahe_kernel = max(32, min(destriped.shape) // 12)
    clahe = exposure.equalize_adapthist(destriped, kernel_size=clahe_kernel, clip_limit=0.018)
    contrast = exposure.adjust_gamma(clahe, gamma=0.9).astype(np.float32)
    return {"raw_rescaled": raw, "flatfield": flat, "destriped": destriped, "contrast": contrast}


def preprocess_he_rgb(rgb: np.ndarray) -> dict[str, np.ndarray]:
    image = _ensure_rgb(rgb).astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    gray = np.dot(image[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray = _robust_rescale(gray, 1.0, 99.5)
    inverted = (1.0 - gray).astype(np.float32)
    rembg_input = np.repeat(inverted[..., None], 3, axis=2)
    return {"gray": gray.astype(np.float32), "inverted_gray": inverted, "rembg_input": rembg_input.astype(np.float32)}


def _rgb_float01(rgb: np.ndarray) -> np.ndarray:
    image = _ensure_rgb(rgb).astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _estimate_macenko_stain_matrix(rgb: np.ndarray, max_dim: int = 900, beta: float = 0.15, angular_percentile: float = 1.0) -> np.ndarray:
    image = _rgb_float01(rgb)
    if max(image.shape[:2]) > max_dim:
        scale = max_dim / float(max(image.shape[:2]))
        small_shape = (max(16, int(round(image.shape[0] * scale))), max(16, int(round(image.shape[1] * scale))))
        image = _resize_rgb(image, small_shape)
    flat = image.reshape(-1, 3)
    od = -np.log(np.clip(flat, 1.0 / 255.0, 1.0))
    od = od[np.any(od > beta, axis=1)]
    if od.shape[0] < 256:
        return MACENKO_TARGET_STAINS.copy()
    if od.shape[0] > 250000:
        rng = np.random.default_rng(7)
        od = od[rng.choice(od.shape[0], size=250000, replace=False)]
    _, _, vt = np.linalg.svd(od, full_matrices=False)
    plane = vt[:2].T
    projected = od @ plane
    phi = np.arctan2(projected[:, 1], projected[:, 0])
    min_phi, max_phi = np.percentile(phi, [angular_percentile, 100.0 - angular_percentile])
    v1 = plane @ np.array([np.cos(min_phi), np.sin(min_phi)], dtype=np.float32)
    v2 = plane @ np.array([np.cos(max_phi), np.sin(max_phi)], dtype=np.float32)
    stains = np.stack([v1, v2], axis=1).astype(np.float32)
    stains = np.abs(stains)
    stains /= np.linalg.norm(stains, axis=0, keepdims=True) + 1e-8
    if stains[0, 0] < stains[0, 1]:
        stains = stains[:, ::-1]
    return stains.astype(np.float32)


def stain_normalize_he_macenko(rgb: np.ndarray, mask: np.ndarray | None = None, chunk_rows: int = 512) -> np.ndarray:
    image = _rgb_float01(rgb)
    stains = _estimate_macenko_stain_matrix(image)
    pinv = np.linalg.pinv(stains).astype(np.float32)
    target = MACENKO_TARGET_STAINS.astype(np.float32)
    normalized = np.empty_like(image, dtype=np.float32)
    max_concentrations: list[np.ndarray] = []
    sample = image
    if max(image.shape[:2]) > 1000:
        scale = 1000 / float(max(image.shape[:2]))
        sample = _resize_rgb(image, (max(16, int(round(image.shape[0] * scale))), max(16, int(round(image.shape[1] * scale)))))
    sample_od = -np.log(np.clip(sample.reshape(-1, 3), 1.0 / 255.0, 1.0))
    sample_c = np.maximum(sample_od @ pinv.T, 0.0)
    valid = np.any(sample_od > 0.15, axis=1)
    if np.any(valid):
        max_conc = np.percentile(sample_c[valid], 99, axis=0).astype(np.float32)
    else:
        max_conc = np.percentile(sample_c, 99, axis=0).astype(np.float32)
    scale_conc = MACENKO_TARGET_MAX_CONC / np.maximum(max_conc, 1e-4)
    for y0 in range(0, image.shape[0], chunk_rows):
        y1 = min(image.shape[0], y0 + chunk_rows)
        block = image[y0:y1]
        od = -np.log(np.clip(block.reshape(-1, 3), 1.0 / 255.0, 1.0))
        c = np.maximum(od @ pinv.T, 0.0) * scale_conc[None, :]
        rgb_block = np.exp(-(c @ target.T)).reshape(block.shape)
        normalized[y0:y1] = np.clip(rgb_block, 0.0, 1.0)
    if mask is not None and mask.shape == image.shape[:2]:
        normalized = np.where(mask[..., None].astype(bool), normalized, 1.0)
    return np.clip(np.round(normalized * 255.0), 0, 255).astype(np.uint8)


def _downsample_rgb_for_stain_fit(rgb: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    image = _ensure_rgb(rgb).astype(np.float32)
    if image.max() <= 1.5:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if max(image.shape[:2]) <= max_dim:
        return image
    scale = max_dim / float(max(image.shape[:2]))
    shape = (max(16, int(round(image.shape[0] * scale))), max(16, int(round(image.shape[1] * scale))))
    return np.clip(_resize_rgb(image.astype(np.float32), shape), 0, 255).astype(np.uint8)


def _torchstain_normalize_he(
    rgb: np.ndarray,
    method: str,
    reference_rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    import torchstain

    image = _ensure_rgb(rgb).astype(np.float32)
    if image.max() <= 1.5:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if method == "macenko":
        normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        if reference_rgb is not None:
            normalizer.fit(_downsample_rgb_for_stain_fit(reference_rgb))
        normalized = normalizer.normalize(image, stains=False)
    elif method == "reinhard":
        if reference_rgb is None:
            reference_rgb = image
        normalizer = torchstain.normalizers.ReinhardNormalizer(backend="numpy")
        normalizer.fit(_downsample_rgb_for_stain_fit(reference_rgb))
        normalized = normalizer.normalize(image)
    else:
        raise ValueError(f"Unsupported torchstain method: {method}")
    if isinstance(normalized, tuple):
        normalized = normalized[0]
    normalized = np.asarray(normalized, dtype=np.uint8)
    if mask is not None and mask.shape == normalized.shape[:2]:
        normalized = np.where(mask[..., None].astype(bool), normalized, 255).astype(np.uint8)
    return normalized


def stain_standardize_he(
    rgb: np.ndarray,
    method: str,
    reference_rgb: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if method == "none":
        image = _ensure_rgb(rgb).astype(np.float32)
        if image.max() <= 1.5:
            image = image * 255.0
        output = np.clip(image, 0, 255).astype(np.uint8)
        if mask is not None and mask.shape == output.shape[:2]:
            output = np.where(mask[..., None].astype(bool), output, 255).astype(np.uint8)
        return output
    if method in {"torchstain_reinhard", "reinhard"}:
        return _torchstain_normalize_he(rgb, "reinhard", reference_rgb=reference_rgb, mask=mask)
    if method in {"torchstain_macenko", "macenko"}:
        return _torchstain_normalize_he(rgb, "macenko", reference_rgb=reference_rgb, mask=mask)
    if method == "internal_macenko":
        return stain_normalize_he_macenko(rgb, mask=mask)
    raise ValueError(f"Unsupported stain normalizer: {method}")


def _resize_gray(image: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    if tuple(image.shape[:2]) == tuple(shape):
        return image.astype(np.float32)
    return transform.resize(image, shape, order=order, preserve_range=True, anti_aliasing=order > 0).astype(np.float32)


def _resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(image.shape[:2]) == tuple(shape):
        return image.astype(np.float32)
    return transform.resize(image, (*shape, image.shape[2]), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return _resize_gray(mask.astype(np.float32), shape, order=0) > 0.5


def _search_shape(native_shape: tuple[int, int], max_dim: int) -> tuple[int, int]:
    scale = min(1.0, max_dim / float(max(native_shape)))
    return (max(32, int(round(native_shape[0] * scale))), max(32, int(round(native_shape[1] * scale))))


def _clean_mask(mask: np.ndarray, min_fraction: float = 0.01) -> np.ndarray:
    mask = ndimage.binary_fill_holes(mask.astype(bool))
    mask = morphology.opening(mask, morphology.disk(2))
    mask = morphology.closing(mask, morphology.disk(4))
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask.astype(bool)
    props = measure.regionprops(labels)
    min_area = max(32, int(mask.size * min_fraction))
    keep = [p.label for p in props if p.area >= min_area]
    if not keep:
        keep = [max(props, key=lambda p: p.area).label]
    return np.isin(labels, keep)


def _remove_border_components(mask: np.ndarray, border_fraction: float = 0.03, min_fraction: float = 0.004) -> np.ndarray:
    mask = mask.astype(bool)
    border = max(2, int(round(min(mask.shape) * border_fraction)))
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    border_labels = set(np.unique(labels[:border, :]))
    border_labels.update(np.unique(labels[-border:, :]))
    border_labels.update(np.unique(labels[:, :border]))
    border_labels.update(np.unique(labels[:, -border:]))
    border_labels.discard(0)
    interior = mask.copy()
    for label in border_labels:
        interior[labels == label] = False
    interior = _clean_mask(interior, min_fraction=min_fraction)
    if int(interior.sum()) >= max(128, int(mask.size * min_fraction)):
        return interior
    return mask


def _rembg_mask(input_rgb: np.ndarray, alpha_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    _prepare_rembg_env()
    from rembg import remove

    arr = np.asarray(input_rgb, dtype=np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    rgba = remove(arr, session=_get_rembg_session())
    alpha = np.asarray(rgba[..., 3] if rgba.ndim == 3 and rgba.shape[-1] >= 4 else rgba, dtype=np.float32) / 255.0
    threshold = float(alpha_threshold)
    if threshold > 1.0:
        threshold = threshold / 255.0
    mask = _clean_mask(alpha > threshold)
    return mask.astype(bool), alpha.astype(np.float32)


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool) ^ morphology.erosion(mask.astype(bool), morphology.disk(2))


def _distance(mask: np.ndarray) -> np.ndarray:
    dist = ndimage.distance_transform_edt(mask.astype(bool))
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist.astype(np.float32)


def _boundary_distance(boundary: np.ndarray) -> np.ndarray:
    dist = ndimage.distance_transform_edt(~boundary.astype(bool))
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist.astype(np.float32)


def _trimmed_boundary_score(
    fixed_boundary_distance: np.ndarray,
    moved_boundary_distance: np.ndarray,
    fixed_boundary: np.ndarray,
    moved_boundary: np.ndarray,
    trim_percentile: float = 70.0,
) -> tuple[float, float, float, float]:
    fixed_boundary = fixed_boundary.astype(bool)
    moved_boundary = moved_boundary.astype(bool)
    if not np.any(fixed_boundary) or not np.any(moved_boundary):
        return 0.0, 1.0, 1.0, 1.0
    moved_to_fixed = fixed_boundary_distance[moved_boundary]
    fixed_to_moved = moved_boundary_distance[fixed_boundary]
    m2f = float(np.percentile(moved_to_fixed, trim_percentile)) if moved_to_fixed.size else 1.0
    f2m = float(np.percentile(fixed_to_moved, trim_percentile)) if fixed_to_moved.size else 1.0
    distance = 0.5 * (m2f + f2m)
    score = float(np.exp(-4.0 * distance))
    return score, distance, m2f, f2m


def _center_distance_score(fixed_mask: np.ndarray, moved_mask: np.ndarray) -> tuple[float, float]:
    fixed_center = _mask_center(fixed_mask)
    moved_center = _mask_center(moved_mask)
    diag = max(1.0, float(np.hypot(*fixed_mask.shape)))
    distance = float(np.linalg.norm(fixed_center - moved_center) / diag)
    return float(np.exp(-8.0 * distance)), distance


def _oct_maps(oct_contrast: np.ndarray, alpha_threshold: float) -> dict[str, np.ndarray]:
    rgb = np.repeat(oct_contrast[..., None], 3, axis=2)
    mask, alpha = _rembg_mask(rgb, alpha_threshold)
    if int(mask.sum()) < 128:
        mask = _clean_mask(oct_contrast > max(0.15, np.percentile(oct_contrast[oct_contrast > 0], 45)))
    gradient = filters.gaussian(filters.sobel(oct_contrast), sigma=1.0) * mask.astype(np.float32)
    feature = exposure.rescale_intensity(0.70 * oct_contrast + 0.30 * gradient, out_range=(0.0, 1.0)) * mask
    return {"mask": mask, "alpha": alpha, "boundary": _boundary(mask), "distance": _distance(mask), "feature": feature.astype(np.float32)}


def _gray_he_mask(he_gray: np.ndarray, percentile: float) -> np.ndarray:
    inverted = 1.0 - he_gray
    threshold = max(0.04, float(np.percentile(inverted, percentile)))
    return _clean_mask(inverted > threshold, min_fraction=0.002)


def _he_maps(
    he_rgb: np.ndarray,
    he_gray: np.ndarray,
    he_rembg_input: np.ndarray,
    alpha_threshold: float,
    mask_mode: str = "auto",
    gray_mask_percentile: float = 67.0,
) -> dict[str, np.ndarray]:
    mask, alpha = _rembg_mask(he_rembg_input, alpha_threshold)
    rgb = np.clip(he_rgb, 0, 1)
    hsv = color.rgb2hsv(rgb)
    od = -np.log(np.clip(rgb, 1e-3, 1.0))
    od_sum = od.sum(axis=2)
    od_thr = max(0.10, float(np.percentile(od_sum, 65)))
    sat_thr = max(0.035, float(np.percentile(hsv[..., 1], 60)))
    fallback = ((hsv[..., 1] > sat_thr) & (od_sum > od_thr)) | ((he_gray < np.percentile(he_gray, 88)) & (od_sum > od_thr * 0.65))
    fallback = _clean_mask(fallback, min_fraction=0.004)
    border = max(2, min(mask.shape) // 32)
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    mask_fraction = float(mask.mean())
    border_fraction = float(mask[border_mask].mean()) if np.any(border_mask) else 0.0
    fallback_fraction = float(fallback.mean())
    gray_mask = _gray_he_mask(he_gray, gray_mask_percentile)
    if mask_mode == "gray":
        mask = gray_mask
    elif mask_mode == "rembg":
        pass
    elif int(mask.sum()) < 128 or mask_fraction > 0.65 or border_fraction > 0.55:
        if int(fallback.sum()) >= 128 and fallback_fraction < max(0.62, mask_fraction * 0.90):
            mask = fallback
        elif int(gray_mask.sum()) >= 128 and float(gray_mask.mean()) < max(0.62, mask_fraction * 0.90):
            mask = gray_mask
        elif mask_fraction > 0.65 or border_fraction > 0.55:
            mask = _remove_border_components(mask, border_fraction=0.04, min_fraction=0.003)
    try:
        hed = color.rgb2hed(np.clip(he_rgb, 0, 1))
        hema = _robust_rescale(hed[..., 0], 1.0, 99.0)
    except Exception:
        hema = 1.0 - he_gray
    gray_feature = 1.0 - he_gray
    hema = _robust_rescale(0.55 * hema + 0.45 * gray_feature, 1.0, 99.0)
    gradient = filters.gaussian(filters.sobel(hema), sigma=1.0) * mask.astype(np.float32)
    feature = exposure.rescale_intensity(0.65 * hema + 0.35 * gradient, out_range=(0.0, 1.0)) * mask
    return {"mask": mask, "alpha": alpha, "boundary": _boundary(mask), "distance": _distance(mask), "feature": feature.astype(np.float32)}


def _mask_center(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.array([mask.shape[0] / 2.0, mask.shape[1] / 2.0], dtype=np.float32)
    return coords.mean(axis=0).astype(np.float32)


def _orientation(mask: np.ndarray) -> float:
    labels = measure.label(mask)
    if labels.max() == 0:
        return 0.0
    prop = max(measure.regionprops(labels), key=lambda p: p.area)
    return -math.degrees(float(prop.orientation))


def _affine(scale: float, rotation_deg: float, ty: float, tx: float, moving_center: np.ndarray, fixed_center: np.ndarray) -> np.ndarray:
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta) * scale, math.sin(theta) * scale
    linear = np.array([[c, -s], [s, c]], dtype=np.float64)
    offset = fixed_center + np.array([ty, tx], dtype=np.float64) - linear @ moving_center
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = offset
    return matrix


def _warp(image: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int], order: int) -> np.ndarray:
    inv = np.linalg.inv(matrix)
    return transform.warp(image, inverse_map=inv, output_shape=output_shape, order=order, preserve_range=True, mode="constant", cval=0).astype(np.float32)


def _warp_maps(maps: dict[str, np.ndarray], matrix: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    moved = {
        "mask": _warp(maps["mask"].astype(np.float32), matrix, shape, order=0) > 0.5,
        "boundary": _warp(maps["boundary"].astype(np.float32), matrix, shape, order=0) > 0.5,
        "distance": _warp(maps["distance"], matrix, shape, order=1),
        "feature": _warp(maps["feature"], matrix, shape, order=1),
    }
    moved["boundary_distance"] = _boundary_distance(moved["boundary"])
    return moved


def _score(fixed: dict[str, np.ndarray], moved: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    fixed_mask = fixed["mask"].astype(bool)
    moved_mask = moved["mask"].astype(bool)
    fixed_fraction = float(fixed_mask.mean())
    moved_fraction = float(moved_mask.mean())
    if fixed_fraction < 0.002 or moved_fraction < 0.002 or fixed_fraction > 0.96 or moved_fraction > 0.96:
        return -1.0, {
            "rejected": 1.0,
            "reject_reason": 1.0,
            "oct_mask_fraction": fixed_fraction,
            "he_mask_fraction": moved_fraction,
        }
    intersection = fixed_mask & moved_mask
    union = fixed_mask | moved_mask
    dice = 2.0 * intersection.sum() / max(1.0, fixed_mask.sum() + moved_mask.sum())
    iou = intersection.sum() / max(1.0, union.sum())
    smaller_fraction = min(fixed_fraction, moved_fraction)
    larger_fraction = max(fixed_fraction, moved_fraction)
    area_ratio = smaller_fraction / max(larger_fraction, 1e-6)
    overlap_smaller = float(intersection.sum() / max(1.0, min(fixed_mask.sum(), moved_mask.sum())))
    overlap_larger = float(intersection.sum() / max(1.0, max(fixed_mask.sum(), moved_mask.sum())))
    if area_ratio < 0.16 and overlap_smaller < 0.82:
        return -0.75, {
            "rejected": 1.0,
            "reject_reason": 2.0,
            "area_ratio": float(area_ratio),
            "overlap_of_smaller": overlap_smaller,
            "overlap_of_larger": overlap_larger,
        }
    fixed_boundary = fixed["boundary"].astype(bool)
    moved_boundary = moved["boundary"].astype(bool)
    fixed_boundary_distance = fixed.get("boundary_distance")
    if fixed_boundary_distance is None:
        fixed_boundary_distance = _boundary_distance(fixed_boundary)
    moved_boundary_distance = moved.get("boundary_distance")
    if moved_boundary_distance is None:
        moved_boundary_distance = _boundary_distance(moved_boundary)
    boundary_score, boundary_distance, he_to_oct, oct_to_he = _trimmed_boundary_score(
        fixed_boundary_distance,
        moved_boundary_distance,
        fixed_boundary,
        moved_boundary,
    )
    center_score, center_distance = _center_distance_score(fixed_mask, moved_mask)
    feature_score = 0.0
    if intersection.sum() > 64:
        a = fixed["feature"][intersection].astype(np.float32)
        b = moved["feature"][intersection].astype(np.float32)
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom > 1e-6:
            feature_score = float((np.sum(a * b) / denom + 1.0) / 2.0)
    containment_penalty = max(0.0, overlap_smaller - overlap_larger) * max(0.0, 0.30 - area_ratio)
    score = (
        0.30 * dice
        + 0.16 * iou
        + 0.34 * boundary_score
        + 0.10 * center_score
        + 0.10 * feature_score
        - 0.30 * containment_penalty
    )
    return float(score), {
        "dice": float(dice),
        "iou": float(iou),
        "boundary": float(boundary_score),
        "boundary_distance": float(boundary_distance),
        "he_boundary_to_oct": float(he_to_oct),
        "oct_boundary_to_he": float(oct_to_he),
        "center": float(center_score),
        "center_distance": float(center_distance),
        "feature": float(feature_score),
        "area_ratio": float(area_ratio),
        "overlap_of_smaller": float(overlap_smaller),
        "overlap_of_larger": float(overlap_larger),
        "containment_penalty": float(containment_penalty),
    }


def _native_matrix(search_matrix: np.ndarray, he_shape: tuple[int, int], oct_shape: tuple[int, int], search_shape: tuple[int, int]) -> np.ndarray:
    he_to_search = np.diag([search_shape[0] / he_shape[0], search_shape[1] / he_shape[1], 1.0])
    search_to_oct = np.diag([oct_shape[0] / search_shape[0], oct_shape[1] / search_shape[1], 1.0])
    return search_to_oct @ search_matrix @ he_to_search


def register_pair(
    pair: PairPaths,
    he_alpha_threshold: float,
    oct_alpha_threshold: float,
    max_search_dim: int,
    he_mask_mode: str = "auto",
    he_gray_mask_percentile: float = 67.0,
    stain_normalizer: str = "torchstain_reinhard",
    stain_reference_rgb: np.ndarray | None = None,
) -> dict[str, Any]:
    oct_raw = _oct_to_gray(_read_tiff(pair.oct_path))
    he_raw = _ensure_rgb(_read_tiff(pair.he_path))
    oct_native_shape = tuple(int(v) for v in oct_raw.shape[:2])
    search_shape = _search_shape(oct_native_shape, max_search_dim)

    oct_pre_native = preprocess_oct_2d(oct_raw)
    he_pre_native = preprocess_he_rgb(he_raw)
    oct_search = _resize_gray(oct_pre_native["contrast"], search_shape)
    he_search_rgb = _resize_rgb(_ensure_rgb(he_raw).astype(np.float32) / (255.0 if np.asarray(he_raw).max() > 1.5 else 1.0), search_shape)
    he_search_gray = _resize_gray(he_pre_native["gray"], search_shape)
    he_search_rembg = _resize_rgb(he_pre_native["rembg_input"], search_shape)

    oct_search_maps = _oct_maps(oct_search, oct_alpha_threshold)
    he_search_maps = _he_maps(
        he_search_rgb,
        he_search_gray,
        he_search_rembg,
        he_alpha_threshold,
        mask_mode=he_mask_mode,
        gray_mask_percentile=he_gray_mask_percentile,
    )
    oct_center = _mask_center(oct_search_maps["mask"])
    he_center = _mask_center(he_search_maps["mask"])
    area_scale = math.sqrt(max(1.0, oct_search_maps["mask"].sum()) / max(1.0, he_search_maps["mask"].sum()))
    seeds: list[np.ndarray] = []
    init_rotation = float(np.clip(_orientation(oct_search_maps["mask"]) - _orientation(he_search_maps["mask"]), -60.0, 60.0))
    for scale_mult in [0.78, 0.92, 1.0, 1.10, 1.24]:
        for rot_delta in [-25.0, -10.0, 0.0, 10.0, 25.0]:
            for ty, tx in [(0.0, 0.0), (-10.0, 0.0), (10.0, 0.0), (0.0, -10.0), (0.0, 10.0)]:
                seeds.append(np.array([area_scale * scale_mult, init_rotation + rot_delta, ty, tx], dtype=np.float64))

    ranked: list[tuple[float, np.ndarray]] = []
    for seed in seeds:
        matrix = _affine(seed[0], seed[1], seed[2], seed[3], he_center, oct_center)
        moved = _warp_maps(he_search_maps, matrix, search_shape)
        score, _ = _score(oct_search_maps, moved)
        ranked.append((score, seed))
    ranked.sort(key=lambda item: item[0], reverse=True)

    best: RegistrationResult | None = None

    def objective(params: np.ndarray) -> float:
        scale, rotation, ty, tx = [float(v) for v in params]
        if not (0.25 <= scale <= 2.5) or abs(rotation) > 180:
            return 1.0
        if abs(ty) > search_shape[0] * 0.40 or abs(tx) > search_shape[1] * 0.40:
            return 1.0
        matrix = _affine(scale, rotation, ty, tx, he_center, oct_center)
        moved = _warp_maps(he_search_maps, matrix, search_shape)
        score, _ = _score(oct_search_maps, moved)
        regularization = 0.0005 * (abs(ty) + abs(tx)) + 0.001 * abs(math.log(max(scale, 1e-3) / max(area_scale, 1e-3)))
        return -(score - regularization)

    for _, seed in ranked[:6]:
        opt = optimize.minimize(objective, seed, method="Powell", options={"maxiter": 36, "disp": False})
        params = opt.x if opt.success else seed
        matrix = _affine(params[0], params[1], params[2], params[3], he_center, oct_center)
        moved = _warp_maps(he_search_maps, matrix, search_shape)
        score, details = _score(oct_search_maps, moved)
        candidate = RegistrationResult(float(params[0]), float(params[1]), float(params[2]), float(params[3]), float(score), details)
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        raise RuntimeError("No registration candidate produced a valid score")

    search_matrix = _affine(best.scale, best.rotation_deg, best.translation_y, best.translation_x, he_center, oct_center)
    native_matrix = _native_matrix(search_matrix, he_raw.shape[:2], oct_native_shape, search_shape)
    he_native_mask = _resize_mask(he_search_maps["mask"], he_raw.shape[:2])
    stain_corrected_he = stain_standardize_he(
        he_raw,
        method=stain_normalizer,
        reference_rgb=stain_reference_rgb,
        mask=he_native_mask,
    )
    warped_he = _warp(stain_corrected_he.astype(np.float32), native_matrix, oct_native_shape, order=1)
    warped_mask = _warp(_resize_mask(he_search_maps["mask"], he_raw.shape[:2]).astype(np.float32), native_matrix, oct_native_shape, order=0) > 0.5
    oct_mask_native = _resize_mask(oct_search_maps["mask"], oct_native_shape)
    overlap = warped_mask & oct_mask_native
    oct_registered = np.asarray(oct_pre_native["contrast"], dtype=np.float32)

    pair.output_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb_tiff(pair.output_dir / "he_registered.tiff", warped_he)
    _save_float(pair.output_dir / "oct_registered.tiff", oct_registered)
    _save_mask_tiff(pair.output_dir / "registered_mask.tiff", overlap)
    _sync_clean_outputs(pair.output_dir)
    _save_rgb(pair.output_dir / "he_registered_preview.png", warped_he)
    _save_gray(pair.output_dir / "oct_registered_preview.png", oct_registered)
    _save_gray(pair.output_dir / "registered_mask_preview.png", overlap.astype(np.float32))
    _save_rgb(pair.output_dir / "he_stain_corrected_native_preview.png", stain_corrected_he)
    _save_gray(pair.output_dir / "oct_raw_display.png", oct_pre_native["raw_rescaled"])
    _save_gray(pair.output_dir / "oct_flatfield_corrected.png", oct_pre_native["flatfield"])
    _save_gray(pair.output_dir / "oct_tile_artifact_suppressed.png", oct_pre_native["destriped"])
    _save_gray(pair.output_dir / "oct_contrast_adjusted.png", oct_pre_native["contrast"])
    _save_gray(pair.output_dir / "he_black_white_input.png", he_pre_native["inverted_gray"])
    _save_gray(pair.output_dir / "oct_rembg_mask_search.png", oct_search_maps["mask"].astype(np.float32))
    _save_gray(pair.output_dir / "he_rembg_mask_search.png", he_search_maps["mask"].astype(np.float32))
    _save_float(pair.output_dir / "oct_rembg_alpha_search_float32.tiff", oct_search_maps["alpha"])
    _save_float(pair.output_dir / "he_rembg_alpha_search_float32.tiff", he_search_maps["alpha"])
    _save_rgb(pair.output_dir / "he_warped_to_oct_xy.png", warped_he)
    _save_gray(pair.output_dir / "oct_volume_mask_xy.png", oct_mask_native.astype(np.float32))
    _save_gray(pair.output_dir / "he_warped_mask.png", warped_mask.astype(np.float32))
    _save_gray(pair.output_dir / "overlap_mask.png", overlap.astype(np.float32))
    _save_rgb(pair.output_dir / "he_warped_overlap_only.png", warped_he * overlap[..., None])
    _save_rgb(pair.output_dir / "overlay_preview.png", _overlay_preview(warped_he, oct_pre_native["contrast"], overlap))
    _save_rgb(pair.output_dir / "overlay_false_color.png", _false_color(warped_he, oct_pre_native["contrast"], oct_mask_native))
    _save_rgb(pair.output_dir / "overlay_contours.png", _contours(warped_he, warped_mask, oct_mask_native))
    _save_rgb(pair.output_dir / "overlay_checkerboard.png", _checkerboard(warped_he, _gray_to_rgb(oct_pre_native["contrast"]), tile=48))

    summary = {
        "case_id": pair.case_id,
        "oct_path": str(pair.oct_path),
        "he_path": str(pair.he_path),
        "pipeline": "align_he_to_oct_2D_v5",
        "preprocessing": {
            "oct": "robust percentile rescale + gaussian flat-field division + row/column median bias suppression + CLAHE + gamma",
            "he_registration_mask": "black-and-white/rembg-compatible downsampled mask derivation; optional grayscale fallback for difficult backgrounds",
            "he_final_image": "library-backed H&E stain/color standardization applied at native resolution before warping; registration transform is applied to the standardized HE image",
        },
        "stain_normalizer": stain_normalizer,
        "he_alpha_threshold": float(he_alpha_threshold),
        "oct_alpha_threshold": float(oct_alpha_threshold),
        "he_mask_mode": he_mask_mode,
        "he_gray_mask_percentile": float(he_gray_mask_percentile),
        "search_shape": list(search_shape),
        "oct_output_shape": list(oct_native_shape),
        "he_transform_into_oct_xy": asdict(best),
        "output_images": {name: str(pair.output_dir / name) for name in [
            "he_registered.tiff",
            "oct_registered.tiff",
            "registered_mask.tiff",
            "he_registered_preview.png",
            "oct_registered_preview.png",
            "registered_mask_preview.png",
            "oct_raw_display.png",
            "oct_flatfield_corrected.png",
            "oct_tile_artifact_suppressed.png",
            "oct_contrast_adjusted.png",
            "he_black_white_input.png",
            "overlay_false_color.png",
            "overlay_preview.png",
            "overlay_contours.png",
            "overlay_checkerboard.png",
            "overlap_mask.png",
            "output/he_registered.tiff",
            "output/oct_registered.tiff",
            "output/registered_mask.tiff",
        ]},
    }
    (pair.output_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(_to_uint8(image)).save(path)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    arr = _to_uint8(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    Image.fromarray(arr[..., :3]).save(path)


def _save_float(path: Path, image: np.ndarray) -> None:
    tifffile.imwrite(str(path), np.asarray(image, dtype=np.float32))


def _save_rgb_tiff(path: Path, image: np.ndarray) -> None:
    arr = _to_uint8(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    tifffile.imwrite(str(path), arr[..., :3], photometric="rgb")


def _save_mask_tiff(path: Path, mask: np.ndarray) -> None:
    tifffile.imwrite(str(path), (np.asarray(mask).astype(bool).astype(np.uint8) * 255))


def _sync_clean_outputs(output_dir: Path) -> None:
    clean_dir = output_dir / "output"
    clean_dir.mkdir(parents=True, exist_ok=True)
    for name in ["he_registered.tiff", "oct_registered.tiff", "registered_mask.tiff"]:
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, clean_dir / name)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(_to_uint8(gray)[..., None], 3, axis=2).astype(np.float32)


def _false_color(he_rgb: np.ndarray, oct_gray: np.ndarray, oct_mask: np.ndarray) -> np.ndarray:
    he = _to_uint8(he_rgb).astype(np.float32) / 255.0
    oct_img = _robust_rescale(oct_gray)
    out = he * 0.65
    out[..., 1] = np.maximum(out[..., 1], oct_img * oct_mask)
    out[..., 2] = np.maximum(out[..., 2], oct_img * oct_mask * 0.45)
    return out


def _overlay_preview(he_rgb: np.ndarray, oct_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    he = _to_uint8(he_rgb).astype(np.float32) / 255.0
    oct_img = _to_uint8(oct_gray).astype(np.float32) / 255.0
    out = he * 0.82
    out[..., 1] = np.maximum(out[..., 1], oct_img * 0.95)
    out[..., 2] = np.maximum(out[..., 2], oct_img * 0.35)
    out[~mask.astype(bool)] *= 0.82
    return np.clip(out, 0, 1)


def _contours(he_rgb: np.ndarray, he_mask: np.ndarray, oct_mask: np.ndarray) -> np.ndarray:
    out = _to_uint8(he_rgb).astype(np.float32) / 255.0
    he_b = _boundary(he_mask)
    oct_b = _boundary(oct_mask)
    out[oct_b] = np.array([0.0, 1.0, 0.0])
    out[he_b] = np.array([1.0, 0.0, 0.0])
    return out


def _checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 48) -> np.ndarray:
    a8 = _to_uint8(a).astype(np.float32) / 255.0
    b8 = _to_uint8(b).astype(np.float32) / 255.0
    yy, xx = np.indices(a8.shape[:2])
    mask = ((yy // tile) + (xx // tile)) % 2 == 0
    out = a8.copy()
    out[mask] = b8[mask]
    return out


def _discover_pairs(input_root: Path, output_root: Path) -> list[PairPaths]:
    pairs: list[PairPaths] = []
    for oct_path in sorted(input_root.glob("*_oct.tif*")):
        case_id = oct_path.name.rsplit("_oct", 1)[0]
        he_candidates = sorted(input_root.glob(f"{case_id}_section.tif*"))
        if not he_candidates:
            continue
        pairs.append(PairPaths(case_id=case_id, oct_path=oct_path, he_path=he_candidates[0], output_dir=output_root / case_id))
    return pairs


def _append_status(status_path: Path, record: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    last_error: OSError | None = None
    for _ in range(3):
        try:
            with status_path.open("a") as status:
                status.write(line)
                status.flush()
            return
        except OSError as exc:
            last_error = exc
    if last_error is not None:
        print(f"Warning: could not write status for {record.get('case')}: {last_error}")


# ---------------------------------------------------------------------------
# v3 registration compatibility layer
# ---------------------------------------------------------------------------
# The interactive app imports these helpers directly.  Keep preprocessing and
# final image generation in v5, but route mask construction, warping, affine
# composition, and scoring through the recovered v3 implementation so the
# registration objective is not changed by the newer preprocessing displays.


def _alpha_for_v3(alpha_threshold: float) -> float:
    threshold = float(alpha_threshold)
    return threshold * 255.0 if threshold <= 1.0 else threshold


def _as_v3_maps(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(maps)
    if "canonical_feature" not in out and "feature" in out:
        out["canonical_feature"] = out["feature"]
    if "feature" not in out and "canonical_feature" in out:
        out["feature"] = out["canonical_feature"]
    if "boundary" not in out and "mask" in out:
        out["boundary"] = _V3._mask_boundary(out["mask"].astype(bool))
    if "boundary_distance" not in out and "boundary" in out:
        out["boundary_distance"] = _V3._distance_to_boundary(out["boundary"].astype(bool))
    if "distance" not in out and "mask" in out:
        out["distance"] = _V3._distance_transform_map(out["mask"].astype(bool))
    if "alpha" not in out and "alpha_mean" in out:
        out["alpha"] = out["alpha_mean"]
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    return _V3._mask_boundary(mask.astype(bool))


def _distance(mask: np.ndarray) -> np.ndarray:
    return _V3._distance_transform_map(mask.astype(bool))


def _oct_maps(oct_contrast: np.ndarray, alpha_threshold: float) -> dict[str, np.ndarray]:
    plane = np.asarray(oct_contrast, dtype=np.float32)
    maps = _V3._build_oct_xy_maps_v2(
        plane[None, ...],
        tuple(int(v) for v in plane.shape[:2]),
        [0],
        alpha_threshold=_alpha_for_v3(alpha_threshold),
    )
    return _as_v3_maps(maps)


def _he_maps(
    he_rgb: np.ndarray,
    he_gray: np.ndarray,
    he_rembg_input: np.ndarray,
    alpha_threshold: float,
    mask_mode: str = "auto",
    gray_mask_percentile: float = 67.0,
) -> dict[str, np.ndarray]:
    maps, _enhanced = _V3._build_he_maps_v2(
        np.asarray(he_rgb, dtype=np.float32),
        alpha_threshold=_alpha_for_v3(alpha_threshold),
    )
    if mask_mode == "gray":
        mask = _gray_he_mask(he_gray, gray_mask_percentile)
        maps = dict(maps)
        maps["mask"] = mask.astype(bool)
        maps["boundary"] = _V3._mask_boundary(mask.astype(bool))
        maps["boundary_distance"] = _V3._distance_to_boundary(maps["boundary"].astype(bool))
        maps["distance"] = _V3._distance_transform_map(mask.astype(bool))
    return _as_v3_maps(maps)


def _affine(scale: float, rotation_deg: float, ty: float, tx: float, moving_center: np.ndarray, fixed_center: np.ndarray) -> np.ndarray:
    return _V3._build_affine_matrix(scale, rotation_deg, 0.0, 0.0, ty, tx, moving_center, fixed_center)


def _affine_with_tilt(
    scale: float,
    rotation_deg: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    ty: float,
    tx: float,
    moving_center: np.ndarray,
    fixed_center: np.ndarray,
) -> np.ndarray:
    return _V3._build_affine_matrix(scale, rotation_deg, tilt_x_deg, tilt_y_deg, ty, tx, moving_center, fixed_center)


def _warp(image: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int], order: int) -> np.ndarray:
    return _V3._warp_with_matrix(image, matrix, output_shape, order=order)


def _warp_maps(maps: dict[str, np.ndarray], matrix: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return _as_v3_maps(_V3._warp_he_maps_2d_v3(_as_v3_maps(maps), matrix, shape))


def _score(fixed: dict[str, np.ndarray], moved: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    return _V3._score_alignment_2d_v3(_as_v3_maps(fixed), _as_v3_maps(moved))


def _native_matrix(search_matrix: np.ndarray, he_shape: tuple[int, int], oct_shape: tuple[int, int], search_shape: tuple[int, int]) -> np.ndarray:
    return _V3._compose_native_matrix(search_matrix, he_shape, oct_shape, search_shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-contained 2D HE-to-OCT registration with v5 preprocessing and v3-style registration.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--oct-path", type=Path, default=None)
    parser.add_argument("--he-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--oct-alpha-threshold", type=float, default=0.376)
    parser.add_argument("--he-alpha-threshold", type=float, default=0.376)
    parser.add_argument("--he-mask-mode", choices=["auto", "rembg", "gray"], default="auto")
    parser.add_argument("--he-gray-mask-percentile", type=float, default=67.0)
    parser.add_argument(
        "--stain-normalizer",
        choices=["torchstain_reinhard", "torchstain_macenko", "reinhard", "macenko", "internal_macenko", "none"],
        default="torchstain_reinhard",
    )
    parser.add_argument("--stain-reference-he", type=Path, default=None)
    parser.add_argument("--max-search-dim", type=int, default=360)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.oct_path is not None or args.he_path is not None:
        if args.oct_path is None or args.he_path is None or args.output_dir is None:
            raise ValueError("--oct-path, --he-path, and --output-dir must be provided together for single-pair mode")
        pairs = [PairPaths(args.case_id or args.oct_path.stem.rsplit("_oct", 1)[0], args.oct_path, args.he_path, args.output_dir)]
        status_path = args.output_dir / "status_manifest.jsonl"
    else:
        pairs = _discover_pairs(args.input_root, args.output_root)
        status_path = args.output_root / "status_manifest.jsonl"

    reference_path = args.stain_reference_he
    if reference_path is None and pairs and args.stain_normalizer not in {"none", "internal_macenko"}:
        reference_path = pairs[0].he_path
    stain_reference_rgb = _ensure_rgb(_read_tiff(reference_path)) if reference_path is not None else None

    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and status_path.exists():
        status_path.unlink()
    for pair in pairs:
        summary_path = pair.output_dir / "alignment_summary.json"
        if summary_path.exists() and not args.overwrite:
            _append_status(status_path, {"case": pair.case_id, "status": "skipped_existing", "output_dir": str(pair.output_dir)})
            continue
        _append_status(status_path, {"case": pair.case_id, "status": "started", "oct_path": str(pair.oct_path), "he_path": str(pair.he_path), "output_dir": str(pair.output_dir)})
        try:
            summary = register_pair(
                pair,
                args.he_alpha_threshold,
                args.oct_alpha_threshold,
                args.max_search_dim,
                he_mask_mode=args.he_mask_mode,
                he_gray_mask_percentile=args.he_gray_mask_percentile,
                stain_normalizer=args.stain_normalizer,
                stain_reference_rgb=stain_reference_rgb,
            )
            _append_status(status_path, {"case": pair.case_id, "status": "completed", "output_dir": str(pair.output_dir), "score": summary["he_transform_into_oct_xy"]["score"]})
        except Exception as exc:
            _append_status(status_path, {"case": pair.case_id, "status": "failed", "output_dir": str(pair.output_dir), "error": str(exc)})


if __name__ == "__main__":
    main()
