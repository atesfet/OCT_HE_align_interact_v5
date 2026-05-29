from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import color, exposure, filters, morphology

from oct_he_pipeline import _clean_binary_mask


@dataclass(frozen=True)
class FeaturePreset:
    name: str
    description: str
    he_weights: dict[str, float]
    oct_weights: dict[str, float]


FEATURE_PRESETS: dict[str, FeaturePreset] = {
    "morphology_hybrid_v1": FeaturePreset(
        name="morphology_hybrid_v1",
        description="Boundary-aware hybrid preset using stain/contrast intensity, gradients, and signed tissue distance.",
        he_weights={"hema": 0.35, "grad": 0.30, "distance": 0.20, "laplace": 0.15},
        oct_weights={"clahe": 0.25, "grad": 0.35, "distance": 0.25, "laplace": 0.15},
    ),
    "gradient_distance_v1": FeaturePreset(
        name="gradient_distance_v1",
        description="Structure-heavy preset emphasizing gradients and silhouette geometry over raw intensity.",
        he_weights={"hema": 0.15, "grad": 0.40, "distance": 0.30, "laplace": 0.15},
        oct_weights={"clahe": 0.10, "grad": 0.45, "distance": 0.30, "laplace": 0.15},
    ),
    "contrast_distance_v1": FeaturePreset(
        name="contrast_distance_v1",
        description="Contrast plus signed-distance preset keeping some modality-specific appearance information.",
        he_weights={"hema": 0.45, "grad": 0.20, "distance": 0.20, "laplace": 0.15},
        oct_weights={"clahe": 0.40, "grad": 0.20, "distance": 0.25, "laplace": 0.15},
    ),
    "laplace_distance_v1": FeaturePreset(
        name="laplace_distance_v1",
        description="Curvature and silhouette focused preset for coarse structural correspondence.",
        he_weights={"hema": 0.20, "grad": 0.20, "distance": 0.25, "laplace": 0.35},
        oct_weights={"clahe": 0.15, "grad": 0.20, "distance": 0.25, "laplace": 0.40},
    ),
}


def _rescale01(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image.astype(np.float32)
    lo = float(np.percentile(image, 1))
    hi = float(np.percentile(image, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _distance_transform_map(mask: np.ndarray) -> np.ndarray:
    inside = ndimage.distance_transform_edt(mask.astype(bool))
    outside = ndimage.distance_transform_edt(~mask.astype(bool))
    signed = inside - outside
    signed = np.clip(signed, -32.0, 32.0)
    signed -= signed.min()
    denom = float(signed.max())
    if denom <= 1e-6:
        return np.zeros_like(signed, dtype=np.float32)
    return (signed / denom).astype(np.float32)


def _masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if int(valid.sum()) < 64:
        return 0.0
    av = np.asarray(a[valid], dtype=np.float32)
    bv = np.asarray(b[valid], dtype=np.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-6:
        return 0.0
    return float(max(0.0, np.dot(av, bv) / denom))


def _gradient_map(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grad = filters.gaussian(filters.sobel(np.asarray(image, dtype=np.float32)), sigma=1.0)
    return (grad * mask.astype(np.float32)).astype(np.float32)


def _enhance_oct_contrast(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    masked = image[mask.astype(bool)]
    if masked.size < 32:
        return image.astype(np.float32)
    p_low, p_high = np.percentile(masked, [1.0, 99.5])
    if p_high <= p_low:
        p_high = p_low + 1e-6
    stretched = np.clip((image - p_low) / (p_high - p_low), 0.0, 1.0)
    adaptive = exposure.equalize_adapthist(stretched, kernel_size=max(16, image.shape[0] // 12), clip_limit=0.02)
    gamma = exposure.adjust_gamma(adaptive, gamma=0.85)
    return (gamma * mask.astype(np.float32)).astype(np.float32)


def _border_connected_background(candidate_background: np.ndarray) -> np.ndarray:
    seeds = np.zeros_like(candidate_background, dtype=bool)
    seeds[[0, -1], :] = True
    seeds[:, [0, -1]] = True
    return ndimage.binary_propagation(seeds & candidate_background, mask=candidate_background)


def _refine_he_mask(rgb: np.ndarray, hematoxylin: np.ndarray, tissue_score: np.ndarray) -> np.ndarray:
    rgb_float = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = color.rgb2gray(rgb_float)
    saturation = color.rgb2hsv(rgb_float)[..., 1]
    signal = (
        0.50 * exposure.rescale_intensity(np.asarray(tissue_score, dtype=np.float32), out_range=(0.0, 1.0))
        + 0.30 * exposure.rescale_intensity(np.asarray(hematoxylin, dtype=np.float32), out_range=(0.0, 1.0))
        + 0.20 * saturation.astype(np.float32)
    )
    weak_background = (gray > 0.90) & (saturation < 0.10) & (signal < 0.30)
    background = _border_connected_background(weak_background)
    mask = ~background
    mask &= signal > 0.04
    mask = _clean_binary_mask(mask, min_fraction=0.01)
    if float(mask.mean()) > 0.92:
        stricter = signal > float(np.percentile(signal, 55))
        mask = _clean_binary_mask(stricter, min_fraction=0.01)
        mask = morphology.convex_hull_image(mask)
    return mask.astype(bool)


def _refine_oct_mask(image: np.ndarray) -> np.ndarray:
    img = exposure.rescale_intensity(np.asarray(image, dtype=np.float32), out_range=(0.0, 1.0))
    smooth = filters.gaussian(img, sigma=1.2)
    weak_background = smooth < max(0.08, float(np.percentile(smooth, 18)))
    background = _border_connected_background(weak_background)
    mask = ~background
    mask &= smooth > max(0.10, float(np.percentile(smooth[smooth > 0], 35)) if np.any(smooth > 0) else 0.10)
    mask = _clean_binary_mask(mask, min_fraction=0.01)
    if float(mask.mean()) > 0.90:
        stronger = smooth > max(0.16, float(np.percentile(smooth[smooth > 0], 55)) if np.any(smooth > 0) else 0.16)
        mask = _clean_binary_mask(stronger, min_fraction=0.01)
        mask = morphology.convex_hull_image(mask)
    return mask.astype(bool)


def _laplace_feature(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lap = np.abs(filters.laplace(filters.gaussian(np.asarray(image, dtype=np.float32), sigma=1.0)))
    return (_rescale01(lap) * mask.astype(np.float32)).astype(np.float32)


def _mix_channels(channels: dict[str, np.ndarray], weights: dict[str, float], mask: np.ndarray) -> np.ndarray:
    total = np.zeros_like(next(iter(channels.values())), dtype=np.float32)
    weight_sum = 0.0
    for key, weight in weights.items():
        total += float(weight) * np.asarray(channels[key], dtype=np.float32)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return np.zeros_like(total, dtype=np.float32)
    return (_rescale01(total / weight_sum) * mask.astype(np.float32)).astype(np.float32)


def extract_he_canonical_feature(rgb: np.ndarray, preset_name: str) -> dict[str, np.ndarray]:
    preset = FEATURE_PRESETS[preset_name]
    rgb_float = np.asarray(rgb, dtype=np.uint8).astype(np.float32) / 255.0
    hema = exposure.rescale_intensity(color.rgb2hed(rgb_float)[..., 0], out_range=(0.0, 1.0)).astype(np.float32)
    tissue_score = (
        -np.log(np.clip(rgb_float, 1.0 / 255.0, 1.0)).sum(axis=2)
        + 0.5 * color.rgb2hsv(rgb_float)[..., 1]
    ).astype(np.float32)
    mask = _refine_he_mask(np.asarray(rgb, dtype=np.uint8), hema, tissue_score)
    grad = _gradient_map(hema, mask)
    distance = _distance_transform_map(mask)
    laplace = _laplace_feature(hema, mask)
    feature = _mix_channels(
        {
            "hema": _rescale01(hema) * mask.astype(np.float32),
            "grad": grad,
            "distance": distance,
            "laplace": laplace,
        },
        preset.he_weights,
        mask,
    )
    return {
        "mask": mask,
        "base_intensity": hema,
        "gradient": grad,
        "distance": distance,
        "laplace": laplace,
        "canonical_feature": feature,
    }


def extract_oct_canonical_feature(image: np.ndarray, preset_name: str) -> dict[str, np.ndarray]:
    preset = FEATURE_PRESETS[preset_name]
    gray = color.rgb2gray(np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32) if image.ndim == 3 else np.asarray(image, dtype=np.float32)
    initial_mask = _refine_oct_mask(gray)
    clahe = _enhance_oct_contrast(gray, initial_mask)
    mask = _refine_oct_mask(clahe)
    grad = _gradient_map(clahe, mask)
    distance = _distance_transform_map(mask)
    laplace = _laplace_feature(clahe, mask)
    feature = _mix_channels(
        {
            "clahe": _rescale01(clahe) * mask.astype(np.float32),
            "grad": grad,
            "distance": distance,
            "laplace": laplace,
        },
        preset.oct_weights,
        mask,
    )
    return {
        "mask": mask,
        "base_intensity": clahe,
        "gradient": grad,
        "distance": distance,
        "laplace": laplace,
        "canonical_feature": feature,
    }


def canonical_feature_similarity(he_maps: dict[str, np.ndarray], oct_maps: dict[str, np.ndarray]) -> dict[str, float]:
    overlap = he_maps["mask"] & oct_maps["mask"]
    union = he_maps["mask"] | oct_maps["mask"]
    feature_corr = _masked_correlation(he_maps["canonical_feature"], oct_maps["canonical_feature"], overlap if overlap.any() else union)
    grad_corr = _masked_correlation(he_maps["gradient"], oct_maps["gradient"], overlap if overlap.any() else union)
    distance_corr = _masked_correlation(he_maps["distance"], oct_maps["distance"], union)
    score = float(0.45 * feature_corr + 0.25 * grad_corr + 0.30 * distance_corr)
    return {
        "canonical_feature_corr": float(feature_corr),
        "gradient_corr": float(grad_corr),
        "distance_corr": float(distance_corr),
        "combined_score": score,
    }
