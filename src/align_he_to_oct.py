from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from skimage import transform

sys.path.append(str(Path(__file__).resolve().parent))

from experimental_similarity.standardized_features import (  # noqa: E402
    _distance_transform_map,
    _masked_correlation,
    extract_he_canonical_feature,
    extract_oct_canonical_feature,
)
from oct_he_pipeline import (  # noqa: E402
    _clean_binary_mask,
    _contour_overlay,
    _false_color_overlay,
    _make_checkerboard,
    _oct_gray_to_rgb,
    _rgb_image_to_uint8,
    _save_gray_png,
    _save_rgb_png,
    find_sample_paths,
    load_he_first_frame,
    load_oct_memmap,
)


Image.MAX_IMAGE_PIXELS = None

FEATURE_PRESET_NAME = "gradient_distance_v1"


@dataclass
class SearchResult:
    scale: float
    rotation_deg: float
    tilt_x_deg: float
    tilt_y_deg: float
    translation_y: float
    translation_x: float
    score: float
    details: dict[str, float]


def _resize_rgb(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return transform.resize(rgb, (*shape, 3), preserve_range=True, anti_aliasing=True, order=1).astype(np.uint8)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2).astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"Unsupported H&E image shape: {array.shape}")
    if array.shape[2] == 3:
        return array.astype(np.uint8)
    if array.shape[2] == 4:
        rgb = array[..., :3].astype(np.float32)
        alpha = array[..., 3:4].astype(np.float32) / 255.0
        composited = rgb * alpha + 255.0 * (1.0 - alpha)
        return np.clip(np.round(composited), 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported H&E channel count: {array.shape[2]}")


def _resize_gray(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return transform.resize(image, shape, preserve_range=True, anti_aliasing=True, order=1).astype(np.float32)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return transform.resize(mask.astype(np.float32), shape, preserve_range=True, anti_aliasing=False, order=0) > 0.5


def _search_shape_from_oct(oct_shape: tuple[int, int], max_dim: int = 320) -> tuple[int, int]:
    scale = min(1.0, float(max_dim) / max(oct_shape))
    return (max(128, int(round(oct_shape[0] * scale))), max(128, int(round(oct_shape[1] * scale))))


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask.astype(bool))
    return mask.astype(bool) & ~eroded


def _border_connected(mask: np.ndarray) -> np.ndarray:
    seeds = np.zeros_like(mask, dtype=bool)
    seeds[[0, -1], :] = True
    seeds[:, [0, -1]] = True
    return ndimage.binary_propagation(seeds & mask.astype(bool), mask=mask.astype(bool))


def _principal_axis_angle(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if coords.shape[0] < 16:
        return 0.0
    center = coords.mean(axis=0)
    cov = np.cov((coords - center).T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return float(math.degrees(math.atan2(float(axis[0]), float(axis[1]))))


def _mask_center(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.shape[0] < 4:
        return np.array([(mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0], dtype=np.float32)
    return coords.mean(axis=0).astype(np.float32)


def _estimate_initial_transform(oct_mask: np.ndarray, he_mask: np.ndarray) -> tuple[float, float]:
    oct_area = max(1.0, float(oct_mask.sum()))
    he_area = max(1.0, float(he_mask.sum()))
    scale = float(np.clip(math.sqrt(oct_area / he_area), 0.45, 1.8))
    rotation_deg = float(np.clip(_principal_axis_angle(oct_mask) - _principal_axis_angle(he_mask), -60.0, 60.0))
    return scale, rotation_deg


def _build_affine_matrix(
    scale: float,
    rotation_deg: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    translation_y: float,
    translation_x: float,
    source_center_yx: np.ndarray,
    target_center_yx: np.ndarray,
) -> np.ndarray:
    shrink_x = max(0.65, math.cos(math.radians(float(tilt_x_deg))))
    shrink_y = max(0.65, math.cos(math.radians(float(tilt_y_deg))))
    sx = float(scale) * shrink_x
    sy = float(scale) * shrink_y
    theta = math.radians(float(rotation_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    linear = np.array(
        [
            [cos_t * sx, -sin_t * sy],
            [sin_t * sx, cos_t * sy],
        ],
        dtype=np.float32,
    )
    src_xy = np.array([float(source_center_yx[1]), float(source_center_yx[0])], dtype=np.float32)
    dst_xy = np.array(
        [
            float(target_center_yx[1]) + float(translation_x),
            float(target_center_yx[0]) + float(translation_y),
        ],
        dtype=np.float32,
    )
    offset = dst_xy - linear @ src_xy
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, :2] = linear
    matrix[0, 2] = offset[0]
    matrix[1, 2] = offset[1]
    return matrix


def _warp_with_matrix(image: np.ndarray, matrix: np.ndarray, output_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    affine = transform.AffineTransform(matrix=matrix)
    warped = transform.warp(
        image,
        inverse_map=affine.inverse,
        output_shape=output_shape,
        preserve_range=True,
        order=order,
        mode="constant",
        cval=0.0,
    )
    return warped.astype(np.float32)


def _warp_he_maps(
    he_maps: dict[str, np.ndarray],
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    moved_mask = _warp_with_matrix(he_maps["mask"].astype(np.float32), matrix, output_shape, order=0) > 0.5
    moved_feature = _warp_with_matrix(he_maps["canonical_feature"], matrix, output_shape, order=1)
    moved_distance = _warp_with_matrix(he_maps["distance"], matrix, output_shape, order=1)
    return {
        "mask": moved_mask,
        "canonical_feature": moved_feature,
        "distance": moved_distance,
    }


def _compose_native_matrix(
    search_matrix: np.ndarray,
    he_full_shape: tuple[int, int],
    oct_native_shape: tuple[int, int],
    search_shape: tuple[int, int],
) -> np.ndarray:
    search_h, search_w = search_shape
    he_h, he_w = he_full_shape
    oct_h, oct_w = oct_native_shape
    source_scale = np.array(
        [
            [search_w / float(he_w), 0.0, 0.0],
            [0.0, search_h / float(he_h), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target_scale_inv = np.array(
        [
            [oct_w / float(search_w), 0.0, 0.0],
            [0.0, oct_h / float(search_h), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return target_scale_inv @ search_matrix @ source_scale


def _build_oct_xy_maps(
    volume: np.ndarray,
    search_shape: tuple[int, int],
    he_guidance_mask: np.ndarray | None = None,
    z_slice_indices: list[int] | None = None,
) -> dict[str, np.ndarray]:
    search_masks: list[np.ndarray] = []
    search_base: list[np.ndarray] = []
    search_features: list[np.ndarray] = []
    if z_slice_indices is None:
        z_iter = list(range(int(volume.shape[0])))
    else:
        z_iter = [int(idx) for idx in z_slice_indices]
    for z_index in z_iter:
        plane_small = _resize_gray(np.asarray(volume[z_index], dtype=np.float32), search_shape)
        maps = extract_oct_canonical_feature(plane_small, FEATURE_PRESET_NAME)
        search_masks.append(maps["mask"].astype(bool))
        search_base.append(maps["base_intensity"].astype(np.float32))
        search_features.append(maps["canonical_feature"].astype(np.float32))

    occupancy = np.mean(np.stack(search_masks, axis=0).astype(np.float32), axis=0)
    occ_values = occupancy[occupancy > 0]
    if occ_values.size:
        occ_threshold = max(0.15, float(np.percentile(occ_values, 35)))
    else:
        occ_threshold = 0.15
    base_intensity = np.median(np.stack(search_base, axis=0), axis=0).astype(np.float32)
    feature_mean = np.mean(np.stack(search_features, axis=0), axis=0).astype(np.float32)
    initial_mask = occupancy >= occ_threshold
    feature_values = feature_mean[feature_mean > 0]
    if initial_mask.any():
        masked_feature_values = feature_mean[initial_mask]
    else:
        masked_feature_values = feature_values
    occ_support_threshold = max(0.50, float(np.percentile(occ_values, 50))) if occ_values.size else 0.50
    feature_support_threshold = max(0.18, float(np.percentile(feature_values, 30))) if feature_values.size else 0.18
    if occ_threshold >= 0.95 and masked_feature_values.size:
        border_feature_threshold = float(np.percentile(masked_feature_values, 20))
        weak_border_candidate = initial_mask & (feature_mean < border_feature_threshold)
    else:
        weak_border_candidate = initial_mask & (occupancy < occ_support_threshold) & (feature_mean < feature_support_threshold)
    border_artifact = _border_connected(weak_border_candidate)
    refined_mask = initial_mask & ~border_artifact
    mask = _clean_binary_mask(refined_mask, min_fraction=0.01)
    if float(mask.sum()) < 0.55 * float(initial_mask.sum()):
        mask = _clean_binary_mask(initial_mask, min_fraction=0.01)
    if he_guidance_mask is not None:
        guidance = ndimage.binary_dilation(he_guidance_mask.astype(bool), iterations=max(8, int(round(0.04 * max(search_shape)))))
        guided_mask = _clean_binary_mask(mask & guidance, min_fraction=0.01)
        if float(guided_mask.sum()) >= 0.45 * float(mask.sum()):
            mask = guided_mask
    boundary = _mask_boundary(mask)
    feature = (feature_mean * mask.astype(np.float32)).astype(np.float32)
    distance = _distance_transform_map(mask)
    return {
        "mask": mask.astype(bool),
        "boundary": boundary.astype(bool),
        "boundary_confidence": np.clip(occupancy, 0.25, 1.0).astype(np.float32),
        "occupancy": occupancy.astype(np.float32),
        "base_intensity": (base_intensity * mask.astype(np.float32)).astype(np.float32),
        "canonical_feature": feature,
        "distance": distance.astype(np.float32),
    }


def _resolve_z_slice_indices(volume_depth: int, z_oct_start: int | None, z_oct_end: int | None) -> list[int]:
    start_1b = 1 if z_oct_start is None else int(z_oct_start)
    end_1b = volume_depth if z_oct_end is None else int(z_oct_end)
    if start_1b < 1 or end_1b < 1 or start_1b > volume_depth or end_1b > volume_depth:
        raise ValueError(f"OCT z slice range must be within 1..{volume_depth}, got start={start_1b}, end={end_1b}")
    if start_1b > end_1b:
        raise ValueError(f"OCT z slice start must be <= end, got start={start_1b}, end={end_1b}")
    return list(range(start_1b - 1, end_1b))


def _boundary_coverage_score(oct_boundary: np.ndarray, he_boundary: np.ndarray, oct_confidence: np.ndarray) -> tuple[float, float]:
    if not oct_boundary.any() or not he_boundary.any():
        return 0.0, 0.0
    dt_he = ndimage.distance_transform_edt(~he_boundary)
    dt_oct = ndimage.distance_transform_edt(~oct_boundary)
    oct_weights = oct_confidence[oct_boundary]
    oct_cover = float(np.average(np.exp(-dt_he[oct_boundary] / 6.0), weights=oct_weights))
    he_cover = float(np.mean(np.exp(-dt_oct[he_boundary] / 8.0)))
    return oct_cover, he_cover


def _score_alignment(oct_maps: dict[str, np.ndarray], moved_he: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    oct_mask = oct_maps["mask"].astype(bool)
    he_mask = moved_he["mask"].astype(bool)
    if int(he_mask.sum()) < 128:
        return -1.0, {"oct_boundary_cover": 0.0, "he_boundary_cover": 0.0, "oct_coverage": 0.0, "he_precision": 0.0, "distance_corr": 0.0, "feature_corr": 0.0, "area_agreement": 0.0}

    overlap = oct_mask & he_mask
    union = oct_mask | he_mask
    oct_boundary = oct_maps["boundary"].astype(bool)
    he_boundary = _mask_boundary(he_mask)
    oct_boundary_cover, he_boundary_cover = _boundary_coverage_score(oct_boundary, he_boundary, oct_maps["boundary_confidence"])
    oct_coverage = float(overlap.sum() / max(1, oct_mask.sum()))
    he_precision = float(overlap.sum() / max(1, he_mask.sum()))
    distance_corr = _masked_correlation(moved_he["distance"], oct_maps["distance"], union)
    feature_corr = _masked_correlation(
        moved_he["canonical_feature"],
        oct_maps["canonical_feature"],
        overlap if int(overlap.sum()) >= 128 else union,
    )
    area_ratio = float(he_mask.sum() / max(1, oct_mask.sum()))
    area_agreement = float(math.exp(-abs(math.log(max(area_ratio, 1e-6)))))
    score = float(
        0.36 * oct_boundary_cover
        + 0.10 * he_boundary_cover
        + 0.20 * oct_coverage
        + 0.10 * he_precision
        + 0.16 * distance_corr
        + 0.05 * feature_corr
        + 0.03 * area_agreement
    )
    details = {
        "oct_boundary_cover": oct_boundary_cover,
        "he_boundary_cover": he_boundary_cover,
        "oct_coverage": oct_coverage,
        "he_precision": he_precision,
        "distance_corr": float(distance_corr),
        "feature_corr": float(feature_corr),
        "area_agreement": area_agreement,
    }
    return score, details


def _initial_candidates(init_scale: float, init_rotation: float) -> list[tuple[float, float, float, float]]:
    scales = [init_scale * factor for factor in (0.78, 0.92, 1.0, 1.10, 1.24)]
    rotations = [init_rotation + delta for delta in (-25.0, -10.0, 0.0, 10.0, 25.0)]
    tilts = [(0.0, 0.0), (-10.0, 0.0), (10.0, 0.0), (0.0, -10.0), (0.0, 10.0)]
    candidates: list[tuple[float, float, float, float]] = []
    for scale in scales:
        for rotation in rotations:
            for tilt_x, tilt_y in tilts:
                candidates.append((float(scale), float(rotation), float(tilt_x), float(tilt_y)))
    return candidates


def align_he_to_oct(
    sample_dir: Path,
    output_dir: Path,
    use_he_mask: bool = False,
    z_oct_start: int | None = None,
    z_oct_end: int | None = None,
) -> dict[str, Any]:
    paths = find_sample_paths(sample_dir)
    he_rgb_full, _, _ = load_he_first_frame(paths.he_path)
    he_rgb_full = _ensure_rgb(he_rgb_full)
    oct_volume = load_oct_memmap(paths.oct_path)
    z_slice_indices = _resolve_z_slice_indices(int(oct_volume.shape[0]), z_oct_start, z_oct_end)
    oct_native_shape = tuple(int(v) for v in oct_volume.shape[1:])
    search_shape = _search_shape_from_oct(oct_native_shape, max_dim=320)

    he_search_rgb = _resize_rgb(he_rgb_full, search_shape)
    he_search_maps = extract_he_canonical_feature(he_search_rgb, FEATURE_PRESET_NAME)
    oct_search_maps = _build_oct_xy_maps(oct_volume, search_shape, z_slice_indices=z_slice_indices)
    he_center = _mask_center(he_search_maps["mask"])
    oct_center = _mask_center(oct_search_maps["mask"])
    init_scale, init_rotation = _estimate_initial_transform(oct_search_maps["mask"], he_search_maps["mask"])
    if use_he_mask:
        guidance_matrix = _build_affine_matrix(
            scale=init_scale,
            rotation_deg=init_rotation,
            tilt_x_deg=0.0,
            tilt_y_deg=0.0,
            translation_y=0.0,
            translation_x=0.0,
            source_center_yx=he_center,
            target_center_yx=oct_center,
        )
        he_guidance_mask = _warp_with_matrix(he_search_maps["mask"].astype(np.float32), guidance_matrix, search_shape, order=0) > 0.5
        oct_search_maps = _build_oct_xy_maps(oct_volume, search_shape, he_guidance_mask=he_guidance_mask, z_slice_indices=z_slice_indices)
        oct_center = _mask_center(oct_search_maps["mask"])
        init_scale, init_rotation = _estimate_initial_transform(oct_search_maps["mask"], he_search_maps["mask"])

    coarse_ranked: list[tuple[float, np.ndarray]] = []
    for scale, rotation_deg, tilt_x_deg, tilt_y_deg in _initial_candidates(init_scale, init_rotation):
        search_matrix = _build_affine_matrix(
            scale=scale,
            rotation_deg=rotation_deg,
            tilt_x_deg=tilt_x_deg,
            tilt_y_deg=tilt_y_deg,
            translation_y=0.0,
            translation_x=0.0,
            source_center_yx=he_center,
            target_center_yx=oct_center,
        )
        moved_he = _warp_he_maps(he_search_maps, search_matrix, search_shape)
        score, _ = _score_alignment(oct_search_maps, moved_he)
        coarse_ranked.append((score, np.array([scale, rotation_deg, tilt_x_deg, tilt_y_deg, 0.0, 0.0], dtype=np.float32)))
    coarse_ranked.sort(key=lambda item: item[0], reverse=True)

    best_result: SearchResult | None = None

    def objective(params: np.ndarray) -> float:
        scale, rotation_deg, tilt_x_deg, tilt_y_deg, ty, tx = [float(v) for v in params]
        if not (0.35 <= scale <= 1.8):
            return 1.0
        if abs(rotation_deg) > 75.0:
            return 1.0
        if abs(tilt_x_deg) > 22.0 or abs(tilt_y_deg) > 22.0:
            return 1.0
        if abs(ty) > search_shape[0] * 0.28 or abs(tx) > search_shape[1] * 0.28:
            return 1.0
        search_matrix = _build_affine_matrix(
            scale=scale,
            rotation_deg=rotation_deg,
            tilt_x_deg=tilt_x_deg,
            tilt_y_deg=tilt_y_deg,
            translation_y=ty,
            translation_x=tx,
            source_center_yx=he_center,
            target_center_yx=oct_center,
        )
        moved_he = _warp_he_maps(he_search_maps, search_matrix, search_shape)
        score, _ = _score_alignment(oct_search_maps, moved_he)
        regularization = 0.002 * (abs(tilt_x_deg) + abs(tilt_y_deg)) + 0.0005 * (abs(ty) + abs(tx))
        return float(-(score - regularization))

    for _, seed in coarse_ranked[:4]:
        result = optimize.minimize(
            objective,
            x0=seed,
            method="Powell",
            options={"maxiter": 28, "disp": False},
        )
        params = result.x if result.success else seed
        search_matrix = _build_affine_matrix(
            scale=float(params[0]),
            rotation_deg=float(params[1]),
            tilt_x_deg=float(params[2]),
            tilt_y_deg=float(params[3]),
            translation_y=float(params[4]),
            translation_x=float(params[5]),
            source_center_yx=he_center,
            target_center_yx=oct_center,
        )
        moved_he = _warp_he_maps(he_search_maps, search_matrix, search_shape)
        score, details = _score_alignment(oct_search_maps, moved_he)
        candidate = SearchResult(
            scale=float(params[0]),
            rotation_deg=float(params[1]),
            tilt_x_deg=float(params[2]),
            tilt_y_deg=float(params[3]),
            translation_y=float(params[4]),
            translation_x=float(params[5]),
            score=float(score),
            details=details,
        )
        if best_result is None or candidate.score > best_result.score:
            best_result = candidate

    if best_result is None:
        raise RuntimeError("Failed to estimate an HE-to-OCT alignment.")

    best_search_matrix = _build_affine_matrix(
        scale=best_result.scale,
        rotation_deg=best_result.rotation_deg,
        tilt_x_deg=best_result.tilt_x_deg,
        tilt_y_deg=best_result.tilt_y_deg,
        translation_y=best_result.translation_y,
        translation_x=best_result.translation_x,
        source_center_yx=he_center,
        target_center_yx=oct_center,
    )
    native_matrix = _compose_native_matrix(best_search_matrix, he_rgb_full.shape[:2], oct_native_shape, search_shape)

    oct_native_mask = _resize_mask(oct_search_maps["mask"], oct_native_shape)
    oct_native_display = (_resize_gray(oct_search_maps["base_intensity"], oct_native_shape) * oct_native_mask.astype(np.float32)).astype(np.float32)
    warped_he_rgb = _warp_with_matrix(he_rgb_full.astype(np.float32), native_matrix, oct_native_shape, order=1)
    warped_he_mask_search = _warp_with_matrix(he_search_maps["mask"].astype(np.float32), best_search_matrix, search_shape, order=0) > 0.5
    warped_he_mask = _resize_mask(warped_he_mask_search, oct_native_shape)
    overlap_mask = warped_he_mask & oct_native_mask
    overlap_rgb = warped_he_rgb * overlap_mask[..., None].astype(np.float32)

    overlay_false_color = _false_color_overlay(warped_he_rgb, oct_native_display, oct_native_mask)
    overlay_contours = _contour_overlay(warped_he_rgb, warped_he_mask, oct_native_mask)
    checkerboard = _make_checkerboard(_rgb_image_to_uint8(warped_he_rgb), _oct_gray_to_rgb(oct_native_display), tile=48)

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb_png(output_dir / "he_warped_to_oct_xy.png", warped_he_rgb)
    _save_rgb_png(output_dir / "he_warped_overlap_only.png", overlap_rgb)
    _save_gray_png(output_dir / "oct_xy_projection.png", oct_native_display)
    _save_gray_png(output_dir / "oct_volume_mask_xy.png", oct_native_mask.astype(np.float32))
    _save_gray_png(output_dir / "he_warped_mask.png", warped_he_mask.astype(np.float32))
    _save_gray_png(output_dir / "overlap_mask.png", overlap_mask.astype(np.float32))
    _save_rgb_png(output_dir / "overlay_false_color.png", overlay_false_color)
    _save_rgb_png(output_dir / "overlay_contours.png", overlay_contours)
    _save_rgb_png(output_dir / "overlay_checkerboard.png", checkerboard)

    summary = {
        "sample_dir": str(sample_dir.resolve()),
        "he_path": str(paths.he_path),
        "oct_path": str(paths.oct_path),
        "feature_preset_name": FEATURE_PRESET_NAME,
        "oct_alignment_mode": "xy_volume_projection",
        "use_he_mask_guidance": bool(use_he_mask),
        "z_oct_start": int(z_slice_indices[0] + 1),
        "z_oct_end": int(z_slice_indices[-1] + 1),
        "search_shape": list(search_shape),
        "oct_output_shape": list(oct_native_shape),
        "he_transform_into_oct_xy": {
            "scale": best_result.scale,
            "rotation_deg": best_result.rotation_deg,
            "tilt_x_deg": best_result.tilt_x_deg,
            "tilt_y_deg": best_result.tilt_y_deg,
            "translation_y": best_result.translation_y,
            "translation_x": best_result.translation_x,
        },
        "score": best_result.score,
        "score_details": best_result.details,
        "output_images": {
            "oct_xy_projection": str(output_dir / "oct_xy_projection.png"),
            "oct_volume_mask_xy": str(output_dir / "oct_volume_mask_xy.png"),
            "he_warped_to_oct_xy": str(output_dir / "he_warped_to_oct_xy.png"),
            "he_warped_mask": str(output_dir / "he_warped_mask.png"),
            "overlap_mask": str(output_dir / "overlap_mask.png"),
            "he_warped_overlap_only": str(output_dir / "he_warped_overlap_only.png"),
            "overlay_false_color": str(output_dir / "overlay_false_color.png"),
            "overlay_contours": str(output_dir / "overlay_contours.png"),
            "overlay_checkerboard": str(output_dir / "overlay_checkerboard.png"),
        },
    }
    (output_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary-driven HE-to-OCT alignment using the OCT XY volume support.")
    parser.add_argument("sample_dir", type=Path, help="Sample directory containing he/ and oct/ subfolders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where outputs will be saved.")
    parser.add_argument("--use-he-mask", action="store_true", help="Use the H&E tissue mask as a broad OCT shape prior during OCT mask estimation.")
    parser.add_argument("--z-oct-start", type=int, default=None, help="1-based starting OCT z slice to include when building the OCT xy mask.")
    parser.add_argument("--z-oct-end", type=int, default=None, help="1-based ending OCT z slice to include when building the OCT xy mask.")
    args = parser.parse_args()

    summary = align_he_to_oct(
        args.sample_dir,
        args.output_dir,
        use_he_mask=args.use_he_mask,
        z_oct_start=args.z_oct_start,
        z_oct_end=args.z_oct_end,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
