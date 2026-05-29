from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, optimize
from skimage import color, exposure, feature, filters, morphology, transform


Image.MAX_IMAGE_PIXELS = None


@dataclass
class SamplePaths:
    sample_dir: Path
    he_path: Path
    oct_path: Path


@dataclass
class TiffSeriesInfo:
    index: int
    shape: tuple[int, ...]
    dtype: str
    axes: str


@dataclass
class ExplorationReport:
    sample_dir: str
    he_path: str
    oct_path: str
    he_size_bytes: int
    oct_size_bytes: int
    he_series: list[TiffSeriesInfo]
    oct_series: list[TiffSeriesInfo]
    he_frames: int
    he_image_shape: tuple[int, int, int]
    he_thumbnail_shape: tuple[int, ...] | None
    he_resolution: dict[str, Any]
    he_channel_means: list[float]
    he_tissue_fraction_rough: float
    oct_shape: tuple[int, int, int]
    oct_dtype: str
    oct_imagej_metadata: dict[str, Any]
    oct_resolution: dict[str, Any]
    oct_virtual_size_bytes: int
    oct_sparse_slice_summaries: list[dict[str, Any]]
    oct_coarse_tissue_peak_z: int
    oct_coarse_tissue_fraction_peak: float
    oct_coarse_yx_stride: int


@dataclass
class HePreprocessed:
    rgb: np.ndarray
    hematoxylin: np.ndarray
    tissue_score: np.ndarray
    feature: np.ndarray
    landmark: np.ndarray
    descriptor: np.ndarray
    mask: np.ndarray
    edges: np.ndarray


@dataclass
class OctPreprocessedPlane:
    raw: np.ndarray
    normalized: np.ndarray
    dark_projection: np.ndarray
    feature: np.ndarray
    landmark: np.ndarray
    descriptor: np.ndarray
    mask: np.ndarray
    edges: np.ndarray


@dataclass
class PoseResult:
    z_index: float
    rx_deg: float
    ry_deg: float
    rz_deg: float
    score: float


@dataclass
class SimilarityResult:
    scale: float
    rotation_deg: float
    translation_y: float
    translation_x: float
    score: float


def find_sample_paths(sample_dir: str | Path) -> SamplePaths:
    sample_path = Path(sample_dir).resolve()
    he_files = sorted((sample_path / "he").glob("*.tif*"))
    oct_files = sorted((sample_path / "oct").glob("*.tif*"))
    oct_files += sorted((sample_path / "oct").glob("*.dcm"))
    if not he_files:
        raise FileNotFoundError(f"No H&E TIFF found under {sample_path / 'he'}")
    if not oct_files:
        raise FileNotFoundError(f"No OCT TIFF found under {sample_path / 'oct'}")
    return SamplePaths(sample_dir=sample_path, he_path=he_files[0], oct_path=oct_files[0])


def describe_tiff(path: Path) -> list[TiffSeriesInfo]:
    with tifffile.TiffFile(path) as tif:
        return [
            TiffSeriesInfo(index=i, shape=tuple(series.shape), dtype=str(series.dtype), axes=series.axes)
            for i, series in enumerate(tif.series)
        ]


def describe_dicom(path: Path) -> list[TiffSeriesInfo]:
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        shape = tuple(int(v) for v in meta.get("shape", ()))
        if len(shape) == 2:
            shape = (1, *shape)
        dtype = str(np.dtype(np.uint8 if meta.get("PixelRepresentation", 0) == 0 else np.int16))
        return [TiffSeriesInfo(index=0, shape=shape, dtype=dtype, axes="ZYX")]
    finally:
        reader.close()


def _resolution_dict(page: tifffile.TiffPage) -> dict[str, Any]:
    def read_tag(name: str) -> Any | None:
        tag = page.tags.get(name)
        return tag.value if tag is not None else None

    return {
        "x_resolution": read_tag("XResolution"),
        "y_resolution": read_tag("YResolution"),
        "resolution_unit": read_tag("ResolutionUnit"),
    }


def _dicom_resolution_dict(meta: dict[str, Any]) -> dict[str, Any]:
    spacing = meta.get("sampling") or meta.get("PixelSpacing")
    return {
        "x_resolution": spacing,
        "y_resolution": spacing,
        "resolution_unit": "dicom_sampling",
    }


def load_he_first_frame(path: Path) -> tuple[np.ndarray, np.ndarray | None, int]:
    image = Image.open(path)
    first = np.array(image)
    thumb = None
    n_frames = getattr(image, "n_frames", 1)
    if n_frames > 1:
        image.seek(1)
        thumb = np.array(image)
    return first, thumb, n_frames


def load_oct_memmap(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".dcm":
        return load_oct_dicom(path)
    return tifffile.memmap(path)


def load_oct_dicom(path: Path) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        shape = tuple(int(v) for v in meta.get("shape", ()))
        if len(shape) == 2:
            shape = (1, *shape)
        num_frames = int(meta.get("NumberOfFrames", shape[0]))
        rows = int(meta.get("Rows", shape[-2]))
        cols = int(meta.get("Columns", shape[-1]))
        first = np.asarray(reader.get_data(0))
        if meta.get("PixelRepresentation", 0) == 0 and first.dtype == np.int8:
            first = first.view(np.uint8)
        volume = np.empty((num_frames, rows, cols), dtype=first.dtype)
        volume[0] = first
        for idx in range(1, num_frames):
            frame = np.asarray(reader.get_data(idx))
            if meta.get("PixelRepresentation", 0) == 0 and frame.dtype == np.int8:
                frame = frame.view(np.uint8)
            volume[idx] = frame
        return volume
    finally:
        reader.close()


def _rough_tissue_fraction_he(rgb: np.ndarray) -> float:
    return float((rgb.mean(axis=2) < 245).mean())


def _oct_slice_percentiles(volume: np.ndarray, indices: list[int]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for idx in indices:
        plane = np.asarray(volume[idx], dtype=np.float32)
        values = np.percentile(plane, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist()
        summaries.append(
            {
                "z_index": idx,
                "percentiles": values,
                "nonzero_fraction": float((plane > 0).mean()),
            }
        )
    return summaries


def _clean_binary_mask(mask: np.ndarray, min_fraction: float = 0.002) -> np.ndarray:
    mask = morphology.binary_closing(mask, morphology.disk(3))
    mask = ndimage.binary_fill_holes(mask)
    min_size = max(64, int(mask.size * min_fraction))
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    keep = int(np.argmax(sizes) + 1)
    return labels == keep


def _rescale01(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return image.astype(np.float32)
    lo = float(np.percentile(image, 1))
    hi = float(np.percentile(image, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _multiscale_blob_response(image: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    image = _rescale01(image)
    responses = []
    for sigma in sigmas:
        response = -ndimage.gaussian_laplace(image, sigma=sigma)
        responses.append(np.clip(response * (sigma ** 2), 0.0, None))
    if not responses:
        return np.zeros_like(image, dtype=np.float32)
    return filters.gaussian(np.max(np.stack(responses, axis=0), axis=0), sigma=0.8).astype(np.float32)


def _shift_image(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(image, dtype=np.float32)
    src_y0 = max(0, -dy)
    src_y1 = image.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = image.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def _mind_descriptor(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    img = _rescale01(image)
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
    local_var = np.zeros_like(img, dtype=np.float32)
    for dy, dx in offsets[:4]:
        shifted = _shift_image(img, dy, dx)
        local_var += filters.gaussian((img - shifted) ** 2, sigma=0.8)
    local_var = np.clip(local_var / 4.0, 1e-4, None)
    channels: list[np.ndarray] = []
    for dy, dx in offsets:
        shifted = _shift_image(img, dy, dx)
        ssd = filters.gaussian((img - shifted) ** 2, sigma=0.8)
        channel = np.exp(-ssd / local_var)
        channels.append(channel.astype(np.float32))
    descriptor = np.stack(channels, axis=-1)
    descriptor /= np.clip(descriptor.mean(axis=-1, keepdims=True), 1e-4, None)
    if mask is not None:
        descriptor *= mask.astype(np.float32)[..., None]
    return descriptor.astype(np.float32)


def _orientation_structure_descriptor(image: np.ndarray, mask: np.ndarray | None = None, bins: int = 6) -> np.ndarray:
    img = _rescale01(image)
    grad_y = filters.sobel_h(img)
    grad_x = filters.sobel_v(img)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x)
    centers = np.linspace(-math.pi, math.pi, bins, endpoint=False)
    channels: list[np.ndarray] = []
    for center in centers:
        weight = np.clip(np.cos(angle - center), 0.0, 1.0) ** 2
        channel = filters.gaussian(weight * magnitude, sigma=1.1)
        channels.append(channel.astype(np.float32))
    structure = np.stack(channels, axis=-1)
    structure /= np.clip(structure.mean(axis=-1, keepdims=True), 1e-4, None)
    if mask is not None:
        structure *= mask.astype(np.float32)[..., None]
    return structure.astype(np.float32)


def _build_multimodal_descriptor(feature_img: np.ndarray, landmark: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mind = _mind_descriptor(feature_img, mask=mask)
    orient = _orientation_structure_descriptor(feature_img, mask=mask)
    landmark_channel = (_rescale01(landmark) * mask.astype(np.float32))[..., None]
    return np.concatenate([mind, orient, landmark_channel], axis=-1).astype(np.float32)


def preprocess_he(rgb: np.ndarray, target_size: int = 256) -> HePreprocessed:
    rgb = np.asarray(rgb, dtype=np.uint8)
    scale = min(1.0, float(target_size) / max(rgb.shape[:2]))
    out_shape = (max(32, int(round(rgb.shape[0] * scale))), max(32, int(round(rgb.shape[1] * scale))))
    rgb_small = transform.resize(
        rgb,
        out_shape,
        preserve_range=True,
        anti_aliasing=True,
        order=1,
    ).astype(np.uint8)
    rgb_float = rgb_small.astype(np.float32) / 255.0
    luminance = color.rgb2gray(rgb_float)
    od = -np.log(np.clip(rgb_float, 1.0 / 255.0, 1.0))
    hematoxylin = exposure.rescale_intensity(color.rgb2hed(rgb_float)[..., 0], out_range=(0.0, 1.0))
    tissue_score = od.sum(axis=2) + 0.5 * color.rgb2hsv(rgb_float)[..., 1]
    valid = tissue_score[tissue_score > np.percentile(tissue_score, 5)]
    thresh = filters.threshold_otsu(valid) if valid.size else float(tissue_score.mean())
    mask = _clean_binary_mask(tissue_score > thresh * 0.65)
    landmark = _multiscale_blob_response(luminance * mask.astype(np.float32), sigmas=(1.4, 2.8, 4.2, 6.0))
    feature_img = 0.7 * filters.gaussian(hematoxylin, sigma=1.0) + 0.3 * landmark
    feature_img = exposure.rescale_intensity(feature_img * mask.astype(np.float32), out_range=(0.0, 1.0))
    descriptor = _build_multimodal_descriptor(feature_img, landmark, mask)
    edges = feature.canny(filters.gaussian(feature_img, sigma=0.8), sigma=1.2)
    return HePreprocessed(
        rgb=rgb_small,
        hematoxylin=hematoxylin.astype(np.float32),
        tissue_score=tissue_score.astype(np.float32),
        feature=feature_img.astype(np.float32),
        landmark=landmark.astype(np.float32),
        descriptor=descriptor.astype(np.float32),
        mask=mask.astype(bool),
        edges=edges.astype(bool),
    )


def build_oct_coarse_volume(volume: np.ndarray, target_xy: int = 340) -> tuple[np.ndarray, int]:
    stride = max(1, int(math.ceil(max(volume.shape[1:]) / float(target_xy))))
    return np.asarray(volume[:, ::stride, ::stride], dtype=np.float32), stride


def estimate_oct_center(coarse_volume: np.ndarray) -> tuple[float, float]:
    nz = coarse_volume[coarse_volume > 0]
    threshold = float(np.percentile(nz, 60)) if nz.size else 0.0
    support = coarse_volume > threshold
    projection = support.any(axis=0)
    coords = np.argwhere(projection)
    if coords.size == 0:
        return (coarse_volume.shape[1] - 1) / 2.0, (coarse_volume.shape[2] - 1) / 2.0
    cy, cx = coords.mean(axis=0)
    return float(cy), float(cx)


def preprocess_oct_plane(plane: np.ndarray) -> OctPreprocessedPlane:
    plane_array = np.asarray(plane, dtype=np.float32)
    if plane_array.ndim == 3:
        raw = plane_array[plane_array.shape[0] // 2]
        pooled = np.median(plane_array, axis=0)
        dark_projection = np.percentile(plane_array, 20, axis=0)
    else:
        raw = plane_array
        pooled = plane_array
        dark_projection = plane_array
    nonzero = pooled[pooled > 0]
    if nonzero.size:
        lo, hi = np.percentile(nonzero, [1, 99])
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((pooled - lo) / (hi - lo), 0.0, 1.0)
    log_img = np.log1p(4.0 * norm) / np.log1p(4.0)
    smooth = filters.gaussian(filters.median(log_img, morphology.disk(1)), sigma=1.2)
    thresh_source = smooth[smooth > 0]
    thresh = filters.threshold_otsu(thresh_source) if thresh_source.size else float(smooth.mean())
    mask = _clean_binary_mask(smooth > max(0.12, thresh * 0.7))
    dark_map = filters.gaussian((1.0 - _rescale01(dark_projection)) * mask.astype(np.float32), sigma=0.9)
    landmark = _multiscale_blob_response(dark_map, sigmas=(1.4, 2.8, 4.2, 6.0))
    gradient = filters.gaussian(filters.sobel(smooth), sigma=0.9)
    sheetness = filters.gaussian(np.abs(filters.laplace(smooth)), sigma=0.8)
    feature_img = exposure.rescale_intensity((0.40 * gradient + 0.35 * landmark + 0.25 * sheetness) * mask.astype(np.float32), out_range=(0.0, 1.0))
    descriptor = _build_multimodal_descriptor(feature_img, landmark, mask)
    edges = feature.canny(filters.gaussian(feature_img, sigma=0.7), sigma=1.0)
    return OctPreprocessedPlane(
        raw=raw,
        normalized=smooth.astype(np.float32),
        dark_projection=dark_map.astype(np.float32),
        feature=feature_img.astype(np.float32),
        landmark=landmark.astype(np.float32),
        descriptor=descriptor.astype(np.float32),
        mask=mask.astype(bool),
        edges=edges.astype(bool),
    )


def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def extract_oct_plane(
    volume: np.ndarray,
    z_index: float,
    center_y: float,
    center_x: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    output_shape: tuple[int, int] = (192, 192),
    normal_offset: float = 0.0,
) -> np.ndarray:
    out_h, out_w = output_shape
    yy = np.linspace(-(volume.shape[1] - 1) / 2.0, (volume.shape[1] - 1) / 2.0, out_h, dtype=np.float32)
    xx = np.linspace(-(volume.shape[2] - 1) / 2.0, (volume.shape[2] - 1) / 2.0, out_w, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    grid = np.stack([np.zeros_like(grid_x), grid_y, grid_x], axis=0).reshape(3, -1)
    rotation = _rotation_matrix(rx_deg, ry_deg, rz_deg)
    rotated = rotation @ grid
    rotated += rotation[:, 0:1] * float(normal_offset)
    rotated[0] += z_index
    rotated[1] += center_y
    rotated[2] += center_x
    plane = ndimage.map_coordinates(volume, rotated, order=1, mode="constant", cval=0.0)
    return plane.reshape(output_shape)


def extract_oct_slab(
    volume: np.ndarray,
    z_index: float,
    center_y: float,
    center_x: float,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    output_shape: tuple[int, int] = (192, 192),
    slab_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> np.ndarray:
    planes = [
        extract_oct_plane(
            volume,
            z_index=z_index,
            center_y=center_y,
            center_x=center_x,
            rx_deg=rx_deg,
            ry_deg=ry_deg,
            rz_deg=rz_deg,
            output_shape=output_shape,
            normal_offset=offset,
        )
        for offset in slab_offsets
    ]
    return np.stack(planes, axis=0).astype(np.float32)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return int(y0), int(y1), int(x0), int(x1)


def crop_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int] = (160, 160),
    margin: float = 0.12,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    bbox = _bbox(mask)
    if bbox is None:
        if image.ndim == 3:
            target_shape = (*output_shape, image.shape[2])
        else:
            target_shape = output_shape
        resized_img = transform.resize(image, target_shape, anti_aliasing=True, preserve_range=True, order=order)
        resized_mask = transform.resize(mask.astype(np.float32), output_shape, order=0, preserve_range=True) > 0.5
        return resized_img.astype(np.float32), resized_mask

    y0, y1, x0, x1 = bbox
    h = y1 - y0
    w = x1 - x0
    pad_y = max(2, int(round(h * margin)))
    pad_x = max(2, int(round(w * margin)))
    y0 = max(0, y0 - pad_y)
    y1 = min(mask.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(mask.shape[1], x1 + pad_x)
    crop_img = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if crop_img.ndim == 3:
        target_shape = (*output_shape, crop_img.shape[2])
    else:
        target_shape = output_shape
    resized_img = transform.resize(crop_img, target_shape, anti_aliasing=True, preserve_range=True, order=order)
    resized_mask = transform.resize(crop_mask.astype(np.float32), output_shape, order=0, preserve_range=True) > 0.5
    return resized_img.astype(np.float32), resized_mask


def normalize_crop(
    image: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int] = (160, 160),
    margin: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    return crop_with_mask(image, mask, output_shape=output_shape, margin=margin, order=1)


def resize_full_frame(
    image: np.ndarray,
    output_shape: tuple[int, int] = (160, 160),
    order: int = 1,
) -> np.ndarray:
    if image.ndim == 3:
        target_shape = (*output_shape, image.shape[2])
    else:
        target_shape = output_shape
    resized = transform.resize(image, target_shape, anti_aliasing=True, preserve_range=True, order=order)
    return resized.astype(np.float32)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _edge_score(a_edges: np.ndarray, b_edges: np.ndarray) -> float:
    if not a_edges.any() or not b_edges.any():
        return 0.0
    dt_a = ndimage.distance_transform_edt(~a_edges)
    dt_b = ndimage.distance_transform_edt(~b_edges)
    chamfer = float(dt_a[b_edges].mean() + dt_b[a_edges].mean()) / 2.0
    return float(math.exp(-chamfer / 8.0))


def _subset_overlap_score(container_mask: np.ndarray, subset_mask: np.ndarray) -> float:
    subset_area = int(subset_mask.astype(bool).sum())
    if subset_area == 0:
        return 0.0
    intersection = int(np.logical_and(container_mask, subset_mask).sum())
    precision = intersection / subset_area
    weak_recall = intersection / max(1, int(container_mask.astype(bool).sum()))
    return float(0.85 * precision + 0.15 * weak_recall)


def _containment_score(container_mask: np.ndarray, subset_mask: np.ndarray) -> float:
    subset_boundary = _mask_boundary(subset_mask.astype(bool))
    if not subset_boundary.any():
        return 0.0
    outside_distance = ndimage.distance_transform_edt(~container_mask.astype(bool))
    mean_distance = float(outside_distance[subset_boundary].mean())
    return float(math.exp(-mean_distance / 4.0))


def _partial_boundary_score(reference_mask: np.ndarray, moving_mask: np.ndarray, quantile: float = 0.35) -> float:
    reference_boundary = _mask_boundary(reference_mask.astype(bool))
    moving_boundary = _mask_boundary(moving_mask.astype(bool))
    if not reference_boundary.any() or not moving_boundary.any():
        return 0.0

    def best_arc_score(source_boundary: np.ndarray, target_boundary: np.ndarray) -> float:
        dists = ndimage.distance_transform_edt(~target_boundary)[source_boundary]
        if dists.size < 16:
            return 0.0
        keep = max(8, int(round(dists.size * quantile)))
        best = np.partition(dists, keep - 1)[:keep]
        return float(math.exp(-float(best.mean()) / 3.5))

    forward = best_arc_score(moving_boundary, reference_boundary)
    reverse = best_arc_score(reference_boundary, moving_boundary)
    return float(0.7 * forward + 0.3 * reverse)


def _normalized_mutual_information(a: np.ndarray, b: np.ndarray, mask: np.ndarray, bins: int = 32) -> float:
    valid = mask.astype(bool)
    if valid.sum() < 128:
        return 0.0
    av = _rescale01(a[valid])
    bv = _rescale01(b[valid])
    hist2d, _, _ = np.histogram2d(av, bv, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]])
    total = float(hist2d.sum())
    if total <= 0.0:
        return 0.0
    pxy = hist2d / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    px_py = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * (np.log(pxy[nz]) - np.log(px_py[nz]))))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    if hx <= 1e-6 or hy <= 1e-6:
        return 0.0
    return float(max(0.0, min(1.0, (2.0 * mi) / (hx + hy))))


def _descriptor_similarity(a_desc: np.ndarray, b_desc: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if valid.sum() < 128:
        return 0.0
    av = np.asarray(a_desc[valid], dtype=np.float32)
    bv = np.asarray(b_desc[valid], dtype=np.float32)
    diff = np.abs(av - bv).mean(axis=1)
    mean_diff = float(diff.mean())
    av_centered = av - av.mean(axis=0, keepdims=True)
    bv_centered = bv - bv.mean(axis=0, keepdims=True)
    numer = np.sum(av_centered * bv_centered, axis=0)
    denom = np.sqrt(np.sum(av_centered ** 2, axis=0) * np.sum(bv_centered ** 2, axis=0))
    corr = np.divide(numer, np.clip(denom, 1e-6, None))
    corr_score = float(np.clip(np.mean(corr), 0.0, 1.0))
    return float(0.6 * math.exp(-mean_diff / 0.35) + 0.4 * corr_score)


def _score_registered_alignment(
    he_mask: np.ndarray,
    he_edges: np.ndarray,
    he_feature: np.ndarray,
    he_landmark: np.ndarray,
    he_descriptor: np.ndarray,
    oct_mask: np.ndarray,
    oct_edges: np.ndarray,
    oct_feature: np.ndarray,
    oct_landmark: np.ndarray,
    oct_descriptor: np.ndarray,
) -> float:
    overlap_mask = he_mask & oct_mask
    return float(
        0.16 * _subset_overlap_score(he_mask, oct_mask)
        + 0.22 * _global_extent_score(he_mask, oct_mask)
        + 0.10 * _containment_score(he_mask, oct_mask)
        + 0.12 * _partial_boundary_score(he_mask, oct_mask)
        + 0.10 * _edge_score(he_edges, oct_edges)
        + 0.22 * _descriptor_similarity(he_descriptor, oct_descriptor, overlap_mask)
        + 0.05 * _normalized_mutual_information(he_feature, oct_feature, overlap_mask)
        + 0.03 * _normalized_mutual_information(he_landmark, oct_landmark, overlap_mask)
    )


def _extent_ratio_score(ratio: float, sigma_under: float = 0.16, sigma_over: float = 0.28) -> float:
    if ratio <= 0.0:
        return 0.0
    if ratio <= 1.0:
        return float(math.exp(-((1.0 - ratio) / sigma_under) ** 2))
    return float(math.exp(-((ratio - 1.0) / sigma_over) ** 2))


def _global_extent_score(reference_mask: np.ndarray, moving_mask: np.ndarray) -> float:
    ref_bbox = _bbox(reference_mask)
    mov_bbox = _bbox(moving_mask)
    if ref_bbox is None or mov_bbox is None:
        return 0.0
    ref_h = max(1, ref_bbox[1] - ref_bbox[0])
    ref_w = max(1, ref_bbox[3] - ref_bbox[2])
    mov_h = max(1, mov_bbox[1] - mov_bbox[0])
    mov_w = max(1, mov_bbox[3] - mov_bbox[2])
    h_score = _extent_ratio_score(mov_h / ref_h)
    w_score = _extent_ratio_score(mov_w / ref_w)
    area_score = _extent_ratio_score(math.sqrt(float(moving_mask.sum()) / max(1.0, float(reference_mask.sum()))))
    hull_score = _dice(morphology.convex_hull_image(reference_mask), morphology.convex_hull_image(moving_mask))
    return float(0.30 * h_score + 0.30 * w_score + 0.20 * area_score + 0.20 * hull_score)


def _tilt_penalty(rx_deg: float, ry_deg: float) -> float:
    return 0.015 * ((float(rx_deg) ** 2 + float(ry_deg) ** 2) / 25.0)


def score_plane(he: HePreprocessed, plane: OctPreprocessedPlane, use_landmarks: bool = False) -> float:
    he_img, he_mask = normalize_crop(he.feature, he.mask, output_shape=(176, 176), margin=0.18)
    he_landmark, _ = normalize_crop(he.landmark, he.mask, output_shape=(176, 176), margin=0.18)
    he_desc = resize_full_frame(he.descriptor, output_shape=(176, 176), order=1)
    oct_img, oct_mask = normalize_crop(plane.feature, plane.mask, output_shape=(176, 176), margin=0.18)
    oct_landmark, _ = normalize_crop(plane.landmark, plane.mask, output_shape=(176, 176), margin=0.18)
    oct_desc = resize_full_frame(plane.descriptor, output_shape=(176, 176), order=1)
    he_edges = feature.canny(filters.gaussian(he_img, sigma=0.8), sigma=1.0)
    oct_edges = feature.canny(filters.gaussian(oct_img, sigma=0.8), sigma=1.0)
    overlap = he_mask & oct_mask
    feature_nmi = _normalized_mutual_information(he_img, oct_img, overlap)
    desc_score = _descriptor_similarity(he_desc, oct_desc, overlap)
    mask_score = _dice(he_mask, oct_mask)
    edge_score = _edge_score(he_edges, oct_edges)
    extent_score = _global_extent_score(he_mask, oct_mask)
    arc_score = _partial_boundary_score(he_mask, oct_mask)
    base_score = 0.28 * desc_score + 0.22 * feature_nmi + 0.18 * mask_score + 0.12 * edge_score + 0.12 * extent_score + 0.08 * arc_score
    if not use_landmarks:
        return float(base_score)
    landmark_score = _normalized_mutual_information(he_landmark, oct_landmark, overlap)
    return float(0.80 * base_score + 0.20 * landmark_score)


def search_best_pose(
    coarse_volume: np.ndarray,
    he: HePreprocessed,
    center_y: float,
    center_x: float,
    output_shape: tuple[int, int] = (192, 192),
) -> tuple[PoseResult, list[PoseResult]]:
    slab5 = (-2.0, -1.0, 0.0, 1.0, 2.0)
    z_candidates: list[PoseResult] = []
    for z_idx in range(coarse_volume.shape[0]):
        plane = preprocess_oct_plane(
            extract_oct_slab(
                coarse_volume,
                z_index=float(z_idx),
                center_y=center_y,
                center_x=center_x,
                rx_deg=0.0,
                ry_deg=0.0,
                rz_deg=0.0,
                output_shape=output_shape,
                slab_offsets=slab5,
            )
        )
        score = score_plane(he, plane, use_landmarks=False)
        z_candidates.append(PoseResult(z_index=float(z_idx), rx_deg=0.0, ry_deg=0.0, rz_deg=0.0, score=score))

    z_candidates.sort(key=lambda item: item.score, reverse=True)
    coarse_angles = [-3.0, 0.0, 3.0]
    coarse_rz = list(np.linspace(0.0, 315.0, 8))
    best = z_candidates[0]
    candidate_pool: list[PoseResult] = [best]
    for seed in z_candidates[:4]:
        for rx_deg in coarse_angles:
            for ry_deg in coarse_angles:
                for rz_deg in coarse_rz:
                    plane = preprocess_oct_plane(
                        extract_oct_slab(
                            coarse_volume,
                            z_index=seed.z_index,
                            center_y=center_y,
                            center_x=center_x,
                            rx_deg=rx_deg,
                            ry_deg=ry_deg,
                            rz_deg=rz_deg,
                            output_shape=output_shape,
                            slab_offsets=slab5,
                        )
                    )
                    score = score_plane(he, plane, use_landmarks=False) - _tilt_penalty(rx_deg, ry_deg)
                    if score > best.score:
                        best = PoseResult(
                            z_index=float(seed.z_index),
                            rx_deg=float(rx_deg),
                            ry_deg=float(ry_deg),
                            rz_deg=float(rz_deg),
                            score=float(score),
                        )
                    candidate_pool.append(
                        PoseResult(
                            z_index=float(seed.z_index),
                            rx_deg=float(rx_deg),
                            ry_deg=float(ry_deg),
                            rz_deg=float(rz_deg),
                            score=float(score),
                        )
                    )

    if best.score < 0.45:
        expanded_angles = [-6.0, -3.0, 0.0, 3.0, 6.0]
        for seed in z_candidates[:2]:
            for rx_deg in expanded_angles:
                for ry_deg in expanded_angles:
                    for rz_deg in coarse_rz:
                        plane = preprocess_oct_plane(
                            extract_oct_slab(
                                coarse_volume,
                                z_index=seed.z_index,
                                center_y=center_y,
                                center_x=center_x,
                                rx_deg=rx_deg,
                                ry_deg=ry_deg,
                                rz_deg=rz_deg,
                                output_shape=output_shape,
                                slab_offsets=slab5,
                            )
                        )
                        score = score_plane(he, plane, use_landmarks=False) - _tilt_penalty(rx_deg, ry_deg)
                        candidate_pool.append(
                            PoseResult(
                                z_index=float(seed.z_index),
                                rx_deg=float(rx_deg),
                                ry_deg=float(ry_deg),
                                rz_deg=float(rz_deg),
                                score=float(score),
                            )
                        )
                        if score > best.score:
                            best = PoseResult(
                                z_index=float(seed.z_index),
                                rx_deg=float(rx_deg),
                                ry_deg=float(ry_deg),
                                rz_deg=float(rz_deg),
                                score=float(score),
                            )

    candidate_pool.sort(key=lambda item: item.score, reverse=True)
    refined_candidates: list[PoseResult] = []
    for seed in candidate_pool[:5]:
        z_values = [value for value in np.linspace(max(0.0, seed.z_index - 1.5), min(coarse_volume.shape[0] - 1, seed.z_index + 1.5), 4)]
        rx_values = [seed.rx_deg + offset for offset in (-1.0, 0.0, 1.0)]
        ry_values = [seed.ry_deg + offset for offset in (-1.0, 0.0, 1.0)]
        rz_values = [float((seed.rz_deg + offset) % 360.0) for offset in (-6.0, 0.0, 6.0)]
        for z_index in z_values:
            for rx_deg in rx_values:
                for ry_deg in ry_values:
                    for rz_deg in rz_values:
                        plane = preprocess_oct_plane(
                            extract_oct_slab(
                                coarse_volume,
                                z_index=float(z_index),
                                center_y=center_y,
                                center_x=center_x,
                                rx_deg=float(rx_deg),
                                ry_deg=float(ry_deg),
                                rz_deg=float(rz_deg),
                                output_shape=output_shape,
                                slab_offsets=slab5,
                            )
                        )
                        score = score_plane(he, plane, use_landmarks=True) - _tilt_penalty(rx_deg, ry_deg)
                        refined_candidates.append(
                            PoseResult(
                                z_index=float(z_index),
                                rx_deg=float(rx_deg),
                                ry_deg=float(ry_deg),
                                rz_deg=float(rz_deg),
                                score=float(score),
                            )
                        )
    if refined_candidates:
        refined_candidates.sort(key=lambda item: item.score, reverse=True)
        best = refined_candidates[0]
        ranked = refined_candidates
    else:
        ranked = candidate_pool

    top_candidates: list[PoseResult] = []
    seen: set[tuple[int, int, int, int]] = set()
    for candidate in ranked:
        key = (
            int(round(candidate.z_index * 2.0)),
            int(round(candidate.rx_deg * 2.0)),
            int(round(candidate.ry_deg * 2.0)),
            int(round(candidate.rz_deg / 3.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        top_candidates.append(candidate)
        if len(top_candidates) >= 5:
            break
    return best, top_candidates


def _mask_centroid_orientation(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    coords = np.argwhere(mask)
    if coords.shape[0] < 3:
        center = np.array([(mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0], dtype=np.float32)
        return center, 0.0, max(1.0, math.sqrt(float(mask.sum()) + 1.0))
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(float(major[0]), float(major[1])))
    scale_hint = max(1.0, math.sqrt(float(mask.sum())))
    return center.astype(np.float32), angle, scale_hint


def warp_similarity(image: np.ndarray, scale: float, rotation_deg: float, translation_y: float, translation_x: float, order: int = 1) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_shift = xx - cx - translation_x
    y_shift = yy - cy - translation_y
    src_x = (cos_t * x_shift + sin_t * y_shift) / scale + cx
    src_y = (-sin_t * x_shift + cos_t * y_shift) / scale + cy
    coords = np.stack([src_y, src_x], axis=0)
    if image.ndim == 2:
        return ndimage.map_coordinates(image, coords, order=order, mode="constant", cval=0.0)
    channels = [
        ndimage.map_coordinates(image[..., channel], coords, order=order, mode="constant", cval=0.0)
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)


def _masked_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool)
    if valid.sum() < 64:
        return 0.0
    av = np.asarray(a[valid], dtype=np.float32)
    bv = np.asarray(b[valid], dtype=np.float32)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-6:
        return 0.0
    return float(max(0.0, np.dot(av, bv) / denom))


def refine_inplane_similarity(
    he_feature: np.ndarray,
    he_mask: np.ndarray,
    he_landmark: np.ndarray,
    he_descriptor: np.ndarray,
    oct_feature: np.ndarray,
    oct_mask: np.ndarray,
    oct_landmark: np.ndarray,
    oct_descriptor: np.ndarray,
) -> SimilarityResult:
    target_size = min(256, he_feature.shape[0], he_feature.shape[1])
    if he_feature.shape[0] != target_size or he_feature.shape[1] != target_size:
        output_shape = (target_size, target_size)
        scale_back_y = he_feature.shape[0] / target_size
        scale_back_x = he_feature.shape[1] / target_size
        he_feature_opt = resize_full_frame(he_feature, output_shape=output_shape, order=1)
        he_mask_opt = resize_full_frame(he_mask.astype(np.float32), output_shape=output_shape, order=0) > 0.5
        he_landmark_opt = resize_full_frame(he_landmark, output_shape=output_shape, order=1)
        he_descriptor_opt = resize_full_frame(he_descriptor, output_shape=output_shape, order=1)
        oct_feature_opt = resize_full_frame(oct_feature, output_shape=output_shape, order=1)
        oct_mask_opt = resize_full_frame(oct_mask.astype(np.float32), output_shape=output_shape, order=0) > 0.5
        oct_landmark_opt = resize_full_frame(oct_landmark, output_shape=output_shape, order=1)
        oct_descriptor_opt = resize_full_frame(oct_descriptor, output_shape=output_shape, order=1)
    else:
        scale_back_y = 1.0
        scale_back_x = 1.0
        he_feature_opt = he_feature
        he_mask_opt = he_mask
        he_landmark_opt = he_landmark
        he_descriptor_opt = he_descriptor
        oct_feature_opt = oct_feature
        oct_mask_opt = oct_mask
        oct_landmark_opt = oct_landmark
        oct_descriptor_opt = oct_descriptor

    he_center, he_angle, he_scale = _mask_centroid_orientation(he_mask_opt)
    oct_center, oct_angle, oct_scale = _mask_centroid_orientation(oct_mask_opt)
    init = np.array(
        [
            np.clip(he_scale / max(oct_scale, 1.0), 0.75, 1.3),
            np.clip(he_angle - oct_angle, -25.0, 25.0),
            float(he_center[0] - oct_center[0]),
            float(he_center[1] - oct_center[1]),
        ],
        dtype=np.float32,
    )

    def score_params(params: np.ndarray) -> float:
        scale, rotation_deg, ty, tx = params
        if not (0.65 <= scale <= 1.45 and abs(rotation_deg) <= 35.0 and abs(ty) <= he_mask_opt.shape[0] / 3.0 and abs(tx) <= he_mask_opt.shape[1] / 3.0):
            return -1.0
        moved_mask = warp_similarity(oct_mask_opt.astype(np.float32), scale, rotation_deg, ty, tx, order=0) > 0.5
        moved_feature = warp_similarity(oct_feature_opt, scale, rotation_deg, ty, tx, order=1)
        moved_landmark = warp_similarity(oct_landmark_opt, scale, rotation_deg, ty, tx, order=1)
        moved_descriptor = warp_similarity(oct_descriptor_opt, scale, rotation_deg, ty, tx, order=1)
        moved_edges = feature.canny(moved_feature, sigma=1.0)
        he_edges = feature.canny(he_feature_opt, sigma=1.0)
        return _score_registered_alignment(
            he_mask_opt,
            he_edges,
            he_feature_opt,
            he_landmark_opt,
            he_descriptor_opt,
            moved_mask,
            moved_edges,
            moved_feature,
            moved_landmark,
            moved_descriptor,
        )

    seeds = [
        init,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.array([init[0], 0.0, init[2], init[3]], dtype=np.float32),
    ]
    best = SimilarityResult(scale=1.0, rotation_deg=0.0, translation_y=0.0, translation_x=0.0, score=-1.0)
    for seed in seeds:
        result = optimize.minimize(
            lambda params: -score_params(params),
            x0=seed,
            method="Powell",
            options={"maxiter": 60, "disp": False},
        )
        score = score_params(result.x if result.success else seed)
        candidate = SimilarityResult(
            scale=float((result.x if result.success else seed)[0]),
            rotation_deg=float((result.x if result.success else seed)[1]),
            translation_y=float((result.x if result.success else seed)[2] * scale_back_y),
            translation_x=float((result.x if result.success else seed)[3] * scale_back_x),
            score=float(score),
        )
        if candidate.score > best.score:
            best = candidate
    return best


def collect_exploration_report(sample_dir: str | Path, coarse_target_xy: int = 340) -> ExplorationReport:
    paths = find_sample_paths(sample_dir)
    he_series = describe_tiff(paths.he_path)
    if paths.oct_path.suffix.lower() == ".dcm":
        oct_series = describe_dicom(paths.oct_path)
    else:
        oct_series = describe_tiff(paths.oct_path)
    he_rgb, he_thumb, he_frames = load_he_first_frame(paths.he_path)
    oct_volume = load_oct_memmap(paths.oct_path)

    with tifffile.TiffFile(paths.he_path) as he_tif:
        he_resolution = _resolution_dict(he_tif.pages[0])
    if paths.oct_path.suffix.lower() == ".dcm":
        reader = imageio.get_reader(paths.oct_path)
        try:
            oct_metadata = dict(reader.get_meta_data())
            oct_resolution = _dicom_resolution_dict(oct_metadata)
        finally:
            reader.close()
    else:
        with tifffile.TiffFile(paths.oct_path) as oct_tif:
            oct_resolution = _resolution_dict(oct_tif.pages[0])
            oct_metadata = dict(oct_tif.imagej_metadata or {})

    sparse_indices = sorted({0, oct_volume.shape[0] // 4, oct_volume.shape[0] // 2, (3 * oct_volume.shape[0]) // 4, oct_volume.shape[0] - 1})
    coarse_volume, stride = build_oct_coarse_volume(oct_volume, target_xy=coarse_target_xy)
    nonzero = coarse_volume[coarse_volume > 0]
    threshold = float(np.percentile(nonzero, 75)) if nonzero.size else 0.0
    support = coarse_volume > threshold
    tissue_fraction_by_z = support.mean(axis=(1, 2))

    return ExplorationReport(
        sample_dir=str(paths.sample_dir),
        he_path=str(paths.he_path),
        oct_path=str(paths.oct_path),
        he_size_bytes=paths.he_path.stat().st_size,
        oct_size_bytes=paths.oct_path.stat().st_size,
        he_series=he_series,
        oct_series=oct_series,
        he_frames=he_frames,
        he_image_shape=tuple(int(v) for v in he_rgb.shape),
        he_thumbnail_shape=tuple(int(v) for v in he_thumb.shape) if he_thumb is not None else None,
        he_resolution=he_resolution,
        he_channel_means=[float(v) for v in he_rgb.reshape(-1, 3).mean(axis=0)],
        he_tissue_fraction_rough=_rough_tissue_fraction_he(he_rgb),
        oct_shape=tuple(int(v) for v in oct_volume.shape),
        oct_dtype=str(oct_volume.dtype),
        oct_imagej_metadata=oct_metadata,
        oct_resolution=oct_resolution,
        oct_virtual_size_bytes=int(np.prod(oct_volume.shape) * np.dtype(oct_volume.dtype).itemsize),
        oct_sparse_slice_summaries=_oct_slice_percentiles(oct_volume, sparse_indices),
        oct_coarse_tissue_peak_z=int(np.argmax(tissue_fraction_by_z)),
        oct_coarse_tissue_fraction_peak=float(tissue_fraction_by_z.max()),
        oct_coarse_yx_stride=stride,
    )


def _make_checkerboard(a_rgb: np.ndarray, b_rgb: np.ndarray, tile: int = 32) -> np.ndarray:
    h, w = a_rgb.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pattern = ((yy // tile) + (xx // tile)) % 2 == 0
    out = a_rgb.copy()
    out[~pattern] = b_rgb[~pattern]
    return out


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask)
    return mask.astype(bool) & ~eroded


def _float_image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    image = exposure.rescale_intensity(image, out_range=(0.0, 1.0))
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def _rgb_image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def _save_gray_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(_float_image_to_uint8(image), mode="L").save(path)


def _save_rgb_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(_rgb_image_to_uint8(image), mode="RGB").save(path)


def _oct_gray_to_rgb(image: np.ndarray) -> np.ndarray:
    gray = _float_image_to_uint8(image)
    return np.stack([gray, gray, gray], axis=-1)


def _false_color_overlay(he_rgb: np.ndarray, oct_gray: np.ndarray, oct_mask: np.ndarray) -> np.ndarray:
    he_u8 = _rgb_image_to_uint8(he_rgb)
    oct_u8 = _float_image_to_uint8(oct_gray)
    oct_color = np.zeros_like(he_u8)
    oct_color[..., 1] = oct_u8
    oct_color[..., 2] = oct_u8
    alpha = 0.35 * oct_mask.astype(np.float32)[..., None]
    blended = he_u8.astype(np.float32) * (1.0 - alpha) + oct_color.astype(np.float32) * alpha
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _contour_overlay(he_rgb: np.ndarray, he_mask: np.ndarray, oct_mask: np.ndarray) -> np.ndarray:
    out = _rgb_image_to_uint8(he_rgb).copy()
    he_boundary = _mask_boundary(he_mask)
    oct_boundary = _mask_boundary(oct_mask)
    out[he_boundary] = np.array([0, 255, 0], dtype=np.uint8)
    out[oct_boundary] = np.array([255, 0, 0], dtype=np.uint8)
    out[he_boundary & oct_boundary] = np.array([255, 255, 0], dtype=np.uint8)
    return out


def _pad_or_crop(center: tuple[int, int], patch_size: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    cy, cx = center
    half = patch_size // 2
    y0 = max(0, min(shape[0] - patch_size, cy - half))
    x0 = max(0, min(shape[1] - patch_size, cx - half))
    y1 = min(shape[0], y0 + patch_size)
    x1 = min(shape[1], x0 + patch_size)
    return slice(y0, y1), slice(x0, x1)


def _select_zoom_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    bbox = _bbox(mask)
    if bbox is None:
        h, w = mask.shape
        return [(h // 2, w // 2)]
    y0, y1, x0, x1 = bbox
    h = y1 - y0
    w = x1 - x0
    points = [
        (int(round(y0 + 0.22 * h)), int(round(x0 + 0.22 * w))),
        (int(round(y0 + 0.22 * h)), int(round(x0 + 0.78 * w))),
        (int(round(y0 + 0.78 * h)), int(round(x0 + 0.22 * w))),
        (int(round(y0 + 0.78 * h)), int(round(x0 + 0.78 * w))),
        (int(round(y0 + 0.50 * h)), int(round(x0 + 0.50 * w))),
    ]
    unique_points: list[tuple[int, int]] = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
    return unique_points


def _make_zoom_montage(he_patch: np.ndarray, oct_patch: np.ndarray, overlay_patch: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    target_shape = (he_patch.shape[0] * scale_factor, he_patch.shape[1] * scale_factor)
    he_big = transform.resize(he_patch, (*target_shape, 3), preserve_range=True, anti_aliasing=False, order=0).astype(np.uint8)
    oct_big = transform.resize(oct_patch, (*target_shape, 3), preserve_range=True, anti_aliasing=False, order=0).astype(np.uint8)
    overlay_big = transform.resize(overlay_patch, (*target_shape, 3), preserve_range=True, anti_aliasing=False, order=0).astype(np.uint8)
    return np.concatenate([he_big, oct_big, overlay_big], axis=1)


def _load_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _labeled_tile(image: np.ndarray, label: str, title_height: int = 22) -> np.ndarray:
    image_u8 = _rgb_image_to_uint8(image)
    canvas = Image.new("RGB", (image_u8.shape[1], image_u8.shape[0] + title_height), color=(250, 250, 250))
    canvas.paste(Image.fromarray(image_u8, mode="RGB"), (0, title_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), label, fill=(20, 20, 20), font=_load_font())
    return np.array(canvas)


def _stack_labeled_row(images: list[np.ndarray], labels: list[str], gutter: int = 8) -> np.ndarray:
    tiles = [_labeled_tile(img, label) for img, label in zip(images, labels, strict=False)]
    height = max(tile.shape[0] for tile in tiles)
    width = sum(tile.shape[1] for tile in tiles) + gutter * (len(tiles) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    offset = 0
    for tile in tiles:
        canvas[: tile.shape[0], offset : offset + tile.shape[1]] = tile
        offset += tile.shape[1] + gutter
    return canvas


def _stack_rows(rows: list[np.ndarray], gutter: int = 12, background: int = 245) -> np.ndarray:
    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + gutter * (len(rows) - 1)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    offset = 0
    for row in rows:
        canvas[offset : offset + row.shape[0], : row.shape[1]] = row
        offset += row.shape[0] + gutter
    return canvas


def _annotate_patch_map(base_image: np.ndarray, centers: list[tuple[int, int]], patch_size: int) -> np.ndarray:
    image = Image.fromarray(_rgb_image_to_uint8(base_image), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    colors = [
        (220, 20, 60),
        (30, 144, 255),
        (50, 205, 50),
        (255, 140, 0),
        (138, 43, 226),
    ]
    for index, center in enumerate(centers, start=1):
        color = colors[(index - 1) % len(colors)]
        y_slice, x_slice = _pad_or_crop(center, patch_size=patch_size, shape=(image.height, image.width))
        draw.rectangle((x_slice.start, y_slice.start, x_slice.stop, y_slice.stop), outline=color, width=3)
        draw.rectangle((x_slice.start + 4, y_slice.start + 4, x_slice.start + 22, y_slice.start + 18), fill=(255, 255, 255))
        draw.text((x_slice.start + 7, y_slice.start + 5), str(index), fill=color, font=font)
    return np.array(image)


def _make_zoom_contact_sheet(
    he_rgb: np.ndarray,
    oct_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    centers: list[tuple[int, int]],
    patch_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    patch_map = _annotate_patch_map(overlay_rgb, centers, patch_size=patch_size)
    rows = [_stack_labeled_row([patch_map], ["Patch map"])]
    patch_tiles: list[np.ndarray] = []
    for index, center in enumerate(centers, start=1):
        y_slice, x_slice = _pad_or_crop(center, patch_size=patch_size, shape=he_rgb.shape[:2])
        he_patch = _rgb_image_to_uint8(he_rgb[y_slice, x_slice])
        oct_patch = _rgb_image_to_uint8(oct_rgb[y_slice, x_slice])
        overlay_patch = _rgb_image_to_uint8(overlay_rgb[y_slice, x_slice])
        triptych = _make_zoom_montage(he_patch, oct_patch, overlay_patch, scale_factor=5)
        patch_tiles.append(triptych)
        rows.append(_stack_labeled_row([triptych], [f"Patch {index}: HE | OCT | overlay"]))
    return _stack_rows(rows), patch_tiles


def _make_overview_contact_sheet(
    he_rgb: np.ndarray,
    oct_rgb: np.ndarray,
    false_color: np.ndarray,
    contour_overlay: np.ndarray,
    checkerboard: np.ndarray,
    overlap_mask: np.ndarray,
) -> np.ndarray:
    overlap_rgb = np.stack([_float_image_to_uint8(overlap_mask.astype(np.float32))] * 3, axis=-1)
    row1 = _stack_labeled_row(
        [he_rgb, oct_rgb, false_color],
        ["HE", "OCT", "False-color overlay"],
    )
    row2 = _stack_labeled_row(
        [contour_overlay, checkerboard, overlap_rgb],
        ["Contour overlay", "Checkerboard", "Overlap mask"],
    )
    return _stack_rows([row1, row2])


def run_registration(
    sample_dir: str | Path,
    he_target_size: int = 256,
    oct_coarse_target_xy: int = 340,
    oct_plane_shape: int = 192,
    qc_target_size: int = 768,
    aligned_shape: int = 512,
) -> dict[str, Any]:
    paths = find_sample_paths(sample_dir)
    he_rgb_full, _, _ = load_he_first_frame(paths.he_path)

    he_search = preprocess_he(he_rgb_full, target_size=he_target_size)
    oct_volume = load_oct_memmap(paths.oct_path)
    coarse_volume, stride = build_oct_coarse_volume(oct_volume, target_xy=oct_coarse_target_xy)
    center_y, center_x = estimate_oct_center(coarse_volume)
    best_pose, top_pose_candidates = search_best_pose(
        coarse_volume,
        he_search,
        center_y=center_y,
        center_x=center_x,
        output_shape=(oct_plane_shape, oct_plane_shape),
    )

    he_qc = preprocess_he(he_rgb_full, target_size=qc_target_size)
    full_center_y = center_y * stride
    full_center_x = center_x * stride
    final_plane_raw = extract_oct_slab(
        oct_volume,
        z_index=best_pose.z_index,
        center_y=full_center_y,
        center_x=full_center_x,
        rx_deg=best_pose.rx_deg,
        ry_deg=best_pose.ry_deg,
        rz_deg=best_pose.rz_deg,
        output_shape=(qc_target_size, qc_target_size),
        slab_offsets=(-2.0, -1.0, 0.0, 1.0, 2.0),
    )
    oct_qc = preprocess_oct_plane(final_plane_raw)

    he_aligned_rgb, he_aligned_mask = crop_with_mask(he_qc.rgb, he_qc.mask, output_shape=(aligned_shape, aligned_shape), order=1)
    he_aligned_hematoxylin, _ = crop_with_mask(he_qc.hematoxylin, he_qc.mask, output_shape=(aligned_shape, aligned_shape), order=1)
    he_aligned_tissue_score, _ = crop_with_mask(he_qc.tissue_score, he_qc.mask, output_shape=(aligned_shape, aligned_shape), order=1)
    he_aligned_feature, _ = crop_with_mask(he_qc.feature, he_qc.mask, output_shape=(aligned_shape, aligned_shape), order=1)
    he_aligned_landmark, _ = crop_with_mask(he_qc.landmark, he_qc.mask, output_shape=(aligned_shape, aligned_shape), order=1)
    he_aligned_descriptor, _ = crop_with_mask(he_qc.descriptor, he_qc.mask, output_shape=(aligned_shape, aligned_shape), order=1)
    he_aligned_edges = feature.canny(he_aligned_feature, sigma=1.2)

    oct_aligned_raw = resize_full_frame(oct_qc.raw, output_shape=(aligned_shape, aligned_shape), order=1)
    oct_aligned_normalized = resize_full_frame(oct_qc.normalized, output_shape=(aligned_shape, aligned_shape), order=1)
    oct_aligned_feature = resize_full_frame(oct_qc.feature, output_shape=(aligned_shape, aligned_shape), order=1)
    oct_aligned_landmark = resize_full_frame(oct_qc.landmark, output_shape=(aligned_shape, aligned_shape), order=1)
    oct_aligned_descriptor = resize_full_frame(oct_qc.descriptor, output_shape=(aligned_shape, aligned_shape), order=1)
    oct_aligned_mask = resize_full_frame(oct_qc.mask.astype(np.float32), output_shape=(aligned_shape, aligned_shape), order=0) > 0.5

    inplane = refine_inplane_similarity(
        he_aligned_feature,
        he_aligned_mask,
        he_aligned_landmark,
        he_aligned_descriptor,
        oct_aligned_feature,
        oct_aligned_mask,
        oct_aligned_landmark,
        oct_aligned_descriptor,
    )

    oct_coreg_raw = warp_similarity(oct_aligned_raw, inplane.scale, inplane.rotation_deg, inplane.translation_y, inplane.translation_x, order=1)
    oct_coreg_normalized = warp_similarity(oct_aligned_normalized, inplane.scale, inplane.rotation_deg, inplane.translation_y, inplane.translation_x, order=1)
    oct_coreg_feature = warp_similarity(oct_aligned_feature, inplane.scale, inplane.rotation_deg, inplane.translation_y, inplane.translation_x, order=1)
    oct_coreg_landmark = warp_similarity(oct_aligned_landmark, inplane.scale, inplane.rotation_deg, inplane.translation_y, inplane.translation_x, order=1)
    oct_coreg_descriptor = warp_similarity(oct_aligned_descriptor, inplane.scale, inplane.rotation_deg, inplane.translation_y, inplane.translation_x, order=1)
    oct_coreg_mask = warp_similarity(oct_aligned_mask.astype(np.float32), inplane.scale, inplane.rotation_deg, inplane.translation_y, inplane.translation_x, order=0) > 0.5
    oct_coreg_edges = feature.canny(oct_coreg_feature, sigma=1.2)

    overlap_mask = he_aligned_mask & oct_coreg_mask
    union_mask = he_aligned_mask | oct_coreg_mask
    overlay_false_color = _false_color_overlay(he_aligned_rgb, oct_coreg_normalized, oct_coreg_mask)
    overlay_contours = _contour_overlay(he_aligned_rgb, he_aligned_mask, oct_coreg_mask)
    checkerboard = _make_checkerboard(_rgb_image_to_uint8(he_aligned_rgb), _oct_gray_to_rgb(oct_coreg_normalized), tile=32)
    overview_contact_sheet = _make_overview_contact_sheet(
        he_aligned_rgb,
        _oct_gray_to_rgb(oct_coreg_normalized),
        overlay_false_color,
        overlay_contours,
        checkerboard,
        overlap_mask,
    )

    zoom_source_mask = overlap_mask if overlap_mask.any() else union_mask
    zoom_centers = _select_zoom_centers(zoom_source_mask)
    zoom_patch_size = max(128, aligned_shape // 4)
    zoom_contact_sheet, zoom_triptychs = _make_zoom_contact_sheet(
        he_aligned_rgb,
        _oct_gray_to_rgb(oct_coreg_normalized),
        overlay_contours,
        zoom_centers,
        patch_size=zoom_patch_size,
    )

    final_score = _score_registered_alignment(
        he_aligned_mask,
        he_aligned_edges,
        he_aligned_feature,
        he_aligned_landmark,
        he_aligned_descriptor,
        oct_coreg_mask,
        oct_coreg_edges,
        oct_coreg_feature,
        oct_coreg_landmark,
        oct_coreg_descriptor,
    )

    top_candidates_qc: list[dict[str, Any]] = []
    for index, candidate in enumerate(top_pose_candidates, start=1):
        candidate_slab = extract_oct_slab(
            oct_volume,
            z_index=candidate.z_index,
            center_y=full_center_y,
            center_x=full_center_x,
            rx_deg=candidate.rx_deg,
            ry_deg=candidate.ry_deg,
            rz_deg=candidate.rz_deg,
            output_shape=(qc_target_size, qc_target_size),
            slab_offsets=(-2.0, -1.0, 0.0, 1.0, 2.0),
        )
        candidate_oct = preprocess_oct_plane(candidate_slab)
        candidate_norm = resize_full_frame(candidate_oct.normalized, output_shape=(aligned_shape, aligned_shape), order=1)
        candidate_mask = resize_full_frame(candidate_oct.mask.astype(np.float32), output_shape=(aligned_shape, aligned_shape), order=0) > 0.5
        candidate_overlay = _contour_overlay(he_aligned_rgb, he_aligned_mask, candidate_mask)
        top_candidates_qc.append(
            {
                "rank": index,
                "pose": asdict(candidate),
                "normalized": candidate_norm,
                "dark_projection": resize_full_frame(candidate_oct.dark_projection, output_shape=(aligned_shape, aligned_shape), order=1),
                "mask": candidate_mask,
                "overlay": candidate_overlay,
            }
        )

    return {
        "sample_dir": str(paths.sample_dir),
        "he_path": str(paths.he_path),
        "oct_path": str(paths.oct_path),
        "oct_stride_xy": int(stride),
        "oct_center_yx_coarse": [float(center_y), float(center_x)],
        "oct_center_yx_fullres": [float(full_center_y), float(full_center_x)],
        "best_pose": asdict(best_pose),
        "top_pose_candidates": [asdict(candidate) for candidate in top_pose_candidates],
        "inplane_similarity": asdict(inplane),
        "final_alignment_score": float(final_score),
        "he_preprocess": {
            "rgb": he_qc.rgb,
            "hematoxylin": he_qc.hematoxylin,
            "tissue_score": he_qc.tissue_score,
            "feature": he_qc.feature,
            "landmark": he_qc.landmark,
            "mask": he_qc.mask,
            "edges": he_qc.edges,
        },
        "oct_preprocess": {
            "raw_plane": oct_qc.raw,
            "normalized": oct_qc.normalized,
            "dark_projection": oct_qc.dark_projection,
            "feature": oct_qc.feature,
            "landmark": oct_qc.landmark,
            "mask": oct_qc.mask,
            "edges": oct_qc.edges,
        },
        "coregistered": {
            "he_rgb": he_aligned_rgb,
            "he_hematoxylin": he_aligned_hematoxylin,
            "he_tissue_score": he_aligned_tissue_score,
            "he_feature": he_aligned_feature,
            "he_landmark": he_aligned_landmark,
            "he_mask": he_aligned_mask,
            "he_edges": he_aligned_edges,
            "oct_raw": oct_coreg_raw,
            "oct_normalized": oct_coreg_normalized,
            "oct_feature": oct_coreg_feature,
            "oct_landmark": oct_coreg_landmark,
            "oct_mask": oct_coreg_mask,
            "oct_edges": oct_coreg_edges,
            "overlap_mask": overlap_mask,
            "union_mask": union_mask,
            "overlay_false_color": overlay_false_color,
            "overlay_contours": overlay_contours,
            "checkerboard": checkerboard,
            "overview_contact_sheet": overview_contact_sheet,
            "zoom_contact_sheet": zoom_contact_sheet,
            "zoom_centers": zoom_centers,
            "zoom_patch_size": int(zoom_patch_size),
            "zoom_triptychs": zoom_triptychs,
        },
        "top_candidates_qc": top_candidates_qc,
    }


def save_registration_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    summary_dir = output_path / "00_summary"
    he_dir = output_path / "01_he_preprocess"
    oct_dir = output_path / "02_oct_preprocess"
    coreg_dir = output_path / "03_coregistered_images"
    review_dir = output_path / "04_qc_review"
    candidates_dir = output_path / "05_top_oct_candidates"
    for folder in [summary_dir, he_dir, oct_dir, coreg_dir, review_dir, candidates_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    he = result["he_preprocess"]
    oct_data = result["oct_preprocess"]
    coreg = result["coregistered"]

    _save_rgb_png(he_dir / "he_downsampled_rgb.png", he["rgb"])
    _save_gray_png(he_dir / "he_hematoxylin.png", he["hematoxylin"])
    _save_gray_png(he_dir / "he_tissue_score.png", he["tissue_score"])
    _save_gray_png(he_dir / "he_feature.png", he["feature"])
    _save_gray_png(he_dir / "he_landmark.png", he["landmark"])
    _save_gray_png(he_dir / "he_mask.png", he["mask"].astype(np.float32))
    _save_gray_png(he_dir / "he_edges.png", he["edges"].astype(np.float32))

    _save_gray_png(oct_dir / "oct_plane_raw.png", oct_data["raw_plane"])
    _save_gray_png(oct_dir / "oct_plane_normalized.png", oct_data["normalized"])
    _save_gray_png(oct_dir / "oct_plane_dark_projection.png", oct_data["dark_projection"])
    _save_gray_png(oct_dir / "oct_plane_feature.png", oct_data["feature"])
    _save_gray_png(oct_dir / "oct_plane_landmark.png", oct_data["landmark"])
    _save_gray_png(oct_dir / "oct_plane_mask.png", oct_data["mask"].astype(np.float32))
    _save_gray_png(oct_dir / "oct_plane_edges.png", oct_data["edges"].astype(np.float32))

    _save_rgb_png(coreg_dir / "he_coregistered_rgb.png", coreg["he_rgb"])
    _save_gray_png(coreg_dir / "oct_coregistered_raw.png", coreg["oct_raw"])
    _save_gray_png(coreg_dir / "oct_coregistered_normalized.png", coreg["oct_normalized"])
    _save_rgb_png(coreg_dir / "overlay_false_color.png", coreg["overlay_false_color"])
    _save_rgb_png(coreg_dir / "overlay_contours.png", coreg["overlay_contours"])
    _save_rgb_png(coreg_dir / "overlay_checkerboard.png", coreg["checkerboard"])

    _save_gray_png(review_dir / "he_coregistered_feature.png", coreg["he_feature"])
    _save_gray_png(review_dir / "oct_coregistered_feature.png", coreg["oct_feature"])
    _save_gray_png(review_dir / "he_coregistered_mask.png", coreg["he_mask"].astype(np.float32))
    _save_gray_png(review_dir / "oct_coregistered_mask.png", coreg["oct_mask"].astype(np.float32))
    _save_gray_png(review_dir / "overlap_mask.png", coreg["overlap_mask"].astype(np.float32))
    _save_rgb_png(review_dir / "overview_contact_sheet.png", coreg["overview_contact_sheet"])
    _save_rgb_png(review_dir / "zoom_contact_sheet.png", coreg["zoom_contact_sheet"])
    for index, patch in enumerate(coreg["zoom_triptychs"], start=1):
        _save_rgb_png(review_dir / f"zoom_patch_{index:02d}.png", patch)

    np.savez_compressed(
        summary_dir / "coregistered_arrays.npz",
        he_rgb=_rgb_image_to_uint8(coreg["he_rgb"]),
        he_feature=np.asarray(coreg["he_feature"], dtype=np.float32),
        he_mask=np.asarray(coreg["he_mask"], dtype=bool),
        oct_raw=np.asarray(coreg["oct_raw"], dtype=np.float32),
        oct_normalized=np.asarray(coreg["oct_normalized"], dtype=np.float32),
        oct_feature=np.asarray(coreg["oct_feature"], dtype=np.float32),
        oct_mask=np.asarray(coreg["oct_mask"], dtype=bool),
        overlap_mask=np.asarray(coreg["overlap_mask"], dtype=bool),
    )

    top_pose_summary = []
    for candidate in result.get("top_candidates_qc", []):
        rank = int(candidate["rank"])
        pose = candidate["pose"]
        _save_gray_png(candidates_dir / f"candidate_{rank:02d}_normalized.png", candidate["normalized"])
        _save_gray_png(candidates_dir / f"candidate_{rank:02d}_dark_projection.png", candidate["dark_projection"])
        _save_gray_png(candidates_dir / f"candidate_{rank:02d}_mask.png", candidate["mask"].astype(np.float32))
        _save_rgb_png(candidates_dir / f"candidate_{rank:02d}_overlay.png", candidate["overlay"])
        top_pose_summary.append({"rank": rank, **pose})

    (summary_dir / "top_pose_candidates.json").write_text(json.dumps(top_pose_summary, indent=2))

    payload = {
        "sample_dir": result["sample_dir"],
        "he_path": result["he_path"],
        "oct_path": result["oct_path"],
        "oct_stride_xy": result["oct_stride_xy"],
        "oct_center_yx_coarse": result["oct_center_yx_coarse"],
        "oct_center_yx_fullres": result["oct_center_yx_fullres"],
        "best_pose": result["best_pose"],
        "top_pose_candidates": result.get("top_pose_candidates", []),
        "inplane_similarity": result["inplane_similarity"],
        "final_alignment_score": result["final_alignment_score"],
        "output_structure": {
            "summary": str(summary_dir),
            "he_preprocess": str(he_dir),
            "oct_preprocess": str(oct_dir),
            "coregistered_images": str(coreg_dir),
            "qc_review": str(review_dir),
            "top_oct_candidates": str(candidates_dir),
        },
    }
    (summary_dir / "registration_result.json").write_text(json.dumps(payload, indent=2))


def exploration_report_to_json(report: ExplorationReport) -> str:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, tuple):
            return [sanitize(v) for v in value]
        if isinstance(value, bytes):
            return f"<bytes:{len(value)}>"
        if isinstance(value, np.generic):
            return value.item()
        return value

    payload = sanitize(asdict(report))
    payload["he_series"] = [asdict(series) for series in report.he_series]
    payload["oct_series"] = [asdict(series) for series in report.oct_series]
    return json.dumps(payload, indent=2)
