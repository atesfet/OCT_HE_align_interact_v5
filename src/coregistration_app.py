from __future__ import annotations

import base64
import cgi
import json
import math
import mimetypes
import os
import shutil
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage, optimize
from skimage import exposure, filters, transform

sys.path.append(str(Path(__file__).resolve().parent))

from align_he_to_oct_2D_v5 import (  # noqa: E402
    _affine,
    _affine_with_tilt,
    _boundary,
    _clean_mask,
    _distance,
    _ensure_rgb,
    _gray_he_mask,
    _he_maps,
    _mask_center,
    _native_matrix,
    _orientation,
    _oct_maps,
    _oct_to_gray,
    _read_tiff,
    _resize_gray,
    _resize_mask,
    _resize_rgb,
    _score,
    _search_shape,
    _to_uint8,
    _warp,
    _warp_maps,
    preprocess_he_rgb,
    preprocess_oct_2d,
    stain_standardize_he,
)


Image.MAX_IMAGE_PIXELS = None

APP_ROOT = Path(
    os.environ.get(
        "OCTHE_APP_OUTPUT_DIR",
        Path(__file__).resolve().parent.parent / "coregistration_outputs" / "interactive_app",
    )
)
MAX_SEARCH_DIM = 420


@dataclass
class SessionPaths:
    root: Path
    oct_path: Path
    he_path: Path


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def _session(session_id: str) -> SessionPaths:
    root = APP_ROOT / session_id
    manifest = root / "session.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Unknown session: {session_id}")
    data = json.loads(manifest.read_text())
    return SessionPaths(root=root, oct_path=Path(data["oct_path"]), he_path=Path(data["he_path"]))


def _write_session(root: Path, oct_path: Path, he_path: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "session.json").write_text(json.dumps({"oct_path": str(oct_path), "he_path": str(he_path)}, indent=2))


def _clean_stem(path: Path, suffixes: tuple[str, ...]) -> str:
    name = path.name
    for suffix in path.suffixes:
        name = name[: -len(suffix)]
    lowered = name.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            lowered = name.lower()
    return name or path.stem or "sample"


def _slug(value: str) -> str:
    cleaned = []
    last_was_sep = False
    for char in value.strip():
        if char.isalnum():
            cleaned.append(char)
            last_was_sep = False
        elif not last_was_sep:
            cleaned.append("_")
            last_was_sep = True
    return "".join(cleaned).strip("_") or "sample"


def _session_id_from_paths(oct_path: Path, he_path: Path) -> str:
    oct_name = _slug(_clean_stem(oct_path, ("_oct", "-oct", " oct")))
    he_name = _slug(_clean_stem(he_path, ("_he", "-he", " he", "_section", "-section", " section")))
    if oct_name and he_name and oct_name != he_name:
        if he_name.startswith(oct_name):
            return he_name
        return f"{oct_name}__{he_name}"
    return oct_name or he_name or "sample"


def _unique_session_root(session_id: str) -> tuple[str, Path]:
    candidate = _slug(session_id)
    root = APP_ROOT / candidate
    if not root.exists():
        return candidate, root
    for index in range(2, 10000):
        indexed = f"{candidate}_{index:02d}"
        root = APP_ROOT / indexed
        if not root.exists():
            return indexed, root
    raise RuntimeError(f"Could not create a unique output folder for {candidate}")


def _save_gray_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(_to_uint8(image)).save(path)


def _save_rgb_png(path: Path, image: np.ndarray) -> None:
    arr = _to_uint8(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    Image.fromarray(arr[..., :3]).save(path)


def _save_rgb_tiff(path: Path, image: np.ndarray) -> None:
    arr = _to_uint8(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    tifffile.imwrite(str(path), arr[..., :3], photometric="rgb")


def _save_float_tiff(path: Path, image: np.ndarray) -> None:
    tifffile.imwrite(str(path), np.asarray(image, dtype=np.float32))


def _save_mask_tiff(path: Path, mask: np.ndarray) -> None:
    tifffile.imwrite(str(path), mask.astype(bool).astype(np.uint8) * 255)


def _sync_clean_outputs(paths: SessionPaths) -> None:
    output_dir = paths.root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["he_registered.tiff", "oct_registered.tiff", "registered_mask.tiff"]:
        src = paths.root / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _preview_image(image: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    shape = image.shape[:2]
    if max(shape) <= max_dim:
        return image
    scale = max_dim / float(max(shape))
    out_shape = (max(1, int(round(shape[0] * scale))), max(1, int(round(shape[1] * scale))))
    if image.ndim == 2:
        return _resize_gray(image, out_shape)
    return _resize_rgb(image, out_shape)


def _mask_overlay(base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float] = (1.0, 0.12, 0.05)) -> np.ndarray:
    if base.ndim == 2:
        rgb = np.repeat(_to_uint8(base)[..., None], 3, axis=2).astype(np.float32) / 255.0
    else:
        rgb = _to_uint8(base).astype(np.float32) / 255.0
    mask = mask.astype(bool)
    if mask.shape != rgb.shape[:2]:
        mask = _resize_mask(mask, rgb.shape[:2])
    overlay = rgb.copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def _load_mask_png(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return arr > 127


def _mask_maps_from_feature(mask: np.ndarray, feature: np.ndarray) -> dict[str, np.ndarray]:
    mask = mask.astype(bool)
    feature = np.asarray(feature, dtype=np.float32) * mask.astype(np.float32)
    return {
        "mask": mask,
        "boundary": _boundary(mask),
        "distance": _distance(mask),
        "feature": exposure.rescale_intensity(feature, out_range=(0.0, 1.0)).astype(np.float32),
    }


def _prepare_state(paths: SessionPaths, stain_normalizer: str, stain_reference_he: str | None = None) -> dict[str, Any]:
    oct_raw = _oct_to_gray(_read_tiff(paths.oct_path))
    he_raw = _ensure_rgb(_read_tiff(paths.he_path))
    oct_pre = preprocess_oct_2d(oct_raw)
    he_pre = preprocess_he_rgb(he_raw)
    ref = _ensure_rgb(_read_tiff(Path(stain_reference_he))) if stain_reference_he else he_raw
    he_standardized = stain_standardize_he(he_raw, method=stain_normalizer, reference_rgb=ref, mask=None)

    _save_float_tiff(paths.root / "oct_registered.tiff", oct_pre["contrast"])
    _save_gray_png(paths.root / "oct_raw_display_preview.png", _preview_image(oct_pre["raw_rescaled"]))
    _save_gray_png(paths.root / "oct_flatfield_corrected_preview.png", _preview_image(oct_pre["flatfield"]))
    _save_gray_png(paths.root / "oct_tile_artifact_suppressed_preview.png", _preview_image(oct_pre["destriped"]))
    _save_gray_png(paths.root / "oct_contrast_adjusted_preview.png", _preview_image(oct_pre["contrast"]))
    _save_gray_png(paths.root / "oct_registered_preview.png", _preview_image(oct_pre["contrast"]))
    _save_rgb_tiff(paths.root / "he_standardized_native.tiff", he_standardized)
    _save_rgb_png(paths.root / "he_standardized_native_preview.png", _preview_image(he_standardized))
    _save_gray_png(paths.root / "he_black_white_input_preview.png", _preview_image(he_pre["inverted_gray"]))

    state = {
        "oct_shape": list(oct_raw.shape[:2]),
        "he_shape": list(he_raw.shape[:2]),
        "stain_normalizer": stain_normalizer,
        "stain_reference_he": stain_reference_he,
    }
    (paths.root / "state.json").write_text(json.dumps(state, indent=2))
    return state


def _compute_masks(paths: SessionPaths, he_mask_mode: str, he_gray_percentile: float, he_alpha: float, oct_alpha: float) -> dict[str, Any]:
    oct_raw = _oct_to_gray(_read_tiff(paths.oct_path))
    he_raw = _ensure_rgb(_read_tiff(paths.he_path))
    oct_shape = tuple(int(v) for v in oct_raw.shape[:2])
    search_shape = _search_shape(oct_shape, MAX_SEARCH_DIM)
    he_pre = preprocess_he_rgb(he_raw)

    # Registration masks/features intentionally follow align_he_to_oct_2D_v3.
    # v5 preprocessing is saved/displayed separately and applied after the v3
    # transform is estimated.
    oct_search = _resize_gray(oct_raw.astype(np.float32), search_shape)
    he_search_rgb = _resize_rgb(he_raw.astype(np.float32) / (255.0 if he_raw.max() > 1.5 else 1.0), search_shape)
    he_search_gray = _resize_gray(he_pre["gray"], search_shape)
    he_search_rembg = _resize_rgb(he_pre["rembg_input"], search_shape)

    oct_maps = _oct_maps(oct_search, oct_alpha)
    he_maps = _he_maps(
        he_search_rgb,
        he_search_gray,
        he_search_rembg,
        he_alpha,
        mask_mode=he_mask_mode,
        gray_mask_percentile=he_gray_percentile,
    )
    _save_gray_png(paths.root / "oct_mask_edit.png", oct_maps["mask"].astype(np.float32))
    _save_gray_png(paths.root / "he_mask_edit.png", he_maps["mask"].astype(np.float32))
    _save_gray_png(paths.root / "oct_search_feature.png", oct_maps.get("canonical_feature", oct_maps.get("feature", oct_search)))
    he_search_bw = _resize_gray(he_pre["inverted_gray"], search_shape)
    _save_gray_png(paths.root / "he_search_bw.png", he_search_bw)
    _save_rgb_png(paths.root / "oct_mask_overlay.png", _mask_overlay(oct_maps.get("base_intensity", oct_search), oct_maps["mask"], color=(0.0, 1.0, 0.28)))
    _save_rgb_png(paths.root / "he_mask_overlay.png", _mask_overlay(he_search_bw, he_maps["mask"], color=(1.0, 0.12, 0.05)))
    mask_state = {
        "search_shape": list(search_shape),
        "he_mask_mode": he_mask_mode,
        "he_gray_percentile": he_gray_percentile,
        "he_mask_fraction": float(he_maps["mask"].mean()),
        "oct_mask_fraction": float(oct_maps["mask"].mean()),
    }
    (paths.root / "mask_state.json").write_text(json.dumps(mask_state, indent=2))
    return mask_state


def _build_search_maps(paths: SessionPaths) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[int, int], np.ndarray, np.ndarray]:
    oct_raw = _oct_to_gray(_read_tiff(paths.oct_path))
    he_raw = _ensure_rgb(_read_tiff(paths.he_path))
    search_shape = tuple(json.loads((paths.root / "mask_state.json").read_text())["search_shape"])
    oct_pre = preprocess_oct_2d(oct_raw)
    he_pre = preprocess_he_rgb(he_raw)
    oct_search = _resize_gray(oct_raw.astype(np.float32), search_shape)
    he_rgb = _resize_rgb(he_raw.astype(np.float32) / (255.0 if he_raw.max() > 1.5 else 1.0), search_shape)
    he_gray = _resize_gray(he_pre["gray"], search_shape)
    he_rembg = _resize_rgb(he_pre["rembg_input"], search_shape)

    oct_mask = _load_mask_png(paths.root / "oct_mask_edit.png")
    he_mask = _load_mask_png(paths.root / "he_mask_edit.png")
    if oct_mask.shape != search_shape:
        oct_mask = _resize_mask(oct_mask, search_shape)
    if he_mask.shape != search_shape:
        he_mask = _resize_mask(he_mask, search_shape)

    oct_maps = _oct_maps(oct_search, 0.376)
    he_maps = _he_maps(he_rgb, he_gray, he_rembg, 0.376)
    oct_maps = dict(oct_maps)
    he_maps = dict(he_maps)
    oct_maps["mask"] = oct_mask.astype(bool)
    he_maps["mask"] = he_mask.astype(bool)
    oct_maps["boundary"] = _boundary(oct_maps["mask"])
    he_maps["boundary"] = _boundary(he_maps["mask"])
    oct_maps["distance"] = _distance(oct_maps["mask"])
    he_maps["distance"] = _distance(he_maps["mask"])
    oct_maps.pop("boundary_distance", None)
    he_maps.pop("boundary_distance", None)
    return oct_maps, he_maps, search_shape, he_raw, oct_pre["contrast"]


def _estimate_registration(paths: SessionPaths) -> dict[str, Any]:
    oct_maps, he_maps, search_shape, he_raw, oct_registered = _build_search_maps(paths)
    oct_center = _mask_center(oct_maps["mask"])
    he_center = _mask_center(he_maps["mask"])
    area_scale = math.sqrt(max(1.0, oct_maps["mask"].sum()) / max(1.0, he_maps["mask"].sum()))
    init_rotation = float(np.clip(_orientation(oct_maps["mask"]) - _orientation(he_maps["mask"]), -60.0, 60.0))
    seeds: list[np.ndarray] = []
    for scale_mult in [0.78, 0.92, 1.0, 1.10, 1.24]:
        for rot_delta in [-25.0, -10.0, 0.0, 10.0, 25.0]:
            for tilt_x, tilt_y in [(0.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (0.0, -3.0), (0.0, 3.0)]:
                seeds.append(np.array([area_scale * scale_mult, init_rotation + rot_delta, tilt_x, tilt_y, 0.0, 0.0], dtype=np.float64))

    ranked: list[tuple[float, np.ndarray]] = []
    for seed in seeds:
        matrix = _affine_with_tilt(seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], he_center, oct_center)
        moved = _warp_maps(he_maps, matrix, search_shape)
        score, _ = _score(oct_maps, moved)
        ranked.append((score, seed))
    ranked.sort(key=lambda item: item[0], reverse=True)

    best_params: np.ndarray | None = None
    best_score = -np.inf
    best_details: dict[str, float] = {}

    def objective(params: np.ndarray) -> float:
        scale, rotation, tilt_x, tilt_y, ty, tx = [float(v) for v in params]
        if not (0.25 <= scale <= 2.5) or abs(rotation) > 180:
            return 1.0
        if abs(tilt_x) > 3.0 or abs(tilt_y) > 3.0:
            return 1.0
        if abs(ty) > search_shape[0] * 0.40 or abs(tx) > search_shape[1] * 0.40:
            return 1.0
        matrix = _affine_with_tilt(scale, rotation, tilt_x, tilt_y, ty, tx, he_center, oct_center)
        moved = _warp_maps(he_maps, matrix, search_shape)
        score, _ = _score(oct_maps, moved)
        return -score

    for _, seed in ranked[:6]:
        result = optimize.minimize(objective, seed, method="Powell", options={"maxiter": 36, "disp": False})
        params = result.x if result.success else seed
        matrix = _affine_with_tilt(params[0], params[1], params[2], params[3], params[4], params[5], he_center, oct_center)
        moved = _warp_maps(he_maps, matrix, search_shape)
        score, details = _score(oct_maps, moved)
        if score > best_score:
            best_score = float(score)
            best_params = np.asarray(params, dtype=np.float64)
            best_details = details
    if best_params is None:
        raise RuntimeError("Auto registration failed to produce a candidate")

    search_matrix = _affine_with_tilt(best_params[0], best_params[1], best_params[2], best_params[3], best_params[4], best_params[5], he_center, oct_center)
    native = _native_matrix(search_matrix, he_raw.shape[:2], tuple(oct_registered.shape[:2]), search_shape)
    transform_state = {
        "auto_params": {
            "scale": float(best_params[0]),
            "rotation_deg": float(best_params[1]),
            "tilt_x_deg": float(best_params[2]),
            "tilt_y_deg": float(best_params[3]),
            "translation_y": float(best_params[4]),
            "translation_x": float(best_params[5]),
            "score": best_score,
            "details": best_details,
        },
        "native_matrix": native.tolist(),
        "manual": {"scale": 1.0, "rotation_deg": 0.0, "translation_y": 0.0, "translation_x": 0.0},
    }
    (paths.root / "transform_state.json").write_text(json.dumps(transform_state, indent=2))
    _apply_current_transform(paths, transform_state)
    return transform_state


def _manual_matrix(shape: tuple[int, int], scale: float, rotation_deg: float, ty: float, tx: float) -> np.ndarray:
    center = np.array([shape[0] / 2.0, shape[1] / 2.0], dtype=np.float64)
    return _affine(scale, rotation_deg, ty, tx, center, center)


def _apply_current_transform(paths: SessionPaths, transform_state: dict[str, Any] | None = None) -> None:
    if transform_state is None:
        transform_state = json.loads((paths.root / "transform_state.json").read_text())
    native_matrix = np.asarray(transform_state["native_matrix"], dtype=np.float64)
    manual = transform_state.get("manual", {})
    oct_registered = tifffile.imread(str(paths.root / "oct_registered.tiff")).astype(np.float32)
    output_shape = tuple(int(v) for v in oct_registered.shape[:2])
    correction = _manual_matrix(
        output_shape,
        float(manual.get("scale", 1.0)),
        float(manual.get("rotation_deg", 0.0)),
        float(manual.get("translation_y", 0.0)),
        float(manual.get("translation_x", 0.0)),
    )
    matrix = correction @ native_matrix
    he_standardized = tifffile.imread(str(paths.root / "he_standardized_native.tiff"))
    he_mask = _resize_mask(_load_mask_png(paths.root / "he_mask_edit.png"), tuple(json.loads((paths.root / "mask_state.json").read_text())["search_shape"]))
    he_mask_native = _resize_mask(he_mask, he_standardized.shape[:2])
    oct_mask = _resize_mask(_load_mask_png(paths.root / "oct_mask_edit.png"), output_shape)
    warped_he = _warp(he_standardized.astype(np.float32), matrix, output_shape, order=1)
    warped_he_mask = _warp(he_mask_native.astype(np.float32), matrix, output_shape, order=0) > 0.5
    overlap = warped_he_mask & oct_mask
    _save_rgb_tiff(paths.root / "he_registered.tiff", warped_he)
    _save_float_tiff(paths.root / "oct_registered.tiff", oct_registered)
    _save_mask_tiff(paths.root / "registered_mask.tiff", overlap)
    _save_rgb_png(paths.root / "he_registered_preview.png", warped_he)
    _save_gray_png(paths.root / "registered_mask_preview.png", overlap.astype(np.float32))
    _save_rgb_png(paths.root / "he_registered_masked_preview.png", warped_he * overlap[..., None].astype(np.float32))
    _save_gray_png(paths.root / "oct_registered_masked_preview.png", oct_registered * overlap.astype(np.float32))
    _save_rgb_png(paths.root / "overlay_preview.png", _overlay_preview(warped_he, oct_registered, overlap))
    _sync_clean_outputs(paths)


def _overlay_preview(he_rgb: np.ndarray, oct_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    he = _to_uint8(he_rgb).astype(np.float32) / 255.0
    oct_img = _to_uint8(oct_gray).astype(np.float32) / 255.0
    out = he * 0.82
    out[..., 1] = np.maximum(out[..., 1], oct_img * 0.95)
    out[..., 2] = np.maximum(out[..., 2], oct_img * 0.35)
    out[~mask.astype(bool)] *= 0.82
    return np.clip(out, 0, 1)


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OCT/HE Interactive Coregistration</title>
  <style>
    :root {
      --ink:#16201d;
      --muted:#66736d;
      --paper:#f7f4ec;
      --panel:#fffdf7;
      --panel-soft:#f5f1e8;
      --accent:#0f6f5d;
      --accent-2:#d46b3d;
      --line:#d7d0c0;
      --shadow:0 24px 70px rgba(24,32,28,.13);
      --radius:22px;
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      font-family:Avenir Next, Avenir, Helvetica Neue, Helvetica, sans-serif;
      color:var(--ink);
      background:
        radial-gradient(circle at 8% 4%, rgba(212,107,61,.18), transparent 28rem),
        radial-gradient(circle at 94% 2%, rgba(15,111,93,.18), transparent 30rem),
        linear-gradient(145deg,#fbf8f0 0%,#edf3ee 48%,#f8f2e7 100%);
      min-height:100vh;
    }
    body:before {
      content:"";
      position:fixed;
      inset:0;
      pointer-events:none;
      opacity:.26;
      background-image:linear-gradient(rgba(22,32,29,.055) 1px, transparent 1px), linear-gradient(90deg, rgba(22,32,29,.045) 1px, transparent 1px);
      background-size:32px 32px;
      mask-image:linear-gradient(to bottom, black, transparent 85%);
    }
    header {
      position:relative;
      padding:34px clamp(20px,4vw,56px) 30px;
      color:#fbf6e9;
      background:
        linear-gradient(135deg,rgba(10,30,26,.98),rgba(19,67,57,.96)),
        radial-gradient(circle at 20% 20%, rgba(212,107,61,.35), transparent 24rem);
      overflow:hidden;
      border-bottom:1px solid rgba(255,255,255,.12);
    }
    header:after {
      content:"";
      position:absolute;
      right:-120px;
      top:-170px;
      width:420px;
      height:420px;
      border-radius:50%;
      border:1px solid rgba(255,255,255,.16);
      box-shadow:0 0 0 42px rgba(255,255,255,.035), 0 0 0 92px rgba(255,255,255,.025);
    }
    h1 { margin:10px 0 8px; font-size:clamp(32px,5vw,58px); line-height:.95; letter-spacing:-.055em; max-width:900px; }
    h2 { display:flex; align-items:center; gap:12px; margin:0 0 16px; font-size:22px; letter-spacing:-.025em; }
    h2:before { content:""; width:11px; height:28px; border-radius:999px; background:linear-gradient(180deg,var(--accent),var(--accent-2)); box-shadow:0 0 0 5px rgba(15,111,93,.08); }
    h3 { margin:22px 0 12px; color:#31413b; font-size:15px; letter-spacing:.08em; text-transform:uppercase; }
    main { position:relative; width:min(1560px,100%); margin:0 auto; padding:28px clamp(16px,3vw,42px) 72px; display:grid; gap:20px; }
    section {
      position:relative;
      background:linear-gradient(180deg,rgba(255,253,247,.92),rgba(249,246,238,.86));
      border:1px solid rgba(215,208,192,.9);
      border-radius:var(--radius);
      padding:22px;
      box-shadow:var(--shadow);
      backdrop-filter:blur(18px);
    }
    section:hover { border-color:rgba(15,111,93,.34); }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; align-items:end; }
    label { display:block; font-weight:800; margin:10px 0 7px; color:#263530; font-size:13px; letter-spacing:.025em; }
    input, select, button {
      font:inherit;
      border-radius:14px;
      border:1px solid var(--line);
      padding:11px 13px;
      min-height:43px;
    }
    input, select { background:#fffefa; color:var(--ink); box-shadow:inset 0 1px 0 rgba(255,255,255,.85); }
    input:focus, select:focus { outline:3px solid rgba(15,111,93,.16); border-color:var(--accent); }
    input[type=text] { width:100%; }
    input[type=range] { accent-color:var(--accent); min-height:auto; padding:0; border:0; background:transparent; }
    input[type=file] { width:100%; background:#fffefa; }
    button {
      background:linear-gradient(135deg,#10745f,#0b584c);
      color:white;
      cursor:pointer;
      border:0;
      margin:10px 10px 0 0;
      font-weight:800;
      letter-spacing:.01em;
      box-shadow:0 10px 24px rgba(15,111,93,.24);
      transition:transform .16s ease, box-shadow .16s ease, opacity .16s ease;
    }
    button:hover:not(:disabled) { transform:translateY(-1px); box-shadow:0 14px 30px rgba(15,111,93,.30); }
    button:active:not(:disabled) { transform:translateY(0); }
    button:disabled { opacity:.52; cursor:wait; transform:none; box-shadow:none; }
    button.secondary { background:linear-gradient(135deg,#69756f,#4e5954); box-shadow:0 10px 22px rgba(50,58,54,.16); }
    .busy {
      display:none;
      margin:14px 0 2px;
      padding:12px 14px;
      border-radius:16px;
      background:linear-gradient(135deg,#edf7f1,#fff8ee);
      border:1px solid #bed8cb;
      color:#21382f;
      font-weight:700;
    }
    .busy.active { display:flex; gap:12px; align-items:center; }
    progress { width:230px; height:11px; vertical-align:middle; accent-color:var(--accent); }
    .status {
      white-space:pre-wrap;
      background:linear-gradient(135deg,#0f211c,#172f28);
      color:#eaf8ee;
      border-radius:18px;
      padding:16px;
      min-height:58px;
      border:1px solid rgba(255,255,255,.08);
      box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
      font-family:Menlo, Monaco, Consolas, monospace;
      font-size:12px;
      line-height:1.5;
    }
    .viewer { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; align-items:start; }
    .viewer > div:not(.image-card) { background:rgba(255,255,255,.52); border:1px solid rgba(215,208,192,.8); border-radius:18px; padding:12px; }
    .viewer img {
      max-width:100%;
      max-height:560px;
      border-radius:16px;
      border:1px solid rgba(215,208,192,.95);
      background:#fff;
      box-shadow:0 12px 32px rgba(38,43,39,.12);
    }
    .image-card {
      min-width:0;
      background:rgba(255,255,255,.58);
      border:1px solid rgba(215,208,192,.85);
      border-radius:18px;
      padding:12px;
    }
    .image-card img { width:100%; max-height:330px; object-fit:contain; display:block; }
    .caption { font-weight:900; margin:2px 0 10px; color:#24342f; font-size:13px; letter-spacing:.02em; }
    canvas {
      width:100%;
      max-width:620px;
      max-height:620px;
      border:1px solid rgba(215,208,192,.95);
      border-radius:18px;
      image-rendering:auto;
      background:#111b18;
      box-shadow:0 16px 38px rgba(22,32,29,.18);
    }
    canvas.mask-editor { touch-action:none; cursor:crosshair; }
    .slider-row {
      display:grid;
      grid-template-columns:150px minmax(160px,1fr) 78px;
      gap:12px;
      align-items:center;
      margin:10px 0;
      padding:10px 12px;
      background:rgba(255,255,255,.56);
      border:1px solid rgba(215,208,192,.72);
      border-radius:16px;
    }
    .slider-row span { font-family:Menlo, Monaco, Consolas, monospace; color:#40504a; text-align:right; }
    .small { color:var(--muted); font-size:13px; line-height:1.45; margin-top:10px; }
    header .small { color:rgba(251,246,233,.76); }
    .app-meta { margin-top:14px; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:rgba(251,246,233,.62); }
    .small input, .small select { min-height:34px; padding:6px 9px; vertical-align:middle; }
    #saveLinks { margin-top:12px; display:grid; gap:8px; }
    #saveLinks a { color:var(--accent); font-weight:800; }
    @media (max-width:760px) {
      header { padding:26px 18px; }
      main { padding:18px 12px 48px; }
      section { padding:16px; border-radius:18px; }
      .grid { grid-template-columns:1fr; }
      .slider-row { grid-template-columns:1fr; gap:6px; }
      .slider-row span { text-align:left; }
      progress { width:160px; }
    }
  </style>
</head>
<body>
<header><div class="small">Interactive registration workspace</div><h1>OCT/HE Coregistration Studio</h1><div class="small">Preprocess | Mask and edit | Auto-register | Manually adjust | Save native-resolution outputs</div><div class="app-meta">Version 5 | Ates Fettahoglu</div></header>
<main>
  <section>
    <h2>1. Load Images</h2>
    <div class="grid">
      <div><label>OCT path</label><input id="octPath" type="text" placeholder="/path/to/sample_oct.tiff"></div>
      <div><label>HE path</label><input id="hePath" type="text" placeholder="/path/to/sample_section.tiff"></div>
    </div>
    <button onclick="loadPaths()">Load From Paths</button>
    <div class="grid">
      <div><label>Upload OCT</label><input id="octFile" type="file" accept=".tif,.tiff,image/tiff"></div>
      <div><label>Upload HE</label><input id="heFile" type="file" accept=".tif,.tiff,image/tiff"></div>
    </div>
    <button onclick="uploadFiles()">Upload Files</button>
    <div id="busy-load" class="busy"><progress></progress> Loading image references...</div>
    <div class="small">Large microscopy files are best loaded by path rather than browser upload.</div>
  </section>
  <section>
    <h2>2. Modality-Specific Preprocessing</h2>
    <label>HE stain/color standardization</label>
    <select id="stain"><option value="torchstain_reinhard">torchstain Reinhard</option><option value="torchstain_macenko">torchstain Macenko</option><option value="none">None</option></select>
    <button onclick="preprocess()">Preprocess Images</button>
    <div id="busy-preprocess" class="busy"><progress></progress> Preprocessing OCT and HE images...</div>
    <h3>OCT preprocessing</h3>
    <div class="viewer">
      <div class="image-card"><div class="caption">Raw display normalization</div><img id="octRaw"></div>
      <div class="image-card"><div class="caption">Flat-field corrected</div><img id="octFlat"></div>
      <div class="image-card"><div class="caption">Tile/artifact suppressed</div><img id="octTile"></div>
      <div class="image-card"><div class="caption">Contrast adjusted / oct_registered</div><img id="octPre"></div>
    </div>
    <h3>HE preprocessing</h3>
    <div class="viewer">
      <div class="image-card"><div class="caption">Stain/color standardized</div><img id="hePre"></div>
      <div class="image-card"><div class="caption">Black-white mask input</div><img id="heBW"></div>
    </div>
  </section>
  <section>
    <h2>3. Remove Background And Edit Masks</h2>
    <div class="grid">
      <div><label>HE mask mode</label><select id="heMaskMode"><option value="auto">auto</option><option value="gray">gray</option><option value="rembg">rembg</option></select></div>
      <div><label>HE gray percentile</label><input id="hePct" type="number" value="67" step="1"></div>
    </div>
    <button onclick="removeBackground()">Remove Background</button>
    <button class="secondary" onclick="saveMasks()">Save Edited Masks</button>
    <div id="busy-mask" class="busy"><progress></progress> Removing background and generating editable masks...</div>
    <div class="small">Draw white to add tissue, black to erase. Brush: <input id="brush" type="range" min="2" max="40" value="12"> <select id="paint"><option value="white">add</option><option value="black">erase</option></select></div>
    <h3>Mask overlays</h3>
    <div class="viewer">
      <div class="image-card"><div class="caption">OCT mask overlay</div><img id="octMaskOverlay"></div>
      <div class="image-card"><div class="caption">HE mask overlay</div><img id="heMaskOverlay"></div>
    </div>
    <h3>Edit masks directly on overlays</h3>
    <button class="secondary" onclick="undoEdit('octCanvas')">Undo OCT Edit</button>
    <button class="secondary" onclick="undoEdit('heCanvas')">Undo HE Edit</button>
    <div class="viewer"><div><b>OCT overlay editor</b><br><canvas id="octCanvas" class="mask-editor"></canvas></div><div><b>HE overlay editor</b><br><canvas id="heCanvas" class="mask-editor"></canvas></div></div>
  </section>
  <section>
    <h2>4. Auto Registration And Manual Adjustment</h2>
    <button onclick="autoRegister()">Run Auto Registration</button>
    <div id="busy-autoreg" class="busy"><progress></progress> Running auto-registration and applying native-resolution transform...</div>
    <div class="slider-row"><label>Scale</label><input id="mScale" type="range" min="0.70" max="1.30" value="1" step="0.002" oninput="manualAdjust()"><span id="mScaleV">1</span></div>
    <div class="slider-row"><label>Rotation</label><input id="mRot" type="range" min="-30" max="30" value="0" step="0.1" oninput="manualAdjust()"><span id="mRotV">0</span></div>
    <div class="slider-row"><label>Translate Y</label><input id="mTy" type="range" min="-400" max="400" value="0" step="1" oninput="manualAdjust()"><span id="mTyV">0</span></div>
    <div class="slider-row"><label>Translate X</label><input id="mTx" type="range" min="-400" max="400" value="0" step="1" oninput="manualAdjust()"><span id="mTxV">0</span></div>
    <div class="slider-row"><label>HE opacity</label><input id="heOpacity" type="range" min="0" max="1" value="0.65" step="0.01" oninput="drawLiveOverlay()"><span id="heOpacityV">0.65</span></div>
    <div class="viewer">
      <div><b>Live masked overlay</b><br><canvas id="liveOverlay" class="mask-editor"></canvas></div>
      <div class="image-card"><div class="caption">Backend overlay/QC</div><img id="overlay"></div>
      <div class="image-card"><div class="caption">Registered mask</div><img id="maskReg"></div>
    </div>
  </section>
  <section>
    <h2>5. Save Coregistered Images</h2>
    <button onclick="saveFinal()">Save Final</button>
    <div id="busy-save" class="busy"><progress></progress> Saving final coregistered outputs...</div>
    <div id="saveLinks"></div>
  </section>
  <section><h2>Status</h2><div id="status" class="status">Ready.</div></section>
</main>
<script>
let sessionId=null; let debounce=null; let liveImages={he:null,oct:null,mask:null};
let editors={};
function setStatus(x){ document.getElementById('status').textContent = x; }
async function api(path, body){ const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); return j; }
function img(name){ return `/api/file?session=${sessionId}&name=${name}&t=${Date.now()}`; }
function setBusy(id, on){ const el=document.getElementById(id); if(el) el.classList.toggle('active', on); document.querySelectorAll('button').forEach(b=>b.disabled=on); }
async function withBusy(id, label, fn){ setBusy(id,true); setStatus(label); try { const out=await fn(); return out; } catch(e) { setStatus('Error: '+e.message); throw e; } finally { setBusy(id,false); } }
async function loadPaths(){ await withBusy('busy-load','Loading paths...', async()=>{ const j=await api('/api/load_paths',{oct_path:octPath.value,he_path:hePath.value}); sessionId=j.session_id; setStatus('Loaded session '+sessionId); }); }
async function uploadFiles(){ await withBusy('busy-load','Uploading files...', async()=>{ const fd=new FormData(); fd.append('oct', octFile.files[0]); fd.append('he', heFile.files[0]); const r=await fetch('/api/upload',{method:'POST',body:fd}); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); sessionId=j.session_id; setStatus('Uploaded session '+sessionId); }); }
async function preprocess(){ await withBusy('busy-preprocess','Preprocessing OCT and HE...', async()=>{ const j=await api('/api/preprocess',{session_id:sessionId,stain_normalizer:stain.value}); octRaw.src=img('oct_raw_display_preview.png'); octFlat.src=img('oct_flatfield_corrected_preview.png'); octTile.src=img('oct_tile_artifact_suppressed_preview.png'); octPre.src=img('oct_registered_preview.png'); hePre.src=img('he_standardized_native_preview.png'); heBW.src=img('he_black_white_input_preview.png'); setStatus(JSON.stringify(j,null,2)); }); }
async function removeBackground(){ await withBusy('busy-mask','Removing background...', async()=>{ const j=await api('/api/masks',{session_id:sessionId,he_mask_mode:heMaskMode.value,he_gray_percentile:parseFloat(hePct.value)}); octMaskOverlay.src=img('oct_mask_overlay.png'); heMaskOverlay.src=img('he_mask_overlay.png'); await loadCanvas('octCanvas','oct_mask_edit.png','oct_search_feature.png',[0,255,70]); await loadCanvas('heCanvas','he_mask_edit.png','he_search_bw.png',[255,35,20]); setStatus(JSON.stringify(j,null,2)); }); }
function imageLoad(src){ return new Promise((res,rej)=>{ const im=new Image(); im.onload=()=>res(im); im.onerror=rej; im.src=src; }); }
async function loadCanvas(id, maskName, baseName, color){ const c=document.getElementById(id); const base=await imageLoad(img(baseName)); const mask=await imageLoad(img(maskName)); c.width=base.width; c.height=base.height; const maskCanvas=document.createElement('canvas'); maskCanvas.width=base.width; maskCanvas.height=base.height; const mctx=maskCanvas.getContext('2d'); mctx.drawImage(mask,0,0,base.width,base.height); editors[id]={base,maskCanvas,maskCtx:mctx,color,history:[]}; redrawEditor(id); setupDraw(c); }
function redrawEditor(id){ const c=document.getElementById(id); const ctx=c.getContext('2d'); const ed=editors[id]; if(!ed)return; ctx.clearRect(0,0,c.width,c.height); ctx.drawImage(ed.base,0,0,c.width,c.height); const maskData=ed.maskCtx.getImageData(0,0,c.width,c.height); const overlay=ctx.getImageData(0,0,c.width,c.height); for(let i=0;i<maskData.data.length;i+=4){ if(maskData.data[i]>127){ overlay.data[i]=Math.round(overlay.data[i]*0.55+ed.color[0]*0.45); overlay.data[i+1]=Math.round(overlay.data[i+1]*0.55+ed.color[1]*0.45); overlay.data[i+2]=Math.round(overlay.data[i+2]*0.55+ed.color[2]*0.45); }} ctx.putImageData(overlay,0,0); }
function setupDraw(c){ let down=false; function paintAt(e){ e.preventDefault(); const ed=editors[c.id]; if(!ed)return; const r=c.getBoundingClientRect(); const x=(e.clientX-r.left)*c.width/r.width, y=(e.clientY-r.top)*c.height/r.height; ed.maskCtx.fillStyle=paint.value==='white'?'white':'black'; ed.maskCtx.beginPath(); ed.maskCtx.arc(x,y,parseFloat(brush.value),0,Math.PI*2); ed.maskCtx.fill(); redrawEditor(c.id); } c.onpointerdown=e=>{ const ed=editors[c.id]; if(ed){ ed.history.push(ed.maskCtx.getImageData(0,0,c.width,c.height)); if(ed.history.length>25)ed.history.shift(); } down=true; c.setPointerCapture(e.pointerId); paintAt(e)}; c.onpointermove=e=>{ if(down)paintAt(e); }; c.onpointerup=e=>{down=false; try{c.releasePointerCapture(e.pointerId)}catch(_){}}; c.onpointercancel=()=>{down=false}; }
function undoEdit(id){ const ed=editors[id]; if(!ed||!ed.history.length){ setStatus('Nothing to undo for '+id); return; } ed.maskCtx.putImageData(ed.history.pop(),0,0); redrawEditor(id); setStatus('Undid last edit for '+id); }
function canvasData(id){ const ed=editors[id]; return ed ? ed.maskCanvas.toDataURL('image/png') : document.getElementById(id).toDataURL('image/png'); }
async function saveMasks(){ const j=await api('/api/save_masks',{session_id:sessionId,oct_mask:canvasData('octCanvas'),he_mask:canvasData('heCanvas')}); setStatus(JSON.stringify(j,null,2)); }
async function autoRegister(){ await withBusy('busy-autoreg','Running auto registration...', async()=>{ await saveMasks(); const j=await api('/api/autoreg',{session_id:sessionId}); await refreshReg(); setStatus(JSON.stringify(j.auto_params,null,2)); }); }
function manualAdjust(){ mScaleV.textContent=mScale.value; mRotV.textContent=mRot.value; mTyV.textContent=mTy.value; mTxV.textContent=mTx.value; drawLiveOverlay(); clearTimeout(debounce); debounce=setTimeout(async()=>{ setStatus('Applying manual adjustment to native outputs...'); await api('/api/manual',{session_id:sessionId,scale:parseFloat(mScale.value),rotation_deg:parseFloat(mRot.value),translation_y:parseFloat(mTy.value),translation_x:parseFloat(mTx.value)}); await refreshReg(); setStatus('Manual adjustment applied.'); },650); }
async function refreshReg(){ overlay.src=img('overlay_preview.png'); maskReg.src=img('registered_mask_preview.png'); liveImages.he=await imageLoad(img('he_registered_masked_preview.png')); liveImages.oct=await imageLoad(img('oct_registered_masked_preview.png')); liveImages.mask=await imageLoad(img('registered_mask_preview.png')); initLiveCanvas(); drawLiveOverlay(); }
function initLiveCanvas(){ if(!liveImages.oct)return; liveOverlay.width=liveImages.oct.width; liveOverlay.height=liveImages.oct.height; }
function drawLiveOverlay(){ heOpacityV.textContent=heOpacity.value; if(!liveImages.he||!liveImages.oct)return; const c=liveOverlay, ctx=c.getContext('2d'); if(c.width!==liveImages.oct.width){initLiveCanvas();} ctx.clearRect(0,0,c.width,c.height); ctx.globalAlpha=1; ctx.drawImage(liveImages.oct,0,0,c.width,c.height); ctx.save(); ctx.translate(c.width/2+parseFloat(mTx.value), c.height/2+parseFloat(mTy.value)); ctx.rotate(parseFloat(mRot.value)*Math.PI/180); const s=parseFloat(mScale.value); ctx.scale(s,s); ctx.globalAlpha=parseFloat(heOpacity.value); ctx.drawImage(liveImages.he,-c.width/2,-c.height/2,c.width,c.height); ctx.restore(); ctx.globalAlpha=1; }
async function saveFinal(){ await withBusy('busy-save','Saving final outputs...', async()=>{ const j=await api('/api/save',{session_id:sessionId}); saveLinks.innerHTML = j.files.map(f=>`<div><a href="/api/file?session=${sessionId}&name=${f}" target="_blank">${f}</a></div>`).join(''); setStatus(JSON.stringify(j,null,2)); }); }
</script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/file":
            qs = parse_qs(parsed.query)
            session_id = qs.get("session", [""])[0]
            name = qs.get("name", [""])[0]
            try:
                paths = _session(session_id)
                file_path = (paths.root / name).resolve()
                if paths.root.resolve() not in file_path.parents and file_path != paths.root.resolve():
                    raise PermissionError("Invalid file path")
                data = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 404)
            return
        _json_response(self, {"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload":
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
                oct_item = form["oct"]
                he_item = form["he"]
                oct_filename = Path(oct_item.filename or "oct.tiff").name
                he_filename = Path(he_item.filename or "he.tiff").name
                session_id, root = _unique_session_root(_session_id_from_paths(Path(oct_filename), Path(he_filename)))
                upload_dir = root / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                oct_path = upload_dir / oct_filename
                he_path = upload_dir / he_filename
                oct_path.write_bytes(oct_item.file.read())
                he_path.write_bytes(he_item.file.read())
                _write_session(root, oct_path, he_path)
                _json_response(self, {"session_id": session_id, "output_dir": str(root), "oct_path": str(oct_path), "he_path": str(he_path)})
            elif parsed.path == "/api/load_paths":
                payload = _read_json(self)
                oct_path = Path(payload["oct_path"]).expanduser().resolve()
                he_path = Path(payload["he_path"]).expanduser().resolve()
                if not oct_path.exists() or not he_path.exists():
                    raise FileNotFoundError("OCT or HE path does not exist")
                session_id, root = _unique_session_root(_session_id_from_paths(oct_path, he_path))
                _write_session(root, oct_path, he_path)
                _json_response(self, {"session_id": session_id, "output_dir": str(root)})
            elif parsed.path == "/api/preprocess":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                state = _prepare_state(paths, payload.get("stain_normalizer", "torchstain_reinhard"), payload.get("stain_reference_he"))
                _json_response(self, {"ok": True, "state": state, "output_dir": str(paths.root)})
            elif parsed.path == "/api/masks":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                state = _compute_masks(
                    paths,
                    payload.get("he_mask_mode", "auto"),
                    float(payload.get("he_gray_percentile", 67)),
                    float(payload.get("he_alpha", 0.376)),
                    float(payload.get("oct_alpha", 0.376)),
                )
                _json_response(self, {"ok": True, "mask_state": state})
            elif parsed.path == "/api/save_masks":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                for key, name in [("oct_mask", "oct_mask_edit.png"), ("he_mask", "he_mask_edit.png")]:
                    data = payload[key].split(",", 1)[1]
                    (paths.root / name).write_bytes(base64.b64decode(data))
                _json_response(self, {"ok": True})
            elif parsed.path == "/api/autoreg":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                state = _estimate_registration(paths)
                _json_response(self, state)
            elif parsed.path == "/api/manual":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                state = json.loads((paths.root / "transform_state.json").read_text())
                state["manual"] = {
                    "scale": float(payload.get("scale", 1.0)),
                    "rotation_deg": float(payload.get("rotation_deg", 0.0)),
                    "translation_y": float(payload.get("translation_y", 0.0)),
                    "translation_x": float(payload.get("translation_x", 0.0)),
                }
                (paths.root / "transform_state.json").write_text(json.dumps(state, indent=2))
                _apply_current_transform(paths, state)
                _json_response(self, {"ok": True, "manual": state["manual"]})
            elif parsed.path == "/api/save":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                _sync_clean_outputs(paths)
                files = [
                    "output/he_registered.tiff",
                    "output/oct_registered.tiff",
                    "output/registered_mask.tiff",
                    "alignment_summary.json",
                ]
                summary = {
                    "session_id": payload["session_id"],
                    "oct_path": str(paths.oct_path),
                    "he_path": str(paths.he_path),
                    "output_dir": str(paths.root),
                    "files": {f: str(paths.root / f) for f in files if (paths.root / f).exists()},
                }
                (paths.root / "alignment_summary.json").write_text(json.dumps(summary, indent=2))
                _json_response(self, {"ok": True, "output_dir": str(paths.root), "files": [f for f in files if (paths.root / f).exists()]})
            else:
                _json_response(self, {"error": "not found"}, 404)
        except Exception as exc:
            _json_response(self, {"error": str(exc)}, 500)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interactive OCT/HE coregistration app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Interactive coregistration app: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
