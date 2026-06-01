from __future__ import annotations

import base64
import cgi
import concurrent.futures
import json
import math
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
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
ALIGN_SCRIPT = Path(__file__).resolve().parent / "align_he_to_oct_2D_v5.py"
BATCH_JOBS: dict[str, dict[str, Any]] = {}
BATCH_LOCK = threading.Lock()


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


def _clean_user_path_value(value: Any) -> str:
    """Normalize paths pasted from shells/Finder, including surrounding quotes."""
    text = str(value or "").strip()
    quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    while len(text) >= 2 and text[0] in quote_pairs and text[-1] == quote_pairs[text[0]]:
        text = text[1:-1].strip()
    return text


def _user_path(value: Any) -> Path:
    return Path(_clean_user_path_value(value)).expanduser().resolve()


def _session(session_id: str) -> SessionPaths:
    root = APP_ROOT / session_id
    manifest = root / "session.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Unknown session: {session_id}")
    data = json.loads(manifest.read_text())
    session_root = Path(data.get("root", root))
    return SessionPaths(root=session_root, oct_path=Path(data["oct_path"]), he_path=Path(data["he_path"]))


def _write_session(root: Path, oct_path: Path, he_path: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    cached = _cache_input_images(root, oct_path, he_path)
    payload = {"root": str(root), "oct_path": str(oct_path), "he_path": str(he_path)}
    payload.update(cached)
    (root / "session.json").write_text(json.dumps(payload, indent=2))


def _input_cache_dir(root: Path) -> Path:
    return root / "inputs"


def _cache_input_images(root: Path, oct_path: Path, he_path: Path) -> dict[str, str]:
    cache_dir = _input_cache_dir(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached: dict[str, str] = {}
    for key, src in (("oct", oct_path), ("he", he_path)):
        src = Path(src)
        dst = cache_dir / f"{key}{src.suffix or '.tiff'}"
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
        if dst.exists():
            cached[f"{key}_relative_path"] = str(dst.relative_to(root))
            cached[f"{key}_original_name"] = src.name
    if cached:
        manifest = cache_dir / "inputs.json"
        current = json.loads(manifest.read_text()) if manifest.exists() else {}
        current.update(cached)
        current.update({"oct_original_path": str(oct_path), "he_original_path": str(he_path)})
        manifest.write_text(json.dumps(current, indent=2))
    return cached


def _write_session_alias(alias_root: Path, output_root: Path, oct_path: Path, he_path: Path) -> None:
    alias_root.mkdir(parents=True, exist_ok=True)
    (alias_root / "session.json").write_text(
        json.dumps({"root": str(output_root), "oct_path": str(oct_path), "he_path": str(he_path)}, indent=2)
    )


def _update_session_he_path(session_id: str, output_root: Path, oct_path: Path, he_path: Path) -> None:
    payload = {"root": str(output_root), "oct_path": str(oct_path), "he_path": str(he_path)}
    (output_root / "session.json").write_text(json.dumps(payload, indent=2))
    alias_root = APP_ROOT / session_id
    if alias_root.exists():
        (alias_root / "session.json").write_text(json.dumps(payload, indent=2))


def _existing_path_from_json(value: Any, sample_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(sample_dir / path)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            return resolved
    return None


def _local_input_candidate(sample_dir: Path, kind: str, original_name: str | None = None) -> Path | None:
    cache_dir = _input_cache_dir(sample_dir)
    candidates: list[Path] = []
    if original_name:
        candidates.extend([cache_dir / original_name, sample_dir / original_name])
    for parent in [cache_dir, sample_dir]:
        if parent.exists():
            candidates.extend(sorted(parent.glob(f"{kind}.*")))
            candidates.extend(sorted(parent.glob(f"*{kind}*.tif*")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _paths_from_processed_output(sample_dir: Path) -> tuple[Path, Path]:
    records: list[dict[str, Any]] = []
    input_manifest = _input_cache_dir(sample_dir) / "inputs.json"
    if input_manifest.exists():
        try:
            records.append(json.loads(input_manifest.read_text()))
        except Exception:
            pass
    session_manifest = sample_dir / "session.json"
    if session_manifest.exists():
        try:
            records.append(json.loads(session_manifest.read_text()))
        except Exception:
            pass
    summary_path = sample_dir / "alignment_summary.json"
    if summary_path.exists():
        try:
            records.append(json.loads(summary_path.read_text()))
        except Exception:
            pass
    oct_path = he_path = None
    for data in records:
        oct_path = oct_path or _existing_path_from_json(data.get("oct_relative_path") or data.get("oct_path") or data.get("oct_original_path"), sample_dir)
        he_path = he_path or _existing_path_from_json(data.get("he_relative_path") or data.get("he_path") or data.get("he_original_path"), sample_dir)
    for data in records:
        oct_path = oct_path or _local_input_candidate(sample_dir, "oct", data.get("oct_original_name"))
        he_path = he_path or _local_input_candidate(sample_dir, "he", data.get("he_original_name"))
    if oct_path and he_path:
        return oct_path, he_path
    raise FileNotFoundError("Processed sample is missing reachable OCT/HE inputs. Keep inputs/ with the output folder, or keep the original files at their saved paths.")


def _scan_processed_outputs(output_root: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError("Processed output directory does not exist")
    markers = {"overlay_preview.png", "alignment_summary.json", "session.json", "he_registered.tiff"}
    candidate_dirs = [output_root] + sorted(p for p in output_root.rglob("*") if p.is_dir())
    for sample_dir in candidate_dirs:
        names = {child.name for child in sample_dir.iterdir() if child.is_file()}
        if not markers.intersection(names):
            continue
        if not ((sample_dir / "he_registered.tiff").exists() or (sample_dir / "output" / "he_registered.tiff").exists()):
            continue
        rel = sample_dir.relative_to(output_root)
        label = output_root.name if str(rel) == "." else str(rel)
        samples.append({"name": label, "path": str(sample_dir)})
    return samples


def _known_output_files(root: Path) -> list[str]:
    names = [
        "oct_raw_display_preview.png",
        "oct_flatfield_corrected_preview.png",
        "oct_tile_artifact_suppressed_preview.png",
        "oct_registered_preview.png",
        "he_standardized_native_preview.png",
        "he_black_white_input_preview.png",
        "oct_mask_overlay.png",
        "he_mask_overlay.png",
        "oct_mask_edit.png",
        "he_mask_edit.png",
        "oct_mask_editor_base.png",
        "he_mask_editor_base.png",
        "oct_search_feature.png",
        "he_search_bw.png",
        "overlay_preview.png",
        "registered_mask_preview.png",
        "he_autoreg_preview.png",
        "he_autoreg_mask_preview.png",
        "oct_mask_preview.png",
        "oct_live_qc_preview.png",
        "he_autoreg_live_qc_preview.png",
        "he_autoreg_mask_live_qc_preview.png",
        "oct_mask_live_qc_preview.png",
        "live_qc_state.json",
        "live_backend_qc_preview.png",
        "he_live_qc_source.png",
        "he_mask_live_qc_source.png",
        "he_registered_masked_preview.png",
        "oct_registered_masked_preview.png",
        "he_registered.tiff",
        "oct_registered.tiff",
        "registered_mask.tiff",
        "output/he_registered.tiff",
        "output/oct_registered.tiff",
        "output/registered_mask.tiff",
    ]
    return [name for name in names if (root / name).exists()]


def _processed_state(root: Path) -> dict[str, Any]:
    state: dict[str, Any] = {}
    transform_path = root / "transform_state.json"
    if transform_path.exists():
        try:
            state["transform_state"] = json.loads(transform_path.read_text())
        except Exception:
            state["transform_state"] = None
    mask_state_path = root / "mask_state.json"
    if mask_state_path.exists():
        try:
            state["mask_state"] = json.loads(mask_state_path.read_text())
        except Exception:
            state["mask_state"] = None
    return state


def _load_alignment_summary(root: Path) -> dict[str, Any]:
    summary_path = root / "alignment_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text())
    except Exception:
        return {}


def _rebuild_transform_state_from_summary(paths: SessionPaths, summary: dict[str, Any]) -> dict[str, Any] | None:
    best = summary.get("he_transform_into_oct_xy") or {}
    if not best or not (paths.root / "mask_state.json").exists():
        return None
    search_shape = tuple(json.loads((paths.root / "mask_state.json").read_text())["search_shape"])
    oct_mask = _load_mask_png(paths.root / "oct_mask_edit.png")
    he_mask = _load_mask_png(paths.root / "he_mask_edit.png")
    if oct_mask.shape != search_shape:
        oct_mask = _resize_mask(oct_mask, search_shape)
    if he_mask.shape != search_shape:
        he_mask = _resize_mask(he_mask, search_shape)
    oct_center = _mask_center(oct_mask)
    he_center = _mask_center(he_mask)
    search_matrix = _affine(
        float(best.get("scale", 1.0)),
        float(best.get("rotation_deg", 0.0)),
        float(best.get("translation_y", 0.0)),
        float(best.get("translation_x", 0.0)),
        he_center,
        oct_center,
    )
    he_shape = _ensure_rgb(_read_tiff(paths.he_path)).shape[:2]
    if (paths.root / "oct_registered.tiff").exists():
        oct_shape = tifffile.imread(str(paths.root / "oct_registered.tiff")).shape[:2]
    else:
        oct_shape = _oct_to_gray(_read_tiff(paths.oct_path)).shape[:2]
    native = _native_matrix(search_matrix, he_shape, oct_shape, search_shape)
    transform_state = {
        "auto_params": {
            "scale": float(best.get("scale", 1.0)),
            "rotation_deg": float(best.get("rotation_deg", 0.0)),
            "tilt_x_deg": 0.0,
            "tilt_y_deg": 0.0,
            "translation_y": float(best.get("translation_y", 0.0)),
            "translation_x": float(best.get("translation_x", 0.0)),
            "score": float(best.get("score", 0.0)),
            "details": best.get("details", {}),
        },
        "native_matrix": native.tolist(),
        "manual": {"scale": 1.0, "stretch_x": 1.0, "stretch_y": 1.0, "rotation_deg": 0.0, "translation_y": 0.0, "translation_x": 0.0, "he_opacity": 0.65},
    }
    (paths.root / "transform_state.json").write_text(json.dumps(transform_state, indent=2))
    return transform_state


def _ensure_interactive_artifacts(paths: SessionPaths) -> list[str]:
    """Backfill files needed to reopen batch/script outputs as editable sessions."""
    paths.root.mkdir(parents=True, exist_ok=True)
    cached = _cache_input_images(paths.root, paths.oct_path, paths.he_path)
    summary = _load_alignment_summary(paths.root)
    created: list[str] = []
    if cached:
        created.append("relocatable input copies")
    required_previews = [
        "oct_raw_display_preview.png",
        "oct_flatfield_corrected_preview.png",
        "oct_tile_artifact_suppressed_preview.png",
        "oct_registered_preview.png",
        "he_standardized_native.tiff",
        "he_standardized_native_preview.png",
        "he_black_white_input_preview.png",
    ]
    if any(not (paths.root / name).exists() for name in required_previews):
        normalizer = str(summary.get("stain_normalizer") or "torchstain_reinhard")
        _prepare_state(paths, normalizer)
        created.append("preprocessing previews")
    required_masks = [
        "mask_state.json",
        "oct_mask_edit.png",
        "he_mask_edit.png",
        "oct_mask_editor_base.png",
        "he_mask_editor_base.png",
        "oct_search_feature.png",
        "he_search_bw.png",
        "oct_mask_overlay.png",
        "he_mask_overlay.png",
    ]
    if any(not (paths.root / name).exists() for name in required_masks):
        _compute_masks(
            paths,
            str(summary.get("he_mask_mode") or "auto"),
            float(summary.get("he_gray_mask_percentile", 67.0)),
            float(summary.get("he_alpha_threshold", 0.376)),
            float(summary.get("oct_alpha_threshold", 0.376)),
        )
        created.append("editable masks")
    if not (paths.root / "transform_state.json").exists():
        rebuilt = _rebuild_transform_state_from_summary(paths, summary)
        if rebuilt is None:
            _estimate_registration(paths)
            created.append("registration transform")
        else:
            created.append("registration transform")
    required_registration_previews = [
        "overlay_preview.png",
        "registered_mask_preview.png",
        "he_autoreg_preview.png",
        "he_autoreg_mask_preview.png",
        "oct_mask_preview.png",
        "oct_live_qc_preview.png",
        "he_autoreg_live_qc_preview.png",
        "he_autoreg_mask_live_qc_preview.png",
        "oct_mask_live_qc_preview.png",
        "live_qc_state.json",
        "live_backend_qc_preview.png",
        "he_live_qc_source.png",
        "he_mask_live_qc_source.png",
        "he_registered_masked_preview.png",
        "oct_registered_masked_preview.png",
    ]
    if any(not (paths.root / name).exists() for name in required_registration_previews):
        _apply_current_transform(paths)
        created.append("registration previews")
    _sync_clean_outputs(paths)
    return created


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


def _unique_child_root(parent: Path, session_id: str) -> tuple[str, Path]:
    candidate = _slug(session_id)
    root = parent / candidate
    if not root.exists():
        return candidate, root
    for index in range(2, 10000):
        indexed = f"{candidate}_{index:02d}"
        root = parent / indexed
        if not root.exists():
            return indexed, root
    raise RuntimeError(f"Could not create a unique output folder for {candidate}")


def _specified_output_root(value: str | None, default_parent: Path = APP_ROOT) -> Path:
    if value is None or not _clean_user_path_value(value):
        return default_parent
    return _user_path(value)


def _session_root_for_output(parent: Path, session_id: str, overwrite: bool = False) -> tuple[str, Path]:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = _slug(session_id)
    root = parent / candidate
    if overwrite:
        if root.exists():
            shutil.rmtree(root)
        return candidate, root
    return _unique_child_root(parent, candidate)


def _find_batch_pairs(input_root: Path, batch_root: Path, overwrite: bool = False) -> list[dict[str, Any]]:
    tif_paths = sorted(
        p for p in input_root.rglob("*.tif*")
        if p.is_file() and not p.name.startswith("._") and "coregistration_outputs" not in p.parts
    )
    oct_paths = [p for p in tif_paths if "oct" in p.stem.lower()]
    cases: list[dict[str, Any]] = []
    seen: set[tuple[Path, Path]] = set()
    for oct_path in oct_paths:
        oct_base = _slug(_clean_stem(oct_path, ("_oct", "-oct", " oct"))).lower()
        folder_tifs = [p for p in tif_paths if p.parent == oct_path.parent and p != oct_path]
        he_candidates = [p for p in folder_tifs if "oct" not in p.stem.lower()]
        matching = [p for p in he_candidates if oct_base and oct_base in _slug(p.stem).lower()]
        selected = matching or he_candidates
        for he_path in selected:
            key = (oct_path.resolve(), he_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            case_id = _session_id_from_paths(oct_path, he_path)
            sample_id, output_dir = _session_root_for_output(batch_root, case_id, overwrite=overwrite)
            cases.append({
                "case_id": sample_id,
                "oct_path": oct_path,
                "he_path": he_path,
                "output_dir": output_dir,
                "session_id": f"{batch_root.name}/{sample_id}",
            })
    return cases


def _run_batch_case(case: dict[str, Any], overwrite: bool = True) -> dict[str, Any]:
    output_dir = Path(case["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_session(output_dir, Path(case["oct_path"]), Path(case["he_path"]))
    log_path = output_dir / "run.log"
    cmd = [
        sys.executable,
        str(ALIGN_SCRIPT),
        "--oct-path",
        str(case["oct_path"]),
        "--he-path",
        str(case["he_path"]),
        "--output-dir",
        str(output_dir),
        "--case-id",
        str(case["case_id"]),
        "--overwrite",
    ]
    if not overwrite:
        cmd.pop()
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        if proc.returncode == 0:
            try:
                rebuilt = _ensure_interactive_artifacts(SessionPaths(output_dir, Path(case["oct_path"]), Path(case["he_path"])))
                if rebuilt:
                    log.write("\nInteractive reload artifacts saved: " + ", ".join(rebuilt) + "\n")
            except Exception as exc:
                proc = subprocess.CompletedProcess(cmd, 1)
                log.write(f"\nFailed to save interactive reload artifacts: {exc}\n")
    result = {
        "case_id": case["case_id"],
        "session_id": case["session_id"],
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - start, 2),
        "oct_path": str(case["oct_path"]),
        "he_path": str(case["he_path"]),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "overlay": "overlay_preview.png" if (output_dir / "overlay_preview.png").exists() else "overlay_false_color.png",
        "keep": True,
    }
    _sync_clean_outputs(SessionPaths(output_dir, Path(case["oct_path"]), Path(case["he_path"])))
    return result


def _run_batch_job(batch_id: str, cases: list[dict[str, Any]], workers: int) -> None:
    with BATCH_LOCK:
        BATCH_JOBS[batch_id]["status"] = "running"
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_case = {executor.submit(_run_batch_case, case): case for case in cases}
        for future in concurrent.futures.as_completed(future_to_case):
            case = future_to_case[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "case_id": case["case_id"],
                    "session_id": case["session_id"],
                    "status": "failed",
                    "error": str(exc),
                    "oct_path": str(case["oct_path"]),
                    "he_path": str(case["he_path"]),
                    "output_dir": str(case["output_dir"]),
                    "keep": True,
                }
            with BATCH_LOCK:
                job = BATCH_JOBS[batch_id]
                job["completed"] += 1
                job["results"].append(result)
    with BATCH_LOCK:
        BATCH_JOBS[batch_id]["status"] = "completed"
        BATCH_JOBS[batch_id]["finished_at"] = time.time()


def _public_batch_job(batch_id: str) -> dict[str, Any]:
    with BATCH_LOCK:
        if batch_id not in BATCH_JOBS:
            raise FileNotFoundError(f"Unknown batch: {batch_id}")
        job = dict(BATCH_JOBS[batch_id])
        job["results"] = [dict(result) for result in BATCH_JOBS[batch_id].get("results", [])]
    job.pop("cases", None)
    return job


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


def _send_file(handler: BaseHTTPRequestHandler, root: Path, name: str) -> None:
    root = root.resolve()
    file_path = (root / name).resolve()
    if root not in file_path.parents and file_path != root:
        raise PermissionError("Invalid file path")
    data = file_path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _batch_output_dir(batch_id: str, case_id: str) -> Path:
    with BATCH_LOCK:
        job = BATCH_JOBS.get(batch_id)
        if not job:
            raise FileNotFoundError(f"Unknown batch: {batch_id}")
        for collection in (job.get("results", []), job.get("cases", [])):
            for item in collection:
                if item.get("case_id") == case_id:
                    return Path(item["output_dir"])
    raise FileNotFoundError(f"Unknown batch case: {case_id}")


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


def _clear_he_dependent_outputs(root: Path) -> None:
    for name in _known_output_files(root):
        if name.startswith("oct_") and name not in {"oct_registered_masked_preview.png"}:
            continue
        path = root / name
        if path.is_file():
            path.unlink()
    for name in ["state.json", "mask_state.json", "transform_state.json", "alignment_summary.json"]:
        path = root / name
        if path.is_file():
            path.unlink()


def _flip_he_for_session(session_id: str, paths: SessionPaths, stain_normalizer: str) -> dict[str, Any]:
    he_raw = tifffile.imread(str(paths.he_path))
    flipped = np.flip(he_raw, axis=1)
    input_dir = _input_cache_dir(paths.root)
    input_dir.mkdir(parents=True, exist_ok=True)
    flipped_path = input_dir / "he_flipped_y_axis.tiff"
    tifffile.imwrite(str(flipped_path), flipped)
    _update_session_he_path(session_id, paths.root, paths.oct_path, flipped_path)
    flipped_paths = SessionPaths(paths.root, paths.oct_path, flipped_path)
    _clear_he_dependent_outputs(paths.root)
    state = _prepare_state(flipped_paths, stain_normalizer)
    return {"ok": True, "he_path": str(flipped_path), "state": state, "output_dir": str(paths.root)}


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
    _save_gray_png(paths.root / "oct_mask_editor_base.png", exposure.rescale_intensity(oct_search, out_range=(0.0, 1.0)))
    _save_rgb_png(paths.root / "he_mask_editor_base.png", he_search_rgb)
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
        "manual": {"scale": 1.0, "stretch_x": 1.0, "stretch_y": 1.0, "rotation_deg": 0.0, "translation_y": 0.0, "translation_x": 0.0, "he_opacity": 0.65},
    }
    (paths.root / "transform_state.json").write_text(json.dumps(transform_state, indent=2))
    _apply_current_transform(paths, transform_state)
    return transform_state


def _manual_matrix(shape: tuple[int, int], scale: float, stretch_x: float, stretch_y: float, rotation_deg: float, ty: float, tx: float) -> np.ndarray:
    center_yx = np.array([shape[0] / 2.0, shape[1] / 2.0], dtype=np.float64)
    src_xy = np.array([center_yx[1], center_yx[0]], dtype=np.float64)
    dst_xy = np.array([center_yx[1] + tx, center_yx[0] + ty], dtype=np.float64)
    theta = math.radians(float(rotation_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    sx = float(scale) * float(stretch_x)
    sy = float(scale) * float(stretch_y)
    linear = np.array([[cos_t * sx, -sin_t * sy], [sin_t * sx, cos_t * sy]], dtype=np.float64)
    offset = dst_xy - linear @ src_xy
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[0, 2] = offset[0]
    matrix[1, 2] = offset[1]
    return matrix


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
        float(manual.get("stretch_x", 1.0)),
        float(manual.get("stretch_y", 1.0)),
        float(manual.get("rotation_deg", 0.0)),
        float(manual.get("translation_y", 0.0)),
        float(manual.get("translation_x", 0.0)),
    )
    matrix = correction @ native_matrix
    he_standardized = tifffile.imread(str(paths.root / "he_standardized_native.tiff"))
    he_mask = _resize_mask(_load_mask_png(paths.root / "he_mask_edit.png"), tuple(json.loads((paths.root / "mask_state.json").read_text())["search_shape"]))
    he_mask_native = _resize_mask(he_mask, he_standardized.shape[:2])
    oct_mask = _resize_mask(_load_mask_png(paths.root / "oct_mask_edit.png"), output_shape)
    auto_he = _warp(he_standardized.astype(np.float32), native_matrix, output_shape, order=1)
    auto_he_mask = _warp(he_mask_native.astype(np.float32), native_matrix, output_shape, order=0) > 0.5
    warped_he = _warp(he_standardized.astype(np.float32), matrix, output_shape, order=1)
    warped_he_mask = _warp(he_mask_native.astype(np.float32), matrix, output_shape, order=0) > 0.5
    overlap = warped_he_mask & oct_mask
    _save_rgb_tiff(paths.root / "he_registered.tiff", warped_he)
    _save_float_tiff(paths.root / "oct_registered.tiff", oct_registered)
    _save_mask_tiff(paths.root / "registered_mask.tiff", overlap)
    _save_rgb_png(paths.root / "he_registered_preview.png", warped_he)
    _save_rgb_png(paths.root / "he_autoreg_preview.png", auto_he)
    _save_gray_png(paths.root / "he_autoreg_mask_preview.png", auto_he_mask.astype(np.float32))
    _save_gray_png(paths.root / "oct_mask_preview.png", oct_mask.astype(np.float32))
    live_oct = _preview_image(oct_registered)
    live_shape = live_oct.shape[:2]
    _save_gray_png(paths.root / "oct_live_qc_preview.png", live_oct)
    _save_rgb_png(paths.root / "he_autoreg_live_qc_preview.png", _resize_rgb(auto_he.astype(np.float32), live_shape))
    _save_gray_png(paths.root / "he_autoreg_mask_live_qc_preview.png", _resize_mask(auto_he_mask, live_shape).astype(np.float32))
    _save_gray_png(paths.root / "oct_mask_live_qc_preview.png", _resize_mask(oct_mask, live_shape).astype(np.float32))
    src_live_shape = (
        max(1, int(round(he_standardized.shape[0] * live_shape[0] / max(1, output_shape[0])))),
        max(1, int(round(he_standardized.shape[1] * live_shape[1] / max(1, output_shape[1])))),
    )
    _save_rgb_png(paths.root / "he_live_qc_source.png", _resize_rgb(he_standardized.astype(np.float32), src_live_shape))
    _save_gray_png(paths.root / "he_mask_live_qc_source.png", _resize_mask(he_mask_native, src_live_shape).astype(np.float32))
    (paths.root / "live_qc_state.json").write_text(
        json.dumps(
            {
                "native_shape": [int(output_shape[0]), int(output_shape[1])],
                "live_shape": [int(live_shape[0]), int(live_shape[1])],
                "he_native_shape": [int(he_standardized.shape[0]), int(he_standardized.shape[1])],
                "he_live_source_shape": [int(src_live_shape[0]), int(src_live_shape[1])],
            },
            indent=2,
        )
    )
    _render_live_qc_preview(paths, manual)
    _save_gray_png(paths.root / "registered_mask_preview.png", overlap.astype(np.float32))
    _save_rgb_png(paths.root / "he_registered_masked_preview.png", warped_he * overlap[..., None].astype(np.float32))
    _save_gray_png(paths.root / "oct_registered_masked_preview.png", oct_registered * overlap.astype(np.float32))
    _save_rgb_png(paths.root / "overlay_preview.png", _overlay_preview(warped_he, oct_registered, overlap, float(manual.get("he_opacity", 0.65))))
    _sync_clean_outputs(paths)


def _overlay_preview(he_rgb: np.ndarray, oct_gray: np.ndarray, mask: np.ndarray, he_opacity: float = 0.65) -> np.ndarray:
    he = _to_uint8(he_rgb).astype(np.float32) / 255.0
    oct_img = _to_uint8(oct_gray).astype(np.float32) / 255.0
    opacity = float(np.clip(he_opacity, 0.0, 1.0))
    he_weight = 0.82 * opacity
    out = he * he_weight
    out[..., 1] = np.maximum(out[..., 1], oct_img * 0.95)
    out[..., 2] = np.maximum(out[..., 2], oct_img * 0.35)
    out[~mask.astype(bool)] *= 0.82
    return np.clip(out, 0, 1)


def _manual_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "scale": float(payload.get("scale", 1.0)),
        "stretch_x": float(payload.get("stretch_x", 1.0)),
        "stretch_y": float(payload.get("stretch_y", 1.0)),
        "rotation_deg": float(payload.get("rotation_deg", 0.0)),
        "translation_y": float(payload.get("translation_y", 0.0)),
        "translation_x": float(payload.get("translation_x", 0.0)),
        "he_opacity": float(payload.get("he_opacity", 0.65)),
    }


def _render_live_qc_preview(paths: SessionPaths, manual: dict[str, float]) -> None:
    state_path = paths.root / "live_qc_state.json"
    required = [
        paths.root / "he_live_qc_source.png",
        paths.root / "he_mask_live_qc_source.png",
        paths.root / "oct_live_qc_preview.png",
        paths.root / "oct_mask_live_qc_preview.png",
    ]
    if not state_path.exists() or any(not p.exists() for p in required):
        _apply_current_transform(paths)
    live_state = json.loads(state_path.read_text())
    live_shape = tuple(int(v) for v in live_state["live_shape"])
    native_shape = tuple(int(v) for v in live_state["native_shape"])
    he_native_shape = tuple(int(v) for v in live_state.get("he_native_shape", native_shape))
    he_live_shape = tuple(int(v) for v in live_state.get("he_live_source_shape", live_shape))
    state = json.loads((paths.root / "transform_state.json").read_text())
    native_matrix = np.asarray(state["native_matrix"], dtype=np.float64)
    correction_native = _manual_matrix(
        native_shape,
        float(manual.get("scale", 1.0)),
        float(manual.get("stretch_x", 1.0)),
        float(manual.get("stretch_y", 1.0)),
        float(manual.get("rotation_deg", 0.0)),
        float(manual.get("translation_y", 0.0)),
        float(manual.get("translation_x", 0.0)),
    )
    full_native = correction_native @ native_matrix
    dest_scale = np.eye(3, dtype=np.float64)
    dest_scale[0, 0] = live_shape[1] / max(1, native_shape[1])
    dest_scale[1, 1] = live_shape[0] / max(1, native_shape[0])
    src_scale = np.eye(3, dtype=np.float64)
    src_scale[0, 0] = he_live_shape[1] / max(1, he_native_shape[1])
    src_scale[1, 1] = he_live_shape[0] / max(1, he_native_shape[0])
    preview_matrix = dest_scale @ full_native @ np.linalg.inv(src_scale)
    he = np.asarray(Image.open(paths.root / "he_live_qc_source.png")).astype(np.float32)
    oct_img = np.asarray(Image.open(paths.root / "oct_live_qc_preview.png")).astype(np.float32)
    he_mask = np.asarray(Image.open(paths.root / "he_mask_live_qc_source.png")).astype(np.float32) > 127
    oct_mask = np.asarray(Image.open(paths.root / "oct_mask_live_qc_preview.png")).astype(np.float32) > 127
    warped_he = _warp(he, preview_matrix, live_shape, order=1)
    warped_he_mask = _warp(he_mask.astype(np.float32), preview_matrix, live_shape, order=0) > 0.5
    overlap = warped_he_mask & oct_mask
    _save_rgb_png(paths.root / "live_backend_qc_preview.png", _overlay_preview(warped_he, oct_img, overlap, float(manual.get("he_opacity", 0.65))))


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
    .batch-progress { flex-wrap:wrap; }
    .batch-progress progress { flex:1 1 260px; max-width:560px; }
    .batch-progress span { font-weight:900; color:#173b31; }
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
    .slider-row .value-input { width:78px; min-height:34px; padding:6px 8px; font-family:Menlo, Monaco, Consolas, monospace; text-align:right; color:#40504a; }
    .small { color:var(--muted); font-size:13px; line-height:1.45; margin-top:10px; }
    header .small { color:rgba(251,246,233,.76); }
    .app-meta { margin-top:14px; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:rgba(251,246,233,.62); }
    .small input, .small select { min-height:34px; padding:6px 9px; vertical-align:middle; }
    #saveLinks { margin-top:12px; display:grid; gap:8px; }
    #saveLinks a { color:var(--accent); font-weight:800; }
    .batch-results { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; margin-top:16px; }
    .batch-card { background:rgba(255,255,255,.62); border:1px solid rgba(215,208,192,.9); border-radius:18px; padding:12px; box-shadow:0 12px 28px rgba(38,43,39,.10); }
    .batch-card.failed { border-color:rgba(212,107,61,.55); background:rgba(255,245,239,.74); }
    .batch-card.deleted { opacity:.56; filter:grayscale(.25); }
    .batch-card img { width:100%; min-height:120px; max-height:260px; object-fit:contain; background:#111b18; border-radius:14px; border:1px solid var(--line); }
    .keep-row { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:10px; font-weight:900; }
    .keep-row input { min-height:auto; transform:scale(1.25); accent-color:var(--accent); }
    .pill { display:inline-flex; align-items:center; border-radius:999px; padding:4px 9px; font-size:12px; font-weight:900; background:#e7f4ed; color:#125b4c; }
    .pill.failed { background:#fff0e9; color:#9a4726; }
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
<header><div class="small">Interactive registration workspace</div><h1>OCT/HE Coregistration Studio</h1><div class="small">Preprocess | Mask and edit | Auto-register | Manually adjust | Save native-resolution outputs</div><div class="app-meta">Version 5.2 | Ates Fettahoglu</div></header>
<main>
  <section>
    <h2>Batch Process</h2>
    <div class="grid">
      <div><label>Input folder</label><input id="batchInput" type="text" placeholder="/path/to/folder/with/multiple/samples"></div>
      <div><label>Samples to process in parallel</label><input id="batchWorkers" type="number" value="4" min="1" max="16" step="1"></div>
    </div>
    <label>Batch output folder</label><input id="batchOutput" type="text" placeholder="Leave blank for OCT_HE_align_interact_v5/coregistration_outputs/interactive_app">
    <div class="small"><label><input id="batchOverwrite" type="checkbox"> overwrite already processed samples in the selected output folder</label></div>
    <button onclick="startBatch()">Run Batch Registration</button>
    <button class="secondary" onclick="applyBatchKeepChoices()">Delete Unchecked Outputs</button>
    <div id="busy-batch" class="busy batch-progress"><progress id="batchProgress" max="1" value="0"></progress><span id="batchProgressText">Detected 0 samples. Completed 0.</span></div>
    <div class="small">Batch mode runs the non-interactive v5 registration for every OCT/HE pair found in the input folder. Results default to keep. Uncheck any result you do not want, then click Delete Unchecked Outputs.</div>
    <div id="batchStatus" class="small">No batch run started.</div>
    <div id="batchResults" class="batch-results"></div>
  </section>
  <section>
    <h2>Load Processed Output</h2>
    <div class="grid">
      <div><label>Pipeline output directory</label><input id="processedRoot" type="text" placeholder="/path/to/OCT_HE_align_interact_v5/coregistration_outputs/interactive_app"></div>
      <div><label>Processed sample</label><select id="processedSample"><option value="">Scan an output directory first</option></select></div>
    </div>
    <button onclick="scanProcessedOutputs()">Scan Output Directory</button>
    <button onclick="loadProcessedSample()">Load Selected Sample</button>
    <div id="busy-processed" class="busy"><progress></progress> Loading processed outputs...</div>
    <div class="small">Use this to reopen a sample that was already processed by this app. If transform information is available, you can adjust registration in Step 4 and save again.</div>
  </section>
  <section>
    <h2>1. Load Images</h2>
    <div class="grid">
      <div><label>OCT path</label><input id="octPath" type="text" placeholder="/path/to/sample_oct.tiff"></div>
      <div><label>HE path</label><input id="hePath" type="text" placeholder="/path/to/sample_section.tiff"></div>
    </div>
    <label>Output folder</label><input id="singleOutput" type="text" placeholder="Leave blank for OCT_HE_align_interact_v5/coregistration_outputs/interactive_app">
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
    <button class="secondary" onclick="flipHe()">Flip HE Left-Right</button>
    <button onclick="runAllProcessing()">Run All Processing And Save</button>
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
    <h3>Edit masks directly on overlays</h3>
    <button class="secondary" onclick="undoEdit('octCanvas')">Undo OCT Edit</button>
    <button class="secondary" onclick="undoEdit('heCanvas')">Undo HE Edit</button>
    <div class="viewer"><div><b>OCT overlay editor</b><br><canvas id="octCanvas" class="mask-editor"></canvas></div><div><b>HE overlay editor</b><br><canvas id="heCanvas" class="mask-editor"></canvas></div></div>
  </section>
  <section>
    <h2>4. Auto Registration And Manual Adjustment</h2>
    <button onclick="autoRegister()">Run Auto Registration</button>
    <div id="busy-autoreg" class="busy"><progress></progress> Running auto-registration and applying native-resolution transform...</div>
    <div class="slider-row"><label>Scale</label><input id="mScale" type="range" min="0.70" max="1.30" value="1" step="0.002" oninput="manualAdjust()"><input id="mScaleV" class="value-input" type="number" value="1" step="0.002" oninput="manualNumberAdjust('mScale','mScaleV')"></div>
    <div class="slider-row"><label>X stretch</label><input id="mStretchX" type="range" min="0.70" max="1.30" value="1" step="0.002" oninput="manualAdjust()"><input id="mStretchXV" class="value-input" type="number" value="1" step="0.002" oninput="manualNumberAdjust('mStretchX','mStretchXV')"></div>
    <div class="slider-row"><label>Y stretch</label><input id="mStretchY" type="range" min="0.70" max="1.30" value="1" step="0.002" oninput="manualAdjust()"><input id="mStretchYV" class="value-input" type="number" value="1" step="0.002" oninput="manualNumberAdjust('mStretchY','mStretchYV')"></div>
    <div class="slider-row"><label>Rotation</label><input id="mRot" type="range" min="-180" max="180" value="0" step="0.1" oninput="manualAdjust()"><input id="mRotV" class="value-input" type="number" value="0" step="0.1" oninput="manualNumberAdjust('mRot','mRotV')"></div>
    <div class="slider-row"><label>Translate Y</label><input id="mTy" type="range" min="-2000" max="2000" value="0" step="1" oninput="manualAdjust()"><input id="mTyV" class="value-input" type="number" value="0" step="1" oninput="manualNumberAdjust('mTy','mTyV')"></div>
    <div class="slider-row"><label>Translate X</label><input id="mTx" type="range" min="-2000" max="2000" value="0" step="1" oninput="manualAdjust()"><input id="mTxV" class="value-input" type="number" value="0" step="1" oninput="manualNumberAdjust('mTx','mTxV')"></div>
    <div class="slider-row"><label>HE opacity</label><input id="heOpacity" type="range" min="0" max="1" value="0.65" step="0.01" oninput="manualAdjust()"><input id="heOpacityV" class="value-input" type="number" min="0" max="1" value="0.65" step="0.01" oninput="opacityNumberAdjust()"></div>
    <div class="viewer">
      <div><b>Live Backend overlay/QC</b><br><canvas id="liveOverlay" class="mask-editor"></canvas></div>
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
let sessionId=null; let debounce=null; let liveDebounce=null; let liveInFlight=false; let liveQueued=false; let liveImages={overlay:null}; let batchId=null; let batchTimer=null;
let editors={};
function setStatus(x){ document.getElementById('status').textContent = x; }
async function api(path, body){ const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); return j; }
function img(name){ return `/api/file?session=${sessionId}&name=${name}&t=${Date.now()}`; }
function setBusy(id, on){ const el=document.getElementById(id); if(el) el.classList.toggle('active', on); document.querySelectorAll('button').forEach(b=>b.disabled=on); }
async function withBusy(id, label, fn){ setBusy(id,true); setStatus(label); try { const out=await fn(); return out; } catch(e) { setStatus('Error: '+e.message); throw e; } finally { setBusy(id,false); } }
function showProcessedImages(files){ if(files.includes('oct_raw_display_preview.png')) octRaw.src=img('oct_raw_display_preview.png'); if(files.includes('oct_flatfield_corrected_preview.png')) octFlat.src=img('oct_flatfield_corrected_preview.png'); if(files.includes('oct_tile_artifact_suppressed_preview.png')) octTile.src=img('oct_tile_artifact_suppressed_preview.png'); if(files.includes('oct_registered_preview.png')) octPre.src=img('oct_registered_preview.png'); if(files.includes('he_standardized_native_preview.png')) hePre.src=img('he_standardized_native_preview.png'); if(files.includes('he_black_white_input_preview.png')) heBW.src=img('he_black_white_input_preview.png'); if(files.includes('overlay_preview.png')) overlay.src=img('overlay_preview.png'); if(files.includes('registered_mask_preview.png')) maskReg.src=img('registered_mask_preview.png'); }
function fitSliderToValue(slider,value){ const n=parseFloat(value); if(!Number.isFinite(n))return; if(n<parseFloat(slider.min))slider.min=String(n); if(n>parseFloat(slider.max))slider.max=String(n); slider.value=String(n); }
function updateManualDisplays(){ mScaleV.value=mScale.value; mStretchXV.value=mStretchX.value; mStretchYV.value=mStretchY.value; mRotV.value=mRot.value; mTyV.value=mTy.value; mTxV.value=mTx.value; }
function manualNumberAdjust(sliderId,valueId){ const slider=document.getElementById(sliderId); const value=document.getElementById(valueId); fitSliderToValue(slider,value.value); manualAdjust(); }
function opacityNumberAdjust(){ const v=Math.max(0,Math.min(1,parseFloat(heOpacityV.value)||0)); heOpacityV.value=v; heOpacity.value=v; manualAdjust(); }
function hydrateManualControls(state){ const manual=(state&&state.transform_state&&state.transform_state.manual)||{}; fitSliderToValue(mScale,manual.scale ?? 1); fitSliderToValue(mStretchX,manual.stretch_x ?? 1); fitSliderToValue(mStretchY,manual.stretch_y ?? 1); fitSliderToValue(mRot,manual.rotation_deg ?? 0); fitSliderToValue(mTy,manual.translation_y ?? 0); fitSliderToValue(mTx,manual.translation_x ?? 0); const op=Math.max(0,Math.min(1,parseFloat(manual.he_opacity ?? 0.65))); heOpacity.value=op; heOpacityV.value=op; updateManualDisplays(); }
function hydrateSaveLinks(files){ const clean=['output/he_registered.tiff','output/oct_registered.tiff','output/registered_mask.tiff','alignment_summary.json'].filter(f=>files.includes(f)); saveLinks.innerHTML=clean.map(f=>`<div><a href="/api/file?session=${sessionId}&name=${f}" target="_blank">${f}</a></div>`).join(''); }
async function scanProcessedOutputs(){ await withBusy('busy-processed','Scanning processed outputs...', async()=>{ const j=await api('/api/processed_scan',{output_root:processedRoot.value}); processedSample.innerHTML = j.samples.length ? j.samples.map(s=>`<option value="${s.path}">${s.name}</option>`).join('') : '<option value="">No processed samples found</option>'; setStatus(`Found ${j.samples.length} processed sample(s).`); }); }
async function loadProcessedSample(){ await withBusy('busy-processed','Loading processed sample...', async()=>{ const j=await api('/api/processed_load',{sample_dir:processedSample.value}); sessionId=j.session_id; const files=j.files||[]; showProcessedImages(files); hydrateManualControls(j.state||{}); hydrateSaveLinks(files); if(files.includes('oct_mask_edit.png')&&files.includes('oct_mask_editor_base.png')) await loadCanvas('octCanvas','oct_mask_edit.png','oct_mask_editor_base.png',[0,255,70]); if(files.includes('he_mask_edit.png')&&files.includes('he_mask_editor_base.png')) await loadCanvas('heCanvas','he_mask_edit.png','he_mask_editor_base.png',[255,35,20]); if(files.includes('live_backend_qc_preview.png')) { try { await refreshReg(); } catch(_) {} } else { liveImages={overlay:null}; } const rebuilt=(j.rebuilt||[]).length ? `\nBackfilled missing reload files: ${j.rebuilt.join(', ')}.` : ''; setStatus(`Loaded processed sample: ${j.sample_name}\nOutput: ${j.output_dir}\nAll available previews, masks, overlays, save links, and manual controls were filled.\n${j.can_manual_adjust ? 'Manual adjustment is available.' : 'Manual adjustment needs transform_state.json from an interactive run.'}${rebuilt}`); }); }
function batchImg(record){ return `/api/batch_file?batch=${encodeURIComponent(batchId||'')}&case=${encodeURIComponent(record.case_id)}&name=${encodeURIComponent(record.overlay||'overlay_preview.png')}&t=${Date.now()}`; }
function updateBatchProgress(job){ const total=Number(job?.total||0); const completed=Number(job?.completed||0); const progress=document.getElementById('batchProgress'); const text=document.getElementById('batchProgressText'); if(progress){ progress.max=Math.max(total,1); progress.value=Math.min(completed,total); } if(text){ const remaining=Math.max(total-completed,0); text.textContent=`Detected ${total} sample${total===1?'':'s'}. Completed ${completed}. Remaining ${remaining}.`; } }
function renderBatch(job){ const status=document.getElementById('batchStatus'); const results=document.getElementById('batchResults'); if(!job){ updateBatchProgress(null); status.textContent='No batch run started.'; if(results)results.innerHTML=''; return; } updateBatchProgress(job); status.textContent=`Batch ${job.batch_id}: ${job.status}. ${job.completed}/${job.total} complete. Output: ${job.output_root}`; const cards=(job.results||[]).slice().sort((a,b)=>(a.case_id||'').localeCompare(b.case_id||'')); if(!cards.length){ results.innerHTML='<div class="small">Waiting for the first completed sample preview...</div>'; return; } results.innerHTML=cards.map(r=>{ const ok=r.status==='ok'; const deleted=r.status==='deleted'; const badge=deleted?'deleted':(ok?'complete':'failed'); const imgHtml=ok&&!deleted?`<img loading="lazy" decoding="async" src="${batchImg(r)}" alt="overlay preview for ${r.case_id}">`:`<div class="small">${r.error||'Registration failed. Check run.log in the output folder.'}</div>`; const checked=r.keep!==false&&!deleted?'checked':''; const disabled=deleted?'disabled':''; return `<div class="batch-card ${ok?'':'failed'} ${deleted?'deleted':''}"><div class="caption">${r.case_id}</div>${imgHtml}<div class="keep-row"><span class="pill ${ok?'':'failed'}">${badge}</span><label><input type="checkbox" data-case="${r.case_id}" ${checked} ${disabled}> keep</label></div><div class="small">${r.output_dir||''}</div></div>`; }).join(''); }
async function pollBatch(){ if(!batchId)return; const job=await api('/api/batch_status',{batch_id:batchId}); renderBatch(job); const running=job.status==='queued'||job.status==='running'; setBusy('busy-batch', running); if(running){ batchTimer=setTimeout(pollBatch,2500); } else { setStatus(`Batch ${batchId} complete. Review overlays and uncheck any outputs to delete.`); } }
async function startBatch(){ await withBusy('busy-batch','Starting batch registration...', async()=>{ const j=await api('/api/batch_start',{input_root:batchInput.value,output_root:batchOutput.value,workers:parseInt(batchWorkers.value||'1',10),overwrite:batchOverwrite.checked}); batchId=j.batch_id; renderBatch(j); clearTimeout(batchTimer); batchTimer=setTimeout(pollBatch,1000); setStatus(`Started batch ${batchId} with ${j.total} sample(s).`); }); }
async function applyBatchKeepChoices(){ if(!batchId){ setStatus('No batch run to finalize.'); return; } const keep={}; document.querySelectorAll('#batchResults input[data-case]').forEach(cb=>{ keep[cb.dataset.case]=cb.checked; }); const j=await api('/api/batch_apply_keep',{batch_id:batchId,keep}); renderBatch(j); setStatus(`Deleted ${j.deleted_count||0} unchecked output folder(s).`); }
async function loadPaths(){ await withBusy('busy-load','Loading paths...', async()=>{ const j=await api('/api/load_paths',{oct_path:octPath.value,he_path:hePath.value,output_root:singleOutput.value}); sessionId=j.session_id; setStatus(`Loaded session ${sessionId}\nOutput: ${j.output_dir}`); }); }
async function uploadFiles(){ await withBusy('busy-load','Uploading files...', async()=>{ const fd=new FormData(); fd.append('oct', octFile.files[0]); fd.append('he', heFile.files[0]); const r=await fetch('/api/upload',{method:'POST',body:fd}); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); sessionId=j.session_id; setStatus('Uploaded session '+sessionId); }); }
async function preprocess(){ await withBusy('busy-preprocess','Preprocessing OCT and HE...', async()=>{ const j=await api('/api/preprocess',{session_id:sessionId,stain_normalizer:stain.value}); octRaw.src=img('oct_raw_display_preview.png'); octFlat.src=img('oct_flatfield_corrected_preview.png'); octTile.src=img('oct_tile_artifact_suppressed_preview.png'); octPre.src=img('oct_registered_preview.png'); hePre.src=img('he_standardized_native_preview.png'); heBW.src=img('he_black_white_input_preview.png'); setStatus(JSON.stringify(j,null,2)); }); }
async function flipHe(){ await withBusy('busy-preprocess','Flipping HE left-right and refreshing preprocessing...', async()=>{ if(!sessionId) throw new Error('Load OCT and HE images first.'); const j=await api('/api/flip_he',{session_id:sessionId,stain_normalizer:stain.value}); octRaw.src=img('oct_raw_display_preview.png'); octFlat.src=img('oct_flatfield_corrected_preview.png'); octTile.src=img('oct_tile_artifact_suppressed_preview.png'); octPre.src=img('oct_registered_preview.png'); hePre.src=img('he_standardized_native_preview.png'); heBW.src=img('he_black_white_input_preview.png'); liveImages={overlay:null}; overlay.removeAttribute('src'); maskReg.removeAttribute('src'); setStatus(`HE image mirrored left-right. Rerun background removal and registration before saving.\nFlipped HE path: ${j.he_path}`); }); }
async function runAllProcessing(){ await withBusy('busy-preprocess','Running full pipeline...', async()=>{ if(!sessionId) throw new Error('Load OCT and HE images first.'); let j=await api('/api/preprocess',{session_id:sessionId,stain_normalizer:stain.value}); octRaw.src=img('oct_raw_display_preview.png'); octFlat.src=img('oct_flatfield_corrected_preview.png'); octTile.src=img('oct_tile_artifact_suppressed_preview.png'); octPre.src=img('oct_registered_preview.png'); hePre.src=img('he_standardized_native_preview.png'); heBW.src=img('he_black_white_input_preview.png'); j=await api('/api/masks',{session_id:sessionId,he_mask_mode:heMaskMode.value,he_gray_percentile:parseFloat(hePct.value)}); await loadCanvas('octCanvas','oct_mask_edit.png','oct_mask_editor_base.png',[0,255,70]); await loadCanvas('heCanvas','he_mask_edit.png','he_mask_editor_base.png',[255,35,20]); await saveMasks(); j=await api('/api/autoreg',{session_id:sessionId}); await refreshReg(); j=await api('/api/save',{session_id:sessionId}); saveLinks.innerHTML = j.files.map(f=>`<div><a href="/api/file?session=${sessionId}&name=${f}" target="_blank">${f}</a></div>`).join(''); setStatus(`Full processing complete.\nOutput: ${j.output_dir}`); }); }
async function removeBackground(){ await withBusy('busy-mask','Removing background...', async()=>{ const j=await api('/api/masks',{session_id:sessionId,he_mask_mode:heMaskMode.value,he_gray_percentile:parseFloat(hePct.value)}); await loadCanvas('octCanvas','oct_mask_edit.png','oct_mask_editor_base.png',[0,255,70]); await loadCanvas('heCanvas','he_mask_edit.png','he_mask_editor_base.png',[255,35,20]); setStatus(JSON.stringify(j,null,2)); }); }
function imageLoad(src){ return new Promise((res,rej)=>{ const im=new Image(); im.onload=()=>res(im); im.onerror=rej; im.src=src; }); }
async function jsonFile(name){ const r=await fetch(img(name)); if(!r.ok) throw new Error(`Could not load ${name}`); return await r.json(); }
async function loadCanvas(id, maskName, baseName, color){ const c=document.getElementById(id); const base=await imageLoad(img(baseName)); const mask=await imageLoad(img(maskName)); c.width=base.width; c.height=base.height; const maskCanvas=document.createElement('canvas'); maskCanvas.width=base.width; maskCanvas.height=base.height; const mctx=maskCanvas.getContext('2d'); mctx.drawImage(mask,0,0,base.width,base.height); editors[id]={base,maskCanvas,maskCtx:mctx,color,history:[]}; redrawEditor(id); setupDraw(c); }
function redrawEditor(id){ const c=document.getElementById(id); const ctx=c.getContext('2d'); const ed=editors[id]; if(!ed)return; ctx.clearRect(0,0,c.width,c.height); ctx.drawImage(ed.base,0,0,c.width,c.height); const maskData=ed.maskCtx.getImageData(0,0,c.width,c.height); const overlay=ctx.getImageData(0,0,c.width,c.height); for(let i=0;i<maskData.data.length;i+=4){ if(maskData.data[i]>127){ overlay.data[i]=Math.round(overlay.data[i]*0.55+ed.color[0]*0.45); overlay.data[i+1]=Math.round(overlay.data[i+1]*0.55+ed.color[1]*0.45); overlay.data[i+2]=Math.round(overlay.data[i+2]*0.55+ed.color[2]*0.45); }} ctx.putImageData(overlay,0,0); }
function setupDraw(c){ let down=false; function paintAt(e){ e.preventDefault(); const ed=editors[c.id]; if(!ed)return; const r=c.getBoundingClientRect(); const x=(e.clientX-r.left)*c.width/r.width, y=(e.clientY-r.top)*c.height/r.height; ed.maskCtx.fillStyle=paint.value==='white'?'white':'black'; ed.maskCtx.beginPath(); ed.maskCtx.arc(x,y,parseFloat(brush.value),0,Math.PI*2); ed.maskCtx.fill(); redrawEditor(c.id); } c.onpointerdown=e=>{ const ed=editors[c.id]; if(ed){ ed.history.push(ed.maskCtx.getImageData(0,0,c.width,c.height)); if(ed.history.length>25)ed.history.shift(); } down=true; c.setPointerCapture(e.pointerId); paintAt(e)}; c.onpointermove=e=>{ if(down)paintAt(e); }; c.onpointerup=e=>{down=false; try{c.releasePointerCapture(e.pointerId)}catch(_){}}; c.onpointercancel=()=>{down=false}; }
function undoEdit(id){ const ed=editors[id]; if(!ed||!ed.history.length){ setStatus('Nothing to undo for '+id); return; } ed.maskCtx.putImageData(ed.history.pop(),0,0); redrawEditor(id); setStatus('Undid last edit for '+id); }
function canvasData(id){ const ed=editors[id]; return ed ? ed.maskCanvas.toDataURL('image/png') : document.getElementById(id).toDataURL('image/png'); }
async function saveMasks(){ const j=await api('/api/save_masks',{session_id:sessionId,oct_mask:canvasData('octCanvas'),he_mask:canvasData('heCanvas')}); setStatus(JSON.stringify(j,null,2)); }
async function autoRegister(){ await withBusy('busy-autoreg','Running auto registration...', async()=>{ await saveMasks(); const j=await api('/api/autoreg',{session_id:sessionId}); await refreshReg(); setStatus(JSON.stringify(j.auto_params,null,2)); }); }
function manualPayload(){ return {session_id:sessionId,scale:parseFloat(mScale.value),stretch_x:parseFloat(mStretchX.value),stretch_y:parseFloat(mStretchY.value),rotation_deg:parseFloat(mRot.value),translation_y:parseFloat(mTy.value),translation_x:parseFloat(mTx.value),he_opacity:parseFloat(heOpacity.value)}; }
function manualAdjust(){ updateManualDisplays(); heOpacityV.value=heOpacity.value; scheduleLivePreview(); clearTimeout(debounce); debounce=setTimeout(async()=>{ if(!sessionId)return; setStatus('Manual adjustment previewed. Click Save Final to write native-resolution outputs.'); },650); }
async function refreshReg(){ overlay.src=img('overlay_preview.png'); maskReg.src=img('registered_mask_preview.png'); liveImages.overlay=await imageLoad(img('live_backend_qc_preview.png')); initLiveCanvas(); drawLiveOverlay(); }
function initLiveCanvas(){ if(!liveImages.overlay)return; liveOverlay.width=liveImages.overlay.width; liveOverlay.height=liveImages.overlay.height; }
function drawLiveOverlay(){ heOpacityV.value=heOpacity.value; if(!liveImages.overlay)return; const c=liveOverlay, ctx=c.getContext('2d'); if(c.width!==liveImages.overlay.width){initLiveCanvas();} ctx.clearRect(0,0,c.width,c.height); ctx.drawImage(liveImages.overlay,0,0,c.width,c.height); }
function scheduleLivePreview(){ liveQueued=true; clearTimeout(liveDebounce); liveDebounce=setTimeout(runLivePreview,220); }
async function runLivePreview(){ if(!sessionId||liveInFlight)return; if(!liveQueued)return; liveQueued=false; liveInFlight=true; try{ await api('/api/live_preview',manualPayload()); liveImages.overlay=await imageLoad(img('live_backend_qc_preview.png')); initLiveCanvas(); drawLiveOverlay(); } catch(e){ setStatus('Live preview error: '+e.message); } finally { liveInFlight=false; if(liveQueued){ clearTimeout(liveDebounce); liveDebounce=setTimeout(runLivePreview,80); } } }
async function saveFinal(){ await withBusy('busy-save','Saving final outputs...', async()=>{ if(sessionId) await api('/api/manual',manualPayload()); const j=await api('/api/save',{session_id:sessionId}); await refreshReg(); saveLinks.innerHTML = j.files.map(f=>`<div><a href="/api/file?session=${sessionId}&name=${f}" target="_blank">${f}</a></div>`).join(''); setStatus(JSON.stringify(j,null,2)); }); }
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
                _send_file(self, paths.root, name)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 404)
            return
        if parsed.path == "/api/batch_file":
            qs = parse_qs(parsed.query)
            batch_id = qs.get("batch", [""])[0]
            case_id = qs.get("case", [""])[0]
            name = qs.get("name", [""])[0]
            try:
                _send_file(self, _batch_output_dir(batch_id, case_id), name)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, 404)
            return
        _json_response(self, {"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/processed_scan":
                payload = _read_json(self)
                output_root = _user_path(payload["output_root"])
                samples = _scan_processed_outputs(output_root)
                _json_response(self, {"output_root": str(output_root), "samples": samples})
            elif parsed.path == "/api/processed_load":
                payload = _read_json(self)
                sample_dir = _user_path(payload["sample_dir"])
                if not sample_dir.exists() or not sample_dir.is_dir():
                    raise FileNotFoundError("Processed sample folder does not exist")
                oct_path, he_path = _paths_from_processed_output(sample_dir)
                rebuilt = _ensure_interactive_artifacts(SessionPaths(sample_dir, oct_path, he_path))
                session_id, alias_root = _unique_session_root(f"loaded_{_slug(sample_dir.name)}")
                _write_session_alias(alias_root, sample_dir, oct_path, he_path)
                files = _known_output_files(sample_dir)
                _json_response(
                    self,
                    {
                        "session_id": session_id,
                        "sample_name": sample_dir.name,
                        "output_dir": str(sample_dir),
                        "files": files,
                        "state": _processed_state(sample_dir),
                        "can_manual_adjust": (sample_dir / "transform_state.json").exists(),
                        "rebuilt": rebuilt,
                    },
                )
            elif parsed.path == "/api/batch_start":
                payload = _read_json(self)
                input_root = _user_path(payload["input_root"])
                overwrite = bool(payload.get("overwrite", False))
                workers = max(1, min(16, int(payload.get("workers", 4))))
                if not input_root.exists() or not input_root.is_dir():
                    raise FileNotFoundError("Batch input folder does not exist")
                requested_output_root = _clean_user_path_value(payload.get("output_root", ""))
                if requested_output_root:
                    batch_root = _user_path(requested_output_root)
                    batch_id = _slug(batch_root.name)
                else:
                    batch_base = f"batch_{_slug(input_root.name)}_{time.strftime('%Y%m%d_%H%M%S')}"
                    batch_id, batch_root = _unique_session_root(batch_base)
                batch_root.mkdir(parents=True, exist_ok=True)
                cases = _find_batch_pairs(input_root, batch_root, overwrite=overwrite)
                if not cases:
                    raise ValueError("No OCT/HE sample pairs were found in the input folder")
                job = {
                    "batch_id": batch_id,
                    "status": "queued",
                    "input_root": str(input_root),
                    "output_root": str(batch_root),
                    "workers": workers,
                    "overwrite": overwrite,
                    "total": len(cases),
                    "completed": 0,
                    "results": [],
                    "started_at": time.time(),
                    "cases": cases,
                }
                with BATCH_LOCK:
                    BATCH_JOBS[batch_id] = job
                thread = threading.Thread(target=_run_batch_job, args=(batch_id, cases, workers), daemon=True)
                thread.start()
                _json_response(self, _public_batch_job(batch_id))
            elif parsed.path == "/api/batch_status":
                payload = _read_json(self)
                _json_response(self, _public_batch_job(payload["batch_id"]))
            elif parsed.path == "/api/batch_apply_keep":
                payload = _read_json(self)
                batch_id = payload["batch_id"]
                keep = payload.get("keep", {})
                deleted_count = 0
                with BATCH_LOCK:
                    if batch_id not in BATCH_JOBS:
                        raise FileNotFoundError(f"Unknown batch: {batch_id}")
                    for result in BATCH_JOBS[batch_id].get("results", []):
                        case_id = result.get("case_id")
                        should_keep = bool(keep.get(case_id, result.get("keep", True)))
                        result["keep"] = should_keep
                        if not should_keep and result.get("status") != "deleted":
                            output_dir = Path(result["output_dir"])
                            if output_dir.exists():
                                shutil.rmtree(output_dir)
                                deleted_count += 1
                            result["status"] = "deleted"
                response = _public_batch_job(batch_id)
                response["deleted_count"] = deleted_count
                _json_response(self, response)
            elif parsed.path == "/api/upload":
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
                oct_item = form["oct"]
                he_item = form["he"]
                oct_filename = Path(oct_item.filename or "oct.tiff").name
                he_filename = Path(he_item.filename or "he.tiff").name
                output_parent = _specified_output_root(None)
                session_id, root = _session_root_for_output(output_parent, _session_id_from_paths(Path(oct_filename), Path(he_filename)))
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
                oct_path = _user_path(payload["oct_path"])
                he_path = _user_path(payload["he_path"])
                if not oct_path.exists() or not he_path.exists():
                    raise FileNotFoundError("OCT or HE path does not exist")
                output_parent = _specified_output_root(payload.get("output_root"))
                session_id, root = _session_root_for_output(output_parent, _session_id_from_paths(oct_path, he_path))
                _write_session(root, oct_path, he_path)
                _json_response(self, {"session_id": session_id, "output_dir": str(root)})
            elif parsed.path == "/api/preprocess":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                state = _prepare_state(paths, payload.get("stain_normalizer", "torchstain_reinhard"), payload.get("stain_reference_he"))
                _json_response(self, {"ok": True, "state": state, "output_dir": str(paths.root)})
            elif parsed.path == "/api/flip_he":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                result = _flip_he_for_session(payload["session_id"], paths, payload.get("stain_normalizer", "torchstain_reinhard"))
                _json_response(self, result)
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
            elif parsed.path == "/api/live_preview":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                manual = _manual_from_payload(payload)
                _render_live_qc_preview(paths, manual)
                _json_response(self, {"ok": True, "preview": "live_backend_qc_preview.png"})
            elif parsed.path == "/api/manual":
                payload = _read_json(self)
                paths = _session(payload["session_id"])
                state = json.loads((paths.root / "transform_state.json").read_text())
                state["manual"] = _manual_from_payload(payload)
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
                summary.update(_cache_input_images(paths.root, paths.oct_path, paths.he_path))
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
