from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def resolve_existing_image_path(
    path_like: str,
    fallback_extensions: tuple[str, ...] | list[str],
) -> Path:
    requested = resolve_repo_path(path_like)
    if requested.exists():
        return requested

    candidates: list[Path] = []
    stem_path = requested.with_suffix("") if requested.suffix else requested
    for ext in fallback_extensions:
        norm = ext if ext.startswith(".") else f".{ext}"
        candidates.append(stem_path.with_suffix(norm.lower()))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(p) for p in [requested, *candidates])
    raise RuntimeError(
        f"Image not found. Tried: {tried}\n"
        "Set DEPTH_IMAGE_INPUT_PATH in depth_estimation/constants.py."
    )


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def compute_center_depth(depth_map: np.ndarray, patch_size: int) -> float:
    h, w = depth_map.shape
    cy, cx = h // 2, w // 2

    patch = max(1, int(patch_size))
    if patch % 2 == 0:
        patch += 1
    half = patch // 2

    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)

    center_patch = depth_map[y0:y1, x0:x1]
    finite = center_patch[np.isfinite(center_patch)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _normalize_depth_to_uint8(depth_map: np.ndarray) -> np.ndarray:
    finite = depth_map[np.isfinite(depth_map)]
    if finite.size == 0:
        return np.zeros(depth_map.shape, dtype=np.uint8)

    d_min = float(np.min(finite))
    d_max = float(np.max(finite))
    denom = max(1e-6, d_max - d_min)
    normalized = np.clip((depth_map - d_min) / denom, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _resolve_colormap_code(colormap_name: str) -> int:
    import cv2

    lookup = {
        "turbo": cv2.COLORMAP_TURBO,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    return lookup.get(colormap_name.lower(), cv2.COLORMAP_TURBO)


def colorize_depth_map(depth_map: np.ndarray, colormap_name: str) -> np.ndarray:
    import cv2

    colormap = _resolve_colormap_code(colormap_name)
    depth_u8 = _normalize_depth_to_uint8(depth_map)
    return cv2.applyColorMap(depth_u8, colormap)


def resize_depth_to_frame(depth_map: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    if depth_map.shape[1] == width and depth_map.shape[0] == height:
        return depth_map
    return cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
