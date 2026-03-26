import cv2
import numpy as np

from constants import *
from midas_model import MiDaSModel
from utils import (
    colorize_depth_map,
    compute_center_depth,
    resolve_existing_image_path,
    resolve_repo_path,
)


def main() -> None:
    image_path = resolve_existing_image_path(
        MIDAS_IMAGE_INPUT_PATH,
        MIDAS_IMAGE_FALLBACK_EXTENSIONS,
    )

    output_dir = resolve_repo_path(MIDAS_IMAGE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = image_path.stem
    output_npy = output_dir / f"{image_stem}{MIDAS_IMAGE_OUTPUT_NPY_SUFFIX}"
    output_vis = output_dir / f"{image_stem}{MIDAS_IMAGE_OUTPUT_VIS_SUFFIX}"

    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    model = MiDaSModel(model_type=MIDAS_MODEL_TYPE, device=MIDAS_DEVICE)
    depth_map = model.predict(frame_rgb)

    center_depth = compute_center_depth(depth_map, MIDAS_CENTER_PATCH_SIZE)
    depth_min = float(np.nanmin(depth_map))
    depth_max = float(np.nanmax(depth_map))

    np.save(output_npy, depth_map)
    depth_vis = colorize_depth_map(
        depth_map,
        MIDAS_COLORMAP,
        invert_colormap=MIDAS_INVERT_COLORMAP,
    )
    cv2.imwrite(str(output_vis), depth_vis)

    print("Depth inference complete (single image, MiDaS).")
    print(f"- image: {image_path}")
    print(f"- depth shape: {depth_map.shape}")
    print(f"- depth range: [{depth_min:.4f}, {depth_max:.4f}]")
    print(f"- center depth (median {MIDAS_CENTER_PATCH_SIZE}x{MIDAS_CENTER_PATCH_SIZE}): {center_depth:.4f}")
    print(f"- depth npy: {output_npy}")
    print(f"- depth visualization: {output_vis}")


if __name__ == "__main__":
    main()
