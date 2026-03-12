import cv2
import numpy as np

from constants import *
from unidepth_v2 import UniDepthV2
from utils import (
    colorize_depth_map,
    compute_center_depth,
    ensure_parent_dir,
    resolve_existing_image_path,
    resolve_repo_path,
)


def main() -> None:
    image_path = resolve_existing_image_path(
        DEPTH_IMAGE_INPUT_PATH,
        DEPTH_IMAGE_FALLBACK_EXTENSIONS,
    )

    output_npy = resolve_repo_path(DEPTH_IMAGE_OUTPUT_NPY_PATH)
    output_vis = resolve_repo_path(DEPTH_IMAGE_OUTPUT_VIS_PATH)
    ensure_parent_dir(output_npy)
    ensure_parent_dir(output_vis)

    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    model = UniDepthV2(resolution_level=DEPTH_RESOLUTION_LEVEL)
    depth_tensor, intrinsics = model(frame_rgb)
    depth_map = depth_tensor.detach().cpu().numpy().squeeze().astype(np.float32)

    center_depth = compute_center_depth(depth_map, DEPTH_CENTER_PATCH_SIZE)
    depth_min = float(np.nanmin(depth_map))
    depth_max = float(np.nanmax(depth_map))

    np.save(output_npy, depth_map)
    depth_vis = colorize_depth_map(
        depth_map,
        DEPTH_COLORMAP,
        invert_colormap=DEPTH_INVERT_COLORMAP,
    )
    cv2.imwrite(str(output_vis), depth_vis)

    print("Depth inference complete (single image).")
    print(f"- image: {image_path}")
    print(f"- depth shape: {depth_map.shape}")
    print(f"- depth range: [{depth_min:.4f}, {depth_max:.4f}]")
    print(f"- center depth (median {DEPTH_CENTER_PATCH_SIZE}x{DEPTH_CENTER_PATCH_SIZE}): {center_depth:.4f}")
    print(f"- intrinsics shape: {tuple(intrinsics.shape)}")
    print(f"- depth npy: {output_npy}")
    print(f"- depth visualization: {output_vis}")


if __name__ == "__main__":
    main()
