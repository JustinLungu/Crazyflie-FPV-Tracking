import torch
from PIL import Image
import numpy as np
import math
from pathlib import Path
import sys
from contextlib import contextmanager
import importlib

class UniDepthV2(torch.nn.Module):
    def __init__(self, resolution_level=None):
        super(UniDepthV2, self).__init__()
        self.model = self._load_unidepth_model()
        self.model = self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            print("CUDA not available for UniDepth. Falling back to CPU.")
        self.model = self.model.to(self.device)

        # Set resolution_level if provided (range: [0, 10))
        if resolution_level is not None:
            if 0 <= resolution_level < 10:
                self.model.resolution_level = resolution_level
                print(f"Set resolution_level to {resolution_level}")
            else:
                print(f"Warning: resolution_level must be in [0, 10), got {resolution_level}")

        # Apply padding bug fix (GitHub issue #139)
        self._patch_padding_function()

    def _find_cached_unidepth_repo(self) -> Path | None:
        hub_dir = Path(torch.hub.get_dir())
        if not hub_dir.exists():
            return None

        candidates = sorted(
            hub_dir.glob("lpiccinelli-eth_UniDepth_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for repo_dir in candidates:
            if (repo_dir / "hubconf.py").exists() and (repo_dir / "unidepth").is_dir():
                return repo_dir
        return None

    def _add_unidepth_repo_to_syspath(self) -> Path | None:
        repo_dir = self._find_cached_unidepth_repo()
        if repo_dir is None:
            return None

        repo_str = str(repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        return repo_dir

    @contextmanager
    def _without_local_unidepth_shadow(self):
        # Running `python depth_estimation/live_depth_estimation.py` places
        # `<repo>/depth_estimation` on sys.path, which exposes our local
        # package `unidepth` and shadows torch-hub's repo package name.
        local_depth_estimation_dir = str(Path(__file__).resolve().parents[1])
        removed_entries = [p for p in sys.path if p == local_depth_estimation_dir]
        if removed_entries:
            sys.path[:] = [p for p in sys.path if p != local_depth_estimation_dir]
        try:
            yield
        finally:
            for p in reversed(removed_entries):
                sys.path.insert(0, p)

    def _drop_conflicting_unidepth_modules(self) -> None:
        local_unidepth_dir = str(Path(__file__).resolve().parent)
        to_remove: list[str] = []

        for module_name, module in list(sys.modules.items()):
            if module_name != "unidepth" and not module_name.startswith("unidepth."):
                continue

            module_file = getattr(module, "__file__", None)
            module_path = getattr(module, "__path__", None)

            if module_file and str(module_file).startswith(local_unidepth_dir):
                to_remove.append(module_name)
                continue

            if module_path:
                for entry in module_path:
                    if str(entry).startswith(local_unidepth_dir):
                        to_remove.append(module_name)
                        break

        for module_name in to_remove:
            sys.modules.pop(module_name, None)

    def _load_unidepth_model(self):
        # If cache already exists, add it before torch.hub import of hubconf.py.
        self._add_unidepth_repo_to_syspath()
        self._drop_conflicting_unidepth_modules()

        load_kwargs = dict(
            repo_or_dir="lpiccinelli-eth/UniDepth",
            model="UniDepth",
            version="v2",
            backbone="vitb14",
            pretrained=True,
            trust_repo=True,
            force_reload=False,
        )

        try:
            with self._without_local_unidepth_shadow():
                return torch.hub.load(**load_kwargs)
        except ModuleNotFoundError as exc:
            # Some torch hub environments do not expose the cached repo on sys.path.
            if exc.name != "unidepth" and exc.name != "unidepth.models":
                raise

            repo_dir = self._add_unidepth_repo_to_syspath()
            if repo_dir is None:
                raise RuntimeError(
                    "UniDepth cache repo was not found under torch.hub directory. "
                    "Try running once with network access or clear torch cache and retry."
                ) from exc

            self._drop_conflicting_unidepth_modules()
            with self._without_local_unidepth_shadow():
                return torch.hub.load(**load_kwargs)

    def _patch_padding_function(self):
        """
        Patch the UniDepth model's padding function to fix dimension collapse bug.

        Issue: https://github.com/lpiccinelli-eth/UniDepth/issues/139
        The original padding function can produce negative padding values for certain
        aspect ratios, causing tensor dimensions to collapse.

        The bug is in line 47 and 53 of unidepthv2.py where int() is used instead of
        math.ceil(), causing dimensions to shrink rather than expand.
        """
        def fixed_get_paddings(original_shape, aspect_ratio_range):
            """
            Fixed padding calculation that prevents negative padding values.

            Args:
                original_shape: (H, W) tuple
                aspect_ratio_range: (min_ratio, max_ratio) tuple

            Returns:
                padding: (left, right, top, bottom) padding values (all non-negative)
                new_shape: (H_new, W_new) tuple
            """
            H_ori, W_ori = original_shape
            orig_aspect_ratio = W_ori / H_ori

            # Determine the closest aspect ratio within the range
            min_ratio, max_ratio = aspect_ratio_range
            target_aspect_ratio = min(max_ratio, max(min_ratio, orig_aspect_ratio))

            if orig_aspect_ratio > target_aspect_ratio:  # Too wide
                W_new = W_ori
                # FIX: Use ceil instead of int to ensure H_new >= H_ori
                H_new = math.ceil(W_ori / target_aspect_ratio)
                pad_top = (H_new - H_ori) // 2
                pad_bottom = H_new - H_ori - pad_top
                pad_left, pad_right = 0, 0
            else:  # Too tall
                H_new = H_ori
                # FIX: Use ceil instead of int to ensure W_new >= W_ori
                W_new = math.ceil(H_ori * target_aspect_ratio)
                pad_left = (W_new - W_ori) // 2
                pad_right = W_new - W_ori - pad_left
                pad_top, pad_bottom = 0, 0

            # Ensure all padding values are non-negative
            assert pad_left >= 0 and pad_right >= 0 and pad_top >= 0 and pad_bottom >= 0, \
                f"Negative padding detected: ({pad_left}, {pad_right}, {pad_top}, {pad_bottom})"

            return (pad_left, pad_right, pad_top, pad_bottom), (H_new, W_new)

        # Monkey-patch the module-level function on the already-loaded module.
        try:
            module_name = self.model.__class__.__module__
            unidepthv2_module = sys.modules.get(module_name)
            if unidepthv2_module is None:
                unidepthv2_module = importlib.import_module(module_name)
            unidepthv2_module.get_paddings = fixed_get_paddings
            print("  Applied padding bug fix (GitHub issue #139)")
        except (ImportError, AttributeError) as e:
            print(f"  Warning: Could not patch get_paddings function: {e}")

    def forward(self, image):
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = torch.Tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2)
        else:
            image = torch.Tensor(image)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
        with torch.no_grad():
            out = self.model.infer(image)
            return out["depth"], out["intrinsics"]
    



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UniDepthV2 inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="depth_output.npy", help="Path to save depth map")
    parser.add_argument("--resolution-level", type=float, default=None, help="Resolution level [0, 10) for speed/detail tradeoff")
    parser.add_argument("--bgr", action="store_true", help="Convert BGR to RGB (for images saved with OpenCV)")
    args = parser.parse_args()

    # Load image from disk
    print(f"Loading image from {args.image}")
    image = Image.open(args.image)

    # Convert BGR to RGB if needed
    if args.bgr:
        print("Converting BGR to RGB")
        image_np = np.array(image)
        image_np = image_np.transpose(1,0,2)  # Flip the color channels
        image = Image.fromarray(image_np)

    width, height = image.size

    # Run inference
    model = UniDepthV2(resolution_level=args.resolution_level)
    depth, intrinsics = model(image)

    # Save depth to disk
    depth_np = depth.cpu().numpy().squeeze()
    np.save(args.output, depth_np)
    print(f"Depth map saved to {args.output}")
    print(f"Depth shape: {depth_np.shape}")
    print(f"Depth range: [{depth_np.min():.3f}, {depth_np.max():.3f}]")

    # Print camera intrinsics and FOV
    print(f"\nCamera Intrinsics:")
    print(f"  Intrinsics tensor shape: {intrinsics.shape}")
    print(f"  Intrinsics:\n{intrinsics}")

    # Extract focal lengths (assuming intrinsics is [fx, fy, cx, cy] or 3x3 matrix)
    intrinsics_np = intrinsics.cpu().numpy().squeeze()
    if intrinsics_np.shape == (4,):
        fx, fy, cx, cy = intrinsics_np
    elif intrinsics_np.shape == (3, 3):
        fx = intrinsics_np[0, 0]
        fy = intrinsics_np[1, 1]
        cx = intrinsics_np[0, 2]
        cy = intrinsics_np[1, 2]
    else:
        print(f"Unknown intrinsics format: {intrinsics_np.shape}")
        fx = fy = None

    if fx is not None:
        # Calculate FOV in degrees
        fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
        print(f"\nCamera Field of View:")
        print(f"  Horizontal FOV: {fov_x:.2f}°")
        print(f"  Vertical FOV: {fov_y:.2f}°")
        print(f"  Focal length (fx, fy): ({fx:.2f}, {fy:.2f})")
        print(f"  Principal point (cx, cy): ({cx:.2f}, {cy:.2f})")
