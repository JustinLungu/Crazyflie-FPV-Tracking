import torch
from PIL import Image
import numpy as np
import math

class UniDepthV2(torch.nn.Module):
    def __init__(self, resolution_level=None):
        super(UniDepthV2, self).__init__()
        self.model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version="v2", backbone="vitb14", pretrained=True, trust_repo=True, force_reload=False)
        self.model = self.model.eval()
        self.model = self.model.to("cuda")

        # Set resolution_level if provided (range: [0, 10))
        if resolution_level is not None:
            if 0 <= resolution_level < 10:
                self.model.resolution_level = resolution_level
                print(f"Set resolution_level to {resolution_level}")
            else:
                print(f"Warning: resolution_level must be in [0, 10), got {resolution_level}")

        # Apply padding bug fix (GitHub issue #139)
        self._patch_padding_function()

    def _patch_padding_function(self):
        """
        Patch the UniDepth model's padding function to fix dimension collapse bug.

        Issue: https://github.com/lpiccinelli-eth/UniDepth/issues/139
        The original padding function can produce negative padding values for certain
        aspect ratios, causing tensor dimensions to collapse.

        The bug is in line 47 and 53 of unidepthv2.py where int() is used instead of
        math.ceil(), causing dimensions to shrink rather than expand.
        """
        import sys

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

        # Monkey-patch the module-level function
        # The function is in unidepth.models.unidepthv2.unidepthv2 module
        try:
            import unidepth.models.unidepthv2.unidepthv2 as unidepthv2_module
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