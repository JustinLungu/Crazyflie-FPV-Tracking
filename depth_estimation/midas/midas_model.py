import numpy as np
import torch
import torch.nn.functional as F


class MiDaSModel:
    def __init__(self, model_type: str = "DPT_Hybrid", device: str = "auto"):
        self.model_type = model_type
        self.device = self._resolve_device(device)

        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self._select_transform(transforms, self.model_type)

        print(f"Loaded MiDaS model '{self.model_type}' on device '{self.device}'.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        if device == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                print("Warning: MPS requested but not available. Falling back to CPU.")
                return "cpu"

        return device

    @staticmethod
    def _select_transform(transforms, model_type: str):
        dpt_models = {
            "DPT_Large",
            "DPT_Hybrid",
            "DPT_BEiT_L_512",
            "DPT_BEiT_L_384",
            "DPT_BEiT_B_384",
            "DPT_SwinV2_L_384",
            "DPT_SwinV2_B_384",
            "DPT_SwinV2_T_256",
            "DPT_LeViT_224",
        }
        if model_type in dpt_models:
            return transforms.dpt_transform
        return transforms.small_transform

    def predict(self, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("Expected RGB frame with shape (H, W, 3).")

        input_batch = self.transform(frame_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        depth_map = prediction.squeeze().detach().cpu().numpy().astype(np.float32)
        return depth_map
