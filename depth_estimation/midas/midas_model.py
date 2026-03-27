import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from pathlib import Path
import sys


class MiDaSModel:
    def __init__(self, model_type: str = "DPT_Hybrid", device: str = "auto"):
        self.model_type = model_type
        self.device = self._resolve_device(device)

        self.model = self._hub_load(self.model_type)
        self.model.to(self.device)
        self.model.eval()

        transforms = self._hub_load("transforms")
        self.transform = self._select_transform(transforms, self.model_type)

        print(f"Loaded MiDaS model '{self.model_type}' on device '{self.device}'.")

    def _find_cached_midas_repo(self) -> Path | None:
        hub_dir = Path(torch.hub.get_dir())
        if not hub_dir.exists():
            return None

        candidates = sorted(
            hub_dir.glob("intel-isl_MiDaS_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for repo_dir in candidates:
            if (repo_dir / "hubconf.py").exists() and (repo_dir / "midas").is_dir():
                return repo_dir
        return None

    def _add_midas_repo_to_syspath(self) -> Path | None:
        repo_dir = self._find_cached_midas_repo()
        if repo_dir is None:
            return None

        repo_str = str(repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        return repo_dir

    @contextmanager
    def _without_local_midas_shadow(self):
        local_depth_estimation_dir = str(Path(__file__).resolve().parents[1])
        removed_entries = [p for p in sys.path if p == local_depth_estimation_dir]
        if removed_entries:
            sys.path[:] = [p for p in sys.path if p != local_depth_estimation_dir]
        try:
            yield
        finally:
            for p in reversed(removed_entries):
                sys.path.insert(0, p)

    def _drop_conflicting_midas_modules(self) -> None:
        local_midas_dir = str(Path(__file__).resolve().parent)
        to_remove: list[str] = []

        for module_name, module in list(sys.modules.items()):
            if module_name != "midas" and not module_name.startswith("midas."):
                continue

            module_file = getattr(module, "__file__", None)
            module_path = getattr(module, "__path__", None)

            if module_file and str(module_file).startswith(local_midas_dir):
                to_remove.append(module_name)
                continue

            if module_path:
                for entry in module_path:
                    if str(entry).startswith(local_midas_dir):
                        to_remove.append(module_name)
                        break

        for module_name in to_remove:
            sys.modules.pop(module_name, None)

    def _hub_load(self, model_name: str, **kwargs):
        self._add_midas_repo_to_syspath()
        self._drop_conflicting_midas_modules()

        load_kwargs = dict(
            repo_or_dir="intel-isl/MiDaS",
            model=model_name,
            trust_repo=True,
            force_reload=False,
        )
        load_kwargs.update(kwargs)

        try:
            with self._without_local_midas_shadow():
                return torch.hub.load(**load_kwargs)
        except ModuleNotFoundError as exc:
            if exc.name not in {"midas", "midas.dpt_depth"}:
                raise

            repo_dir = self._add_midas_repo_to_syspath()
            if repo_dir is None:
                raise RuntimeError(
                    "MiDaS cache repo was not found under torch.hub directory."
                ) from exc

            self._drop_conflicting_midas_modules()
            with self._without_local_midas_shadow():
                return torch.hub.load(**load_kwargs)

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
