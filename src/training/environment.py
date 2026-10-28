"""Hardware and library descriptors recorded as MLflow tags."""

import socket
import sys

import torch

from src.mlflow_utils import HOST_TAG


def runtime_environment_tags(device: torch.device) -> dict[str, str]:
    """Describe where a run trains so its timings are comparable across machines.

    Every key is always present so the comparison table has a stable column
    set; ``gpu_name`` and ``cuda_version`` are ``"none"`` on CPU-only runs.
    """
    tags = {
        "device": device.type,
        "gpu_name": "none",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        HOST_TAG: "windows" if sys.platform == "win32" else socket.gethostname(),
    }
    if device.type == "cuda":
        tags["gpu_name"] = torch.cuda.get_device_name(device)
    return tags
