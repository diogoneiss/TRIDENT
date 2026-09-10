"""Hardware and library descriptors recorded as MLflow tags."""

import torch


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
    }
    if device.type == "cuda":
        tags["gpu_name"] = torch.cuda.get_device_name(device)
    return tags
