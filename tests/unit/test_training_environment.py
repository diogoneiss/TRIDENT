import torch

from src.training.environment import runtime_environment_tags


def test_cpu_environment_tags_keep_every_key(monkeypatch) -> None:
    monkeypatch.setattr(torch, "__version__", "2.5.1+cpu")
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device: (_ for _ in ()).throw(AssertionError)
    )

    assert runtime_environment_tags(torch.device("cpu")) == {
        "device": "cpu",
        "gpu_name": "none",
        "torch_version": "2.5.1+cpu",
        "cuda_version": "none",
    }


def test_cuda_environment_tags_record_the_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu121")
    monkeypatch.setattr(torch.version, "cuda", "12.1")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")

    assert runtime_environment_tags(torch.device("cuda")) == {
        "device": "cuda",
        "gpu_name": "Test GPU",
        "torch_version": "2.5.1+cu121",
        "cuda_version": "12.1",
    }
