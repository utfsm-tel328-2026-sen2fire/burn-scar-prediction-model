import importlib

import pytest


@pytest.mark.parametrize("module", ["torch", "tensorboard", "numpy"])
def test_importable(module):
    assert importlib.import_module(module) is not None


def test_tensorboard_writes_event_file(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(tmp_path))
    writer.add_scalar("test/loss", 0.5, 1)
    writer.close()
    assert list(tmp_path.glob("events.out.tfevents.*"))


def test_tensor_compute_on_available_device():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(64, 64, device=device)
    assert (x @ x).shape == (64, 64)
