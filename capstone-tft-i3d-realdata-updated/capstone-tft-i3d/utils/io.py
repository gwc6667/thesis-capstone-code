from pathlib import Path
import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(model: torch.nn.Module, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(model: torch.nn.Module, path, map_location=None):
    if map_location is None:
        map_location = "cpu"
    state = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(state)
    return model
