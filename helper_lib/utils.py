import torch
from pathlib import Path
from typing import Union

def save_model(model: torch.nn.Module, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))

def load_model(model: torch.nn.Module, path: Union[str, Path], map_location="cpu") -> torch.nn.Module:
    state = torch.load(str(path), map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model
