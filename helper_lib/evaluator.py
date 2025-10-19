# helper_lib/evaluator.py
import torch
from typing import Callable, Tuple


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataloader and return (avg_loss, accuracy_percent).
    For non-classification heads, accuracy will be 0.0 unless logits with class dimension are detected.
    """
    model.eval()
    model.to(device)
    total_loss, total, correct = 0.0, 0, 0

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)

        if outputs.ndim == 2 and outputs.size(1) > 1:
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    denom = total if total > 0 else len(data_loader.dataset)
    avg_loss = total_loss / max(1, denom)
    acc = (100.0 * correct / total) if total > 0 else 0.0
    return avg_loss, acc
