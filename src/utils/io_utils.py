"""Shared I/O utilities: checkpoint save/load, JSON read/write."""

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    metadata: dict[str, Any],
                    filepath: Path | str) -> None:
    """Save a training checkpoint with model state, optimizer state, and metadata.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        metadata: Dict with keys like 'loss_history', 'config', 'n_users', 'n_items'.
        filepath: Target file path for the .pth checkpoint.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **metadata,
    }
    torch.save(checkpoint, str(filepath))
    logger.info("Checkpoint saved to %s", filepath)


def load_checkpoint(filepath: Path | str,
                    map_location: str = "cpu") -> dict[str, Any]:
    """Safely load a training checkpoint.

    Args:
        filepath: Path to the .pth checkpoint file.
        map_location: Device to map tensors to.

    Returns:
        The checkpoint dict with model_state_dict, optimizer_state_dict, etc.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(str(filepath), map_location=map_location,
                            weights_only=True)
    logger.info("Checkpoint loaded from %s", filepath)
    return checkpoint


def read_json(filepath: Path | str) -> Any:
    """Read and parse a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, filepath: Path | str, indent: int = 2) -> None:
    """Write data to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
