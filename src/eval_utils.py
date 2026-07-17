"""05_evaluate_internal.py ve 06_evaluate_external.py arasinda ortak
checkpoint yukleme + inference yardimcilari."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train import DinoLivenessModel


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Checkpoint'teki 'args' icindeki unfreeze_blocks'a gore dogru mimariyi
    kurar, head (+ varsa backbone) agirliklarini yukler. (model, ckpt) doner.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    unfreeze_blocks = ckpt["args"].get("unfreeze_blocks", 0)

    model = DinoLivenessModel(unfreeze_blocks=unfreeze_blocks).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    if ckpt.get("backbone_state_dict") is not None:
        model.backbone.load_state_dict(ckpt["backbone_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """(labels, scores) doner — scores: modelin spoof (label=1) olasiligi."""
    model.eval()
    all_labels, all_scores = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs.cpu().numpy())
        all_labels.append(np.asarray(labels))
    return np.concatenate(all_labels), np.concatenate(all_scores)
