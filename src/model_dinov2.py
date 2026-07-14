"""DINOv2 (with Registers) backbone loader — Faz A.0.

Kaynak: torch.hub("facebookresearch/dinov2", "dinov2_vitb14_reg")
"reg" varyantı ZORUNLU (attention artifact fix, bkz. dinov2_liveness_plan.md).
"""

import torch
import torch.nn as nn

MODEL_ID = "dinov2_vitb14_reg"
PATCH_SIZE = 14
EMBED_DIM = 768  # ViT-B/14 CLS token boyutu


class DINOv2Backbone(nn.Module):
    def __init__(self, model_id: str = MODEL_ID, freeze_backbone: bool = True):
        super().__init__()
        self.model_id = model_id
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_id)
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, 3, H, W) — H ve W, PATCH_SIZE'in katı olmalı (224 veya 518).

        Dönüş:
            cls_token:    (B, EMBED_DIM)
            patch_tokens: (B, num_patches, EMBED_DIM)
        """
        features = self.backbone.forward_features(x)
        return {
            "cls_token": features["x_norm_clstoken"],
            "patch_tokens": features["x_norm_patchtokens"],
        }
