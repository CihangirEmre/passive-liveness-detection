"""Faz A.0 kabul kriteri check script'i.

Model yukler, dummy input (1,3,224,224) ile forward pass yapar,
CLS token ve patch token boyutlarini dogrular. Hem lokal (CPU) hem de
Colab (GPU) ortaminda calisir.

Kullanim:
    python scripts/00_check_dinov2_setup.py
    python scripts/00_check_dinov2_setup.py --freeze-backbone false --resolution 518
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_dinov2 import DINOv2Backbone, MODEL_ID, PATCH_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 setup / forward-pass dogrulama")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--resolution", type=int, default=224, help="14'un kati olmali (224 veya 518)")
    parser.add_argument(
        "--freeze-backbone",
        type=lambda v: v.lower() != "false",
        default=True,
        help="true/false — backbone donuk mu egitilebilir mi baslasin",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resolution % PATCH_SIZE != 0:
        raise ValueError(f"resolution ({args.resolution}), patch_size ({PATCH_SIZE}) ile bolunebilir olmali.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model yukleniyor: {args.model_id} (freeze_backbone={args.freeze_backbone}) ...")

    model = DINOv2Backbone(model_id=args.model_id, freeze_backbone=args.freeze_backbone).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, args.resolution, args.resolution, device=device)
    with torch.no_grad():
        out = model(dummy_input)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"CLS token shape:    {tuple(out['cls_token'].shape)}")
    print(f"Patch tokens shape: {tuple(out['patch_tokens'].shape)}")
    print(f"Toplam parametre:   {total:,}")
    print(f"Egitilebilir param: {trainable:,} ({'unfrozen' if trainable else 'frozen'})")
    print("Forward pass basarili.")


if __name__ == "__main__":
    main()
