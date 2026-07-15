"""
train.py

Faz A.2.1 — Linear probing: DINOv2 backbone FROZEN, sadece Linear(768->2)
head egitilir.

Varsayilan degerler artik GERCEK A.2.1 BASELINE'ina gore ayarlanmistir
(epochs=12, batch_size=128, limit=0 -> tum veri; plan.md A.2.1 ile uyumlu:
lr=1e-3, sadece head egitilir). Pipeline dogrulamasi (SMOKE TEST) icin
kucuk degerlerle acikca override edilir: --epochs 1 --batch_size 32 --limit 1000.

Kullanim (Colab, gercek baseline — Drive mount edilmis, varsayilan yollar, A100 onerilir):
    from google.colab import drive
    drive.mount('/content/drive')
    !python src/train.py

Kullanim (smoke test — pipeline'i hizlica dogrulamak icin):
    !python src/train.py --epochs 1 --batch_size 32 --limit 1000

Kullanim (yollari acikca vererek, local veya Colab):
    python src/train.py \\
        --train_csv /content/drive/MyDrive/passive-liveness-dinov2/splits/train.csv \\
        --val_csv /content/drive/MyDrive/passive-liveness-dinov2/splits/val.csv \\
        --images_root /content/drive/MyDrive/passive-liveness-dinov2/processed_dedup \\
        --output_dir /content/drive/MyDrive/passive-liveness-dinov2/checkpoints \\
        --epochs 12 --batch_size 128 --limit 0
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colab_utils import mount_drive, default_splits_dir, default_processed_dedup_dir
from src.dataset import CelebASpoofSplitDataset, build_transform
from src.model_dinov2 import DINOv2Backbone, EMBED_DIM


class LinearProbeModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = DINOv2Backbone(freeze_backbone=freeze_backbone)
        self.freeze_backbone = freeze_backbone
        self.head = nn.Linear(EMBED_DIM, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            # Backbone frozen oldugunda dropout/stochastic davranisi kapatmak
            # icin daima eval() — model.train()/eval() cagrilarindan bagimsiz.
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features["cls_token"])


def run_epoch(model, loader, device, criterion, optimizer=None) -> tuple:
    is_train = optimizer is not None
    model.head.train(is_train)

    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += images.size(0)

    return total_loss / total_n, total_correct / total_n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz A.2.1 linear probing (varsayilan: gercek baseline).")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--images_root", type=str, default=None,
                         help="02b'nin cikti klasoru (yuz-kirpilmis + dedup edilmis goruntuler).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Checkpoint'in yazilacagi klasor (verilmezse Drive'da 'checkpoints').")
    parser.add_argument("--epochs", type=int, default=12, help="Plan.md A.2.1: 10-15. Smoke test icin 1 ver.")
    parser.add_argument("--batch_size", type=int, default=128, help="Smoke test icin 32 ver.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0,
                         help="Train/val'i ilk N satirla sinirlar. Varsayilan 0 = tum veri; smoke test icin 1000 ver.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit > 0 else None

    drive_root = None
    if args.train_csv is None or args.val_csv is None or args.images_root is None or args.output_dir is None:
        drive_root = mount_drive()

    train_csv = args.train_csv or str(default_splits_dir(str(drive_root)) / "train.csv")
    val_csv = args.val_csv or str(default_splits_dir(str(drive_root)) / "val.csv")
    images_root = args.images_root or str(default_processed_dedup_dir(str(drive_root)))
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(drive_root)) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"Cihaz: {device}{gpu_name}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV:   {val_csv}")
    print(f"Goruntu koku: {images_root}")
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, limit={limit}")

    train_ds = CelebASpoofSplitDataset(train_csv, images_root, build_transform(train=True), limit=limit)
    val_ds = CelebASpoofSplitDataset(val_csv, images_root, build_transform(train=False), limit=limit)
    print(f"Train: {len(train_ds)} goruntu, Val: {len(val_ds)} goruntu")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LinearProbeModel(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, optimizer=None)
        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.1%}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.1%}  "
              f"({time.time() - t0:.1f}s)")

    ckpt_name = "smoketest_linear_probe.pt" if limit is not None else "linear_probe_a2_1.pt"
    ckpt_path = output_dir / ckpt_name
    torch.save({"head_state_dict": model.head.state_dict(), "args": vars(args)}, ckpt_path)
    print("\n--- Egitim tamamlandi ---")
    print(f"Checkpoint (backbone frozen oldugu icin sadece head kaydedildi): {ckpt_path}")


if __name__ == "__main__":
    main()
