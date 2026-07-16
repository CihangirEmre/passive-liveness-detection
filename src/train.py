"""
train.py

Faz A.2 — DINOv2 uzerine iki asamali egitim, ayni script'ten:
  - A.2.1 Linear probing: --unfreeze_blocks 0 (varsayilan). Backbone TAMAMEN
    frozen, sadece Linear(768->2) head egitilir.
  - A.2.2 Kademeli unfreeze fine-tuning: --unfreeze_blocks N (N>0). Backbone'un
    SON N transformer blogu da egitilir (geri kalani frozen), discriminative
    LR ile: head icin --lr (varsayilan 1e-3), backbone icin --backbone_lr
    (varsayilan 1e-5). CosineAnnealingLR scheduler ve val_acer'a gore early
    stopping (--patience) sadece unfreeze_blocks>0 iken devrede.

Varsayilan degerler A.2.1 GERCEK BASELINE'ina gore ayarlanmistir (epochs=12,
batch_size=128, limit=0 -> tum veri; plan.md A.2.1 ile uyumlu). Pipeline
dogrulamasi (SMOKE TEST) icin kucuk degerlerle acikca override edilir:
--epochs 1 --batch_size 32 --limit 1000.

Checkpoint her zaman SON epoch'un (early stopping tetiklenirse durulan
epoch'un) agirliklarini kaydeder — en iyi val_acer'a gore secim yapilmiyor.

Kullanim (Colab, A.2.1 baseline — Drive mount edilmis, varsayilan yollar, A100 onerilir):
    from google.colab import drive
    drive.mount('/content/drive')
    !python src/train.py

Kullanim (Colab, A.2.2 kademeli unfreeze — son 2 blok, discriminative LR, early stopping):
    !python src/train.py --unfreeze_blocks 2 --epochs 20 --patience 5

Kullanim (smoke test — pipeline'i hizlica dogrulamak icin, unfreeze ile de denenebilir):
    !python src/train.py --epochs 1 --batch_size 32 --limit 1000
    !python src/train.py --unfreeze_blocks 2 --epochs 1 --batch_size 32 --limit 1000

Kullanim (yollari acikca vererek, local veya Colab):
    python src/train.py \\
        --train_csv /content/drive/MyDrive/passive-liveness-dinov2/splits/train.csv \\
        --val_csv /content/drive/MyDrive/passive-liveness-dinov2/splits/val.csv \\
        --images_root /content/drive/MyDrive/passive-liveness-dinov2/processed_dedup \\
        --output_dir /content/drive/MyDrive/passive-liveness-dinov2/checkpoints \\
        --epochs 12 --batch_size 128 --limit 0
"""

import argparse
import csv
import random
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DinoLivenessModel(nn.Module):
    """unfreeze_blocks=0 -> Faz A.2.1 (linear probe, backbone tamamen frozen).
    unfreeze_blocks>0 -> Faz A.2.2 (backbone'un son N transformer blogu da egitilir)."""

    def __init__(self, unfreeze_blocks: int = 0):
        super().__init__()
        self.backbone = DINOv2Backbone(freeze_backbone=True)
        self.backbone.set_unfreeze_last_n_blocks(unfreeze_blocks)
        self.frozen = unfreeze_blocks == 0
        self.head = nn.Linear(EMBED_DIM, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            # Backbone tamamen frozen oldugunda dropout/stochastic davranisi
            # kapatmak icin daima eval() — disaridaki train()/eval()
            # cagrilarindan bagimsiz. unfreeze_blocks>0 iken bu dal calismaz;
            # backbone'un train/eval modu run_epoch'taki model.train(is_train)
            # cagrisiyla belirlenir.
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features["cls_token"])


def run_epoch(model, loader, device, criterion, optimizer=None) -> dict:
    """label_id sozlesmesi (bkz. 03_build_splits.py): 0=live (bona fide), 1=spoof (attack).
    ACER = (APCER+BPCER)/2 — plan.md A.2.1 kabul kriteri.
      APCER: gercek spoof'lardan live diye yanlis siniflandirilanlarin orani.
      BPCER: gercek live'lardan spoof diye yanlis siniflandirilanlarin orani.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_n = 0.0, 0, 0
    live_total, live_correct, spoof_total, spoof_correct = 0, 0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            correct = preds == labels
            total_loss += loss.item() * images.size(0)
            total_correct += correct.sum().item()
            total_n += images.size(0)

            live_mask = labels == 0
            spoof_mask = labels == 1
            live_total += live_mask.sum().item()
            live_correct += correct[live_mask].sum().item()
            spoof_total += spoof_mask.sum().item()
            spoof_correct += correct[spoof_mask].sum().item()

    bpcer = 1 - live_correct / live_total if live_total else 0.0
    apcer = 1 - spoof_correct / spoof_total if spoof_total else 0.0
    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz A.2 egitimi (A.2.1 linear probe / A.2.2 kademeli unfreeze).")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--images_root", type=str, default=None,
                         help="02b'nin cikti klasoru (yuz-kirpilmis + dedup edilmis goruntuler).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Checkpoint'in yazilacagi klasor (verilmezse Drive'da 'checkpoints').")
    parser.add_argument("--epochs", type=int, default=12,
                         help="A.2.1 icin plan.md: 10-15. A.2.2 icin: 15-20. Smoke test icin 1 ver.")
    parser.add_argument("--batch_size", type=int, default=128, help="Smoke test icin 32 ver.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Head learning rate.")
    parser.add_argument("--unfreeze_blocks", type=int, default=0,
                         help="0 = A.2.1 (backbone tamamen frozen). N>0 = A.2.2: backbone'un son N "
                              "transformer blogu da egitilir (plan.md onerisi: 2-4).")
    parser.add_argument("--backbone_lr", type=float, default=1e-5,
                         help="A.2.2'de unfreeze edilen backbone bloklari icin LR (dusuk tutulur, "
                              "pretrained bilgiyi bozmamak icin). unfreeze_blocks=0 iken kullanilmaz.")
    parser.add_argument("--patience", type=int, default=0,
                         help="val_acer N epoch'tur iyilesmezse egitimi erken durdurur. 0 = kapali "
                              "(A.2.1 varsayilani). A.2.2 icin plan.md onerisi: 5.")
    parser.add_argument("--limit", type=int, default=0,
                         help="Train/val'i ilk N satirla sinirlar. Varsayilan 0 = tum veri; smoke test icin 1000 ver.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42,
                         help="Head init, shuffle ve augmentation icin sabit seed (projedeki diger seed'lerle tutarli).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = args.limit if args.limit > 0 else None
    set_seed(args.seed)

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
    print(f"epochs={args.epochs}, batch_size={args.batch_size}, limit={limit}, seed={args.seed}")
    print(f"unfreeze_blocks={args.unfreeze_blocks}, backbone_lr={args.backbone_lr}, patience={args.patience}")

    train_ds = CelebASpoofSplitDataset(train_csv, images_root, build_transform(train=True), limit=limit)
    val_ds = CelebASpoofSplitDataset(val_csv, images_root, build_transform(train=False), limit=limit)
    print(f"Train: {len(train_ds)} goruntu, Val: {len(val_ds)} goruntu")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DinoLivenessModel(unfreeze_blocks=args.unfreeze_blocks).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Egitilebilir parametre sayisi: {n_trainable:,}")

    criterion = nn.CrossEntropyLoss()
    param_groups = [{"params": model.head.parameters(), "lr": args.lr}]
    backbone_trainable_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_trainable_params:
        param_groups.append({"params": backbone_trainable_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = None
    if args.unfreeze_blocks > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_acer = float("inf")
    epochs_without_improve = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, device, criterion, optimizer)
        val_m = run_epoch(model, val_loader, device, criterion, optimizer=None)
        if scheduler:
            scheduler.step()
        elapsed = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.1%} train_acer={train_m['acer']:.1%}  "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['acc']:.1%} val_acer={val_m['acer']:.1%} "
              f"(val_apcer={val_m['apcer']:.1%} val_bpcer={val_m['bpcer']:.1%})  "
              f"({elapsed:.1f}s)")
        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_acc": train_m["acc"], "train_acer": train_m["acer"],
            "val_loss": val_m["loss"], "val_acc": val_m["acc"],
            "val_apcer": val_m["apcer"], "val_bpcer": val_m["bpcer"], "val_acer": val_m["acer"],
            "elapsed_sec": elapsed,
        })

        if args.patience > 0:
            if val_m["acer"] < best_val_acer:
                best_val_acer = val_m["acer"]
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.patience:
                    print(f"\nErken durdurma: val_acer {args.patience} epoch'tur iyilesmedi "
                          f"(en iyi val_acer={best_val_acer:.1%}).")
                    break

    if limit is not None:
        ckpt_name = "smoketest_linear_probe.pt" if args.unfreeze_blocks == 0 else f"smoketest_finetune_u{args.unfreeze_blocks}.pt"
    else:
        ckpt_name = "linear_probe_a2_1.pt" if args.unfreeze_blocks == 0 else f"finetune_a2_2_u{args.unfreeze_blocks}.pt"
    ckpt_path = output_dir / ckpt_name

    checkpoint = {"head_state_dict": model.head.state_dict(), "args": vars(args), "history": history}
    if args.unfreeze_blocks > 0:
        # Backbone'un bir kismi da egitildigi icin stock pretrained agirliklarla
        # yetinilemez — tam backbone state_dict'i de kaydedilmeli.
        checkpoint["backbone_state_dict"] = model.backbone.state_dict()
    torch.save(checkpoint, ckpt_path)

    history_path = output_dir / (ckpt_path.stem + "_history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\n--- Egitim tamamlandi ---")
    print(f"Checkpoint (en son epoch'un agirliklari, en iyi degil): {ckpt_path}")
    print(f"Epoch gecmisi (kalici): {history_path}")


if __name__ == "__main__":
    main()
