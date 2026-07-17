"""
05_evaluate_internal.py

Faz A.3 — Faz A.2'de egitilmis bir checkpoint'i CelebA-Spoof TEST split'i
uzerinde degerlendirir (internal degerlendirme; test split egitim/val
sirasinda hic gorulmedi, subject-disjoint).

Esik (threshold) secimi: plan.md'nin belirttigi gibi VAL split'inin EER
noktasindan turetilir, sonra bu SABIT esik test split'ine uygulanir — esigi
dogrudan test'e gore secmek (ornegin test'in kendi EER'i) iyimser/sizinti
riski tasir, bu yuzden val->test ayrimi korunur.

Cikti: konsola ozet + docs/internal_eval_report.md (toplam metrikler +
spoof_type bazli APCER kirilimi).

Kullanim (Colab):
    python scripts/05_evaluate_internal.py \
        --checkpoint /content/drive/MyDrive/passive-liveness-dinov2/checkpoints_v2/finetune_a2_2_u2.pt \
        --val_csv /content/drive/MyDrive/passive-liveness-dinov2/splits_v2/val.csv \
        --test_csv /content/drive/MyDrive/passive-liveness-dinov2/splits_v2/test.csv \
        --images_root /content/celeba_processed_dedup_v2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import CelebASpoofSplitDataset, build_transform
from src.eval_utils import load_checkpoint_model, run_inference
from src.metrics import compute_apcer_bpcer_acer, compute_auc, compute_eer


def build_report(
    val_eer: float, val_eer_threshold: float,
    test_metrics: dict, test_eer: float, test_auc: float,
    spoof_type_breakdown: pd.DataFrame,
    checkpoint_path: str,
) -> str:
    lines = ["# Internal Degerlendirme Raporu (Faz A.3)\n"]
    lines.append(f"**Checkpoint:** `{checkpoint_path}`\n")

    lines.append("## Esik Secimi (Val Split)")
    lines.append(f"- Val EER: {val_eer:.2%}")
    lines.append(f"- Val EER esigi (test'e sabit uygulanir): {val_eer_threshold:.4f}\n")

    lines.append("## Test Split Sonuclari (bu esikte)")
    lines.append("| Metrik | Deger |")
    lines.append("|---|---|")
    lines.append(f"| APCER | {test_metrics['apcer']:.2%} |")
    lines.append(f"| BPCER | {test_metrics['bpcer']:.2%} |")
    lines.append(f"| ACER | {test_metrics['acer']:.2%} |")
    lines.append(f"| EER (esik-bagimsiz) | {test_eer:.2%} |")
    lines.append(f"| AUC | {test_auc:.4f} |\n")

    lines.append("## Spoof Tipi Bazli APCER Kirilimi (val EER esiginde)\n")
    lines.append("| Spoof Tipi | N | APCER |")
    lines.append("|---|---|---|")
    for _, row in spoof_type_breakdown.iterrows():
        lines.append(f"| {row['spoof_type']} | {int(row['n'])} | {row['apcer']:.2%} |")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz A.3 — CelebA-Spoof test split'inde internal degerlendirme.")
    parser.add_argument("--checkpoint", type=str, required=True, help="train.py'nin uretttigi .pt dosyasi.")
    parser.add_argument("--val_csv", type=str, required=True, help="Esik (EER) secimi icin val split.")
    parser.add_argument("--test_csv", type=str, required=True, help="Nihai raporlama icin test split.")
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="docs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test icin ilk N satirla sinirla.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Checkpoint yukleniyor: {args.checkpoint}")
    model, ckpt = load_checkpoint_model(args.checkpoint, device)
    print(f"unfreeze_blocks={ckpt['args'].get('unfreeze_blocks', 0)}, "
          f"egitim history uzunlugu={len(ckpt.get('history', []))} epoch")

    transform = build_transform(train=False)

    val_ds = CelebASpoofSplitDataset(args.val_csv, args.images_root, transform, limit=args.limit)
    test_ds = CelebASpoofSplitDataset(args.test_csv, args.images_root, transform, limit=args.limit)
    print(f"Val: {len(val_ds)} goruntu, Test: {len(test_ds)} goruntu")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Val split uzerinde EER esigi hesaplaniyor...")
    val_labels, val_scores = run_inference(model, val_loader, device)
    val_eer, val_eer_threshold = compute_eer(val_labels, val_scores)
    print(f"Val EER: {val_eer:.2%}  (esik: {val_eer_threshold:.4f})")

    print("Test split uzerinde degerlendirme yapiliyor...")
    test_labels, test_scores = run_inference(model, test_loader, device)
    test_metrics = compute_apcer_bpcer_acer(test_labels, test_scores, val_eer_threshold)
    test_eer, _ = compute_eer(test_labels, test_scores)
    test_auc = compute_auc(test_labels, test_scores)

    print(f"\n--- Test Sonuclari (val EER esiginde) ---")
    print(f"APCER: {test_metrics['apcer']:.2%}  BPCER: {test_metrics['bpcer']:.2%}  "
          f"ACER: {test_metrics['acer']:.2%}")
    print(f"Test EER (esik-bagimsiz): {test_eer:.2%}  AUC: {test_auc:.4f}")

    # Spoof tipi bazli kirilim — test.csv'yi ayni sira/limit ile tekrar okuyoruz
    # (Dataset shuffle=False oldugu icin siralama test_labels/test_scores ile birebir eslesir).
    test_df = pd.read_csv(args.test_csv, dtype={"subject_id": str})
    if args.limit:
        test_df = test_df.iloc[: args.limit].reset_index(drop=True)

    preds = (test_scores >= val_eer_threshold).astype(int)
    rows = []
    for spoof_type, group_idx in test_df.groupby("spoof_type").groups.items():
        if spoof_type == "Live":
            continue
        idx = np.array(group_idx)
        apcer = float(np.mean(preds[idx] == 0))  # bu spoof tipinin live sanilma orani
        rows.append({"spoof_type": spoof_type, "n": len(idx), "apcer": apcer})
    breakdown_df = pd.DataFrame(rows).sort_values("apcer", ascending=False)

    print("\n--- Spoof Tipi Bazli APCER ---")
    print(breakdown_df.to_string(index=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        val_eer, val_eer_threshold, test_metrics, test_eer, test_auc, breakdown_df, args.checkpoint
    )
    report_path = output_dir / "internal_eval_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nRapor kaydedildi: {report_path}")
    print("\n(06_evaluate_external.py ile internal/external karsilastirmasi icin yukaridaki "
          "ACER/EER/AUC degerlerini --internal_acer/--internal_eer/--internal_auc olarak kopyalayin.)")


if __name__ == "__main__":
    main()
