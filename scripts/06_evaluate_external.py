"""
06_evaluate_external.py


Kullanim (Colab, internal rapordan gelen esik + karsilastirma icin):
    python scripts/06_evaluate_external.py \
        --checkpoint /content/drive/MyDrive/passive-liveness-dinov2/checkpoints_v2/finetune_a2_2_u2.pt \
        --live_manifest /content/LCC_FASD/LCC_FASD/CLIENT_TEST.txt \
        --spoof_manifest /content/LCC_FASD/LCC_FASD/IMPOSTER_TEST.txt \
        --images_root /content/LCC_FASD/LCC_FASD \
        --threshold 0.4724 \
        --internal_acer 0.0044 --internal_eer 0.0045 --internal_auc 0.9999
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import ManifestPairDataset, build_transform
from src.eval_utils import load_checkpoint_model, run_inference
from src.metrics import compute_apcer_bpcer_acer, compute_auc, compute_eer


def build_report(
    images_root: str, checkpoint_path: str, n_live: int, n_spoof: int,
    threshold: float, metrics_at_threshold: dict,
    ext_eer: float, ext_auc: float,
    internal_acer: float, internal_eer: float, internal_auc: float,
) -> str:
    lines = ["# External (Zero-Shot) Degerlendirme Raporu (Faz A.3)\n"]
    lines.append(f"**Checkpoint:** `{checkpoint_path}`")
    lines.append(f"**Harici veri seti:** `{images_root}` (live={n_live}, spoof={n_spoof})\n")

    lines.append("## Esik-Bagimsiz Metrikler")
    lines.append(f"- External EER: {ext_eer:.2%}")
    lines.append(f"- External AUC: {ext_auc:.4f}\n")

    if metrics_at_threshold is not None:
        lines.append(f"## Sabit Esikte Sonuclar (esik={threshold:.4f}, internal val EER'den)")
        lines.append("| Metrik | Deger |")
        lines.append("|---|---|")
        lines.append(f"| APCER | {metrics_at_threshold['apcer']:.2%} |")
        lines.append(f"| BPCER | {metrics_at_threshold['bpcer']:.2%} |")
        lines.append(f"| ACER | {metrics_at_threshold['acer']:.2%} |\n")

    if internal_eer is not None and internal_auc is not None:
        lines.append("## Internal vs External Karsilastirma\n")
        lines.append("| Metrik | Internal (test) | External | Fark (external - internal) |")
        lines.append("|---|---|---|---|")
        ext_acer = metrics_at_threshold["acer"] if metrics_at_threshold else None
        if internal_acer is not None and ext_acer is not None:
            lines.append(f"| ACER | {internal_acer:.2%} | {ext_acer:.2%} | {ext_acer - internal_acer:+.2%} |")
        lines.append(f"| EER | {internal_eer:.2%} | {ext_eer:.2%} | {ext_eer - internal_eer:+.2%} |")
        lines.append(f"| AUC | {internal_auc:.4f} | {ext_auc:.4f} | {ext_auc - internal_auc:+.4f} |")
        lines.append(
            "\nBuyuk bir ACER/EER farki (external >> internal), modelin CelebA-Spoof'un "
            "kendi cekim/recapture pipeline'ina ozgu izlere fazla uyum sagladigini "
            "(gercek liveness sinyalinden fazla) gosterebilir — bkz. plan.md riskler bolumu."
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faz A.3 — harici veri setinde zero-shot degerlendirme.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--live_manifest", type=str, required=True,
                         help="Genuine/client goruntu yollarini satir satir listeleyen .txt (orn. CLIENT_TEST.txt).")
    parser.add_argument("--spoof_manifest", type=str, required=True,
                         help="Spoof/imposter goruntu yollarini satir satir listeleyen .txt (orn. IMPOSTER_TEST.txt).")
    parser.add_argument("--images_root", type=str, required=True,
                         help="Goruntulerin gercekte bulundugu yerel klasor (manifest'teki yollar sadece "
                              "dosya adi icin kullanilir, --images_root ile yeniden kurulur).")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Internal val EER esigi (05_evaluate_internal.py ciktisi). "
                              "Verilmezse sadece esik-bagimsiz EER/AUC raporlanir.")
    parser.add_argument("--internal_acer", type=float, default=None,
                         help="05_evaluate_internal.py'nin konsola bastigi test ACER'i (orn. 0.0044). "
                              "internal_eer ve internal_auc ile birlikte verilirse karsilastirma tablosu eklenir.")
    parser.add_argument("--internal_eer", type=float, default=None,
                         help="05'in konsola bastigi test EER'i.")
    parser.add_argument("--internal_auc", type=float, default=None,
                         help="05'in konsola bastigi test AUC'u.")
    parser.add_argument("--output_dir", type=str, default="docs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test icin ilk N goruntuyle sinirla.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Checkpoint yukleniyor: {args.checkpoint}")
    model, ckpt = load_checkpoint_model(args.checkpoint, device)
    print(f"unfreeze_blocks={ckpt['args'].get('unfreeze_blocks', 0)}")

    transform = build_transform(train=False)
    ds = ManifestPairDataset(args.live_manifest, args.spoof_manifest, args.images_root,
                              transform, limit=args.limit)
    n_live = sum(1 for _, label in ds.items if label == 0)
    n_spoof = sum(1 for _, label in ds.items if label == 1)
    print(f"Harici veri seti: {len(ds)} goruntu (live={n_live}, spoof={n_spoof})")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Inference calisiyor (zero-shot, fine-tuning YOK)...")
    labels, scores = run_inference(model, loader, device)

    ext_eer, ext_eer_threshold = compute_eer(labels, scores)
    ext_auc = compute_auc(labels, scores)
    print(f"\nExternal EER: {ext_eer:.2%}  (kendi esigi: {ext_eer_threshold:.4f})")
    print(f"External AUC: {ext_auc:.4f}")

    metrics_at_threshold = None
    if args.threshold is not None:
        metrics_at_threshold = compute_apcer_bpcer_acer(labels, scores, args.threshold)
        print(f"\n--- Sabit esikte ({args.threshold:.4f}, internal val EER'den) ---")
        print(f"APCER: {metrics_at_threshold['apcer']:.2%}  "
              f"BPCER: {metrics_at_threshold['bpcer']:.2%}  "
              f"ACER: {metrics_at_threshold['acer']:.2%}")
    else:
        print("\n[UYARI] --threshold verilmedi, sabit esikte ACER/APCER/BPCER hesaplanmadi "
              "— sadece esik-bagimsiz EER/AUC raporlaniyor.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        args.images_root, args.checkpoint, n_live, n_spoof,
        args.threshold, metrics_at_threshold, ext_eer, ext_auc,
        args.internal_acer, args.internal_eer, args.internal_auc,
    )
    report_path = output_dir / "external_eval_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nRapor kaydedildi: {report_path}")


if __name__ == "__main__":
    main()
