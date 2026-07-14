"""
04_stats_report.py

03'un ciktisi (train/val/test.csv) uzerinden veri seti istatistik raporu
uretir: toplam goruntu, live/spoof orani, spoof tipi dagilimi, subject
sayisi (genel + split bazli) — hem konsola yazdirir hem de
data_stats_report.md + sample_batch.png olarak kaydeder (A.1 kabul kriteri).

Kullanim (Colab):
    python scripts/04_stats_report.py
    python scripts/04_stats_report.py --splits_dir /content/drive/MyDrive/passive-liveness-dinov2/splits
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colab_utils import mount_drive, default_splits_dir

SPLITS = ("train", "val", "test")


def load_splits(splits_dir: Path) -> dict:
    dfs = {}
    for split in SPLITS:
        path = splits_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadi. Once scripts/03_build_splits.py calistirilmali.")
        dfs[split] = pd.read_csv(path)
    return dfs


def build_report(dfs: dict) -> str:
    all_df = pd.concat(dfs.values(), ignore_index=True)
    lines = ["# Veri Seti Istatistik Raporu (Faz A.1)\n"]

    lines.append(f"**Toplam goruntu:** {len(all_df)}")
    lines.append(f"**Toplam subject:** {all_df['subject_id'].nunique()}\n")

    lines.append("## Split Bazli Ozet\n")
    lines.append("| Split | Goruntu | Subject | Live | Spoof | Live oran |")
    lines.append("|---|---|---|---|---|---|")
    for split in SPLITS:
        df = dfs[split]
        n_live = int((df["label"] == "live").sum())
        n_spoof = int((df["label"] == "spoof").sum())
        live_ratio = n_live / len(df) if len(df) else 0.0
        lines.append(f"| {split} | {len(df)} | {df['subject_id'].nunique()} | {n_live} | {n_spoof} | {live_ratio:.1%} |")

    lines.append("\n## Spoof Tipi Dagilimi (tum split'ler)\n")
    if "spoof_type" in all_df.columns:
        spoof_counts = all_df["spoof_type"].value_counts()
        lines.append("| Spoof Tipi | Adet |")
        lines.append("|---|---|")
        for spoof_type, count in spoof_counts.items():
            lines.append(f"| {spoof_type} | {count} |")
        if (all_df["spoof_type"] == "unknown").all():
            lines.append("\n> [UYARI] Tum spoof_type degerleri 'unknown' — metas/ label dosyalari "
                          "bulunamadi/eslenemedi. Sadece binary live/spoof label guvenilir.")
    else:
        lines.append("spoof_type kolonu bulunamadi.")

    lines.append("\n## Subject Overlap Kontrolu\n")
    subj_sets = {split: set(dfs[split]["subject_id"]) for split in SPLITS}
    overlaps = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = subj_sets[a] & subj_sets[b]
        overlaps.append((a, b, len(overlap)))
    all_clean = all(n == 0 for _, _, n in overlaps)
    for a, b, n in overlaps:
        lines.append(f"- {a} vs {b}: {n} ortak subject")
    lines.append(f"\n**Sonuc: {'PASS — subject-disjoint dogrulandi' if all_clean else 'FAIL — overlap tespit edildi!'}**")

    return "\n".join(lines)


def save_sample_batch(dfs: dict, output_path: Path, n_samples: int = 16, seed: int = 42) -> None:
    train_df = dfs["train"]
    rng = random.Random(seed)
    n_samples = min(n_samples, len(train_df))
    sample_rows = train_df.sample(n=n_samples, random_state=seed).to_dict("records")

    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for ax, row in zip(axes, sample_rows):
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            ax.imshow(img)
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, f"okuma hatasi:\n{e}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{row['label']} / {row.get('spoof_type', '?')}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(sample_rows):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Veri seti istatistik raporu uretir.")
    parser.add_argument("--splits_dir", type=str, default=None,
                         help="03'un cikti klasoru (verilmezse Drive'daki varsayilan yol kullanilir).")
    parser.add_argument("--output_dir", type=str, default="docs",
                         help="Rapor ve ornek batch goruntusunun yazilacagi klasor.")
    parser.add_argument("--n_samples", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.splits_dir:
        splits_dir = Path(args.splits_dir)
    else:
        drive_root = mount_drive()
        splits_dir = default_splits_dir(str(drive_root))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs = load_splits(splits_dir)

    report = build_report(dfs)
    print(report)

    report_path = output_dir / "data_stats_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nRapor kaydedildi: {report_path}")

    sample_path = output_dir / "sample_batch.png"
    save_sample_batch(dfs, sample_path, n_samples=args.n_samples)
    print(f"Ornek batch gorsellestirmesi kaydedildi: {sample_path}")


if __name__ == "__main__":
    main()
