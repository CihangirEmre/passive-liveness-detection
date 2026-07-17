"""
02b_dedupe_phash.py

02'nin ciktisi (yuz-kirpilmis goruntuler) icinde neredeyse birebir ayni
goruntuleri (ayni cekim oturumunun art arda kareleri gibi) perceptual hash
(pHash) ile tespit edip eler. Ayni (subject_id, label) grubu icinde
calisir — farkli kisileri veya live/spoof'u birbirine karistirmaz.

Yontem: Her goruntunun pHash'i (hash_size=8 -> 64 bit) cikarilir. Ayni grup
icinde, zaten "tutulan" bir goruntuye Hamming distance'i <= --threshold olan
goruntuler duplicate sayilip elenir (greedy, goruntu sirasina gore).
Bu sadece — near-exact frame duplicate'lerini
temizler. 

Kullanim (Colab):
    python scripts/02b_dedupe_phash.py
    python scripts/02b_dedupe_phash.py --threshold 8 --limit 500   # smoke test
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colab_utils import mount_drive, default_processed_dir, default_processed_dedup_dir
from src.scan_utils import iter_split_subject_label_images


def compute_phash(image_path: Path, hash_size: int = 8) -> imagehash.ImageHash:
    with Image.open(image_path) as img:
        return imagehash.phash(img.convert("RGB"), hash_size=hash_size)


def dedupe_group(items, hash_size: int, threshold: int):
    """items: ayni (subject_id, label) grubuna ait (split, subject_id, label, path)
    tuple'lari. (kept, dropped) doner — dropped elemanlari (item, sebep) seklinde.
    """
    kept, kept_hashes, dropped = [], [], []

    for item in items:
        _, _, _, path = item
        try:
            h = compute_phash(path, hash_size=hash_size)
        except Exception as e:  # noqa: BLE001 — bozuk/okunamayan tek goruntu run'i durdurmamali
            dropped.append((item, f"HASH_ERROR: {e}"))
            continue

        is_dup = any((h - kh) <= threshold for kh in kept_hashes)
        if is_dup:
            dropped.append((item, "NEAR_DUPLICATE"))
        else:
            kept.append(item)
            kept_hashes.append(h)

    return kept, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pHash ile near-duplicate goruntu eleme (Faz A.1, 1. asama dedup).")
    parser.add_argument("--processed_dir", type=str, default=None,
                         help="02'nin cikti klasoru (verilmezse Drive'daki varsayilan yol kullanilir).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Deduplike edilmis goruntulerin kopyalanacagi klasor (verilmezse Drive'da 'processed_dedup').")
    parser.add_argument("--hash_size", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=5,
                         help="Bu Hamming distance'in ALTINDAKI (esit dahil) goruntuler duplicate sayilir.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test icin ilk N goruntuyle sinirla.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    drive_root = None
    if args.processed_dir is None or args.output_dir is None:
        drive_root = mount_drive()

    processed_dir = Path(args.processed_dir) if args.processed_dir else default_processed_dir(str(drive_root))
    output_dir = Path(args.output_dir) if args.output_dir else default_processed_dedup_dir(str(drive_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Girdi:  {processed_dir}")
    print(f"Cikti:  {output_dir}")

    all_items = list(iter_split_subject_label_images(processed_dir))
    if args.limit:
        all_items = all_items[: args.limit]
    print(f"Taranan goruntu: {len(all_items)}")

    groups = defaultdict(list)
    for item in all_items:
        _split, subject_id, label, _path = item
        groups[(subject_id, label)].append(item)
    print(f"Grup sayisi (subject_id x label): {len(groups)}")

    total_kept, total_dropped = 0, 0
    drop_log_path = output_dir / "dedupe_dropped.txt"

    with open(drop_log_path, "w") as drop_log:
        for (_subject_id, _label), items in tqdm(groups.items(), desc="Dedupe (grup bazli)"):
            kept, dropped = dedupe_group(items, hash_size=args.hash_size, threshold=args.threshold)

            for split, sid, lbl, path in kept:
                dest = output_dir / split / sid / lbl / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(path, dest)

            for (split, sid, lbl, path), reason in dropped:
                drop_log.write(f"{path}\t{reason}\n")

            total_kept += len(kept)
            total_dropped += len(dropped)

    total = total_kept + total_dropped
    print("\n--- Tamamlandi ---")
    print(f"Tutulan:  {total_kept} / {total}  ({total_kept / max(1, total):.1%})")
    print(f"Elenen:   {total_dropped} / {total}  ({total_dropped / max(1, total):.1%})")
    print(f"Elenenlerin detayi: {drop_log_path}")
    print(f"Deduplike edilmis veri seti (Drive'a yazildi): {output_dir}")
    print("\nNot: 03_build_splits.py'i bu klasorle calistirmak icin:")
    print(f"    python scripts/03_build_splits.py --processed_dir {output_dir}")


if __name__ == "__main__":
    main()
