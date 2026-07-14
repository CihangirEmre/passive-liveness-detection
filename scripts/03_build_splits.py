"""
03_build_splits.py

02'nin ciktisi (yuz-kirpilmis CelebA-Spoof goruntuleri) uzerinden
SUBJECT-DISJOINT train/val/test split uretir (varsayilan oran 70/15/15).
Split, subject_id kumesi uzerinde yapilir ve her subject'in TUM goruntuleri
ayni split'e atanir — boylece ayni kisi hem train hem test'te olamaz.

Opsiyonel: --metas_dir verilirse (01'in indirdigi ham veri icindeki metas/
klasoru), resmi CelebA-Spoof label dosyalarindan (train_label.txt /
test_label.txt) spoof_type/illumination/environment bilgisi de eklenir.
Label semasi resmi repo README'sinden dogrulanmistir
(https://github.com/ZhangYuanhan-AI/CelebA-Spoof):
    label_vector[40] = spoof type (0=Live, 1=Photo, 2=Poster, 3=A4,
                                    4=Face Mask, 5=Upper Body Mask,
                                    6=Region Mask, 7=PC, 8=Pad, 9=Phone,
                                    10=3D Mask)
    label_vector[41] = illumination condition
    label_vector[42] = environment
    label_vector[43] = live/spoof binary label
Bu Kaggle mirror'inda metas/ bulunmayabilir — bu durumda spoof_type
kolonu "unknown" ile doldurulur, binary live/spoof label (klasor
adindan alinan, guvenilir) etkilenmez.

Kullanim (Colab):
    python scripts/03_build_splits.py
    python scripts/03_build_splits.py --metas_dir /content/celeba_spoof_raw/metas
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colab_utils import mount_drive, default_processed_dir, default_splits_dir
from src.scan_utils import iter_split_subject_label_images

SPOOF_TYPE_NAMES = {
    0: "Live", 1: "Photo", 2: "Poster", 3: "A4", 4: "Face Mask",
    5: "Upper Body Mask", 6: "Region Mask", 7: "PC", 8: "Pad",
    9: "Phone", 10: "3D Mask",
}


def load_spoof_type_lookup(metas_dir: Path) -> dict:
    """metas/ altindaki train_label.txt / test_label.txt dosyalarini bulup
    parse eder. Anahtar: "<subject_id>/<live|spoof>/<filename>" (son 3 path
    parcasi) -> spoof_type_name. Dosya bulunamaz/parse edilemezse bos dict doner.
    """
    lookup = {}
    label_files = list(metas_dir.rglob("train_label.txt")) + list(metas_dir.rglob("test_label.txt"))

    if not label_files:
        print(f"[UYARI] {metas_dir} altinda train_label.txt/test_label.txt bulunamadi. "
              f"spoof_type 'unknown' olarak doldurulacak.")
        return lookup

    for label_file in label_files:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 44:
                    continue
                rel_path = parts[0].replace("\\", "/")
                try:
                    spoof_type_id = int(parts[40])
                except ValueError:
                    continue
                key = "/".join(rel_path.split("/")[-3:])
                lookup[key] = SPOOF_TYPE_NAMES.get(spoof_type_id, f"unknown_{spoof_type_id}")

    print(f"spoof_type lookup: {len(label_files)} label dosyasindan {len(lookup)} kayit yuklendi.")
    return lookup


def build_metadata(processed_dir: Path, spoof_type_lookup: dict) -> pd.DataFrame:
    rows = []
    for _split, subject_id, label, img_path in iter_split_subject_label_images(processed_dir):
        key = f"{subject_id}/{label}/{img_path.name}"
        spoof_type = spoof_type_lookup.get(key, "Live" if label == "live" else "unknown")
        rows.append({
            "image_path": str(img_path),
            "subject_id": subject_id,
            "label": label,
            "label_id": 0 if label == "live" else 1,
            "spoof_type": spoof_type,
        })

    if not rows:
        raise RuntimeError(
            f"{processed_dir} altinda hic goruntu bulunamadi. "
            f"02_extract_faces.py'nin cikti klasoru dogru mu, klasor yapisi "
            f"<split>/<subject_id>/<live|spoof>/<image> seklinde mi kontrol et."
        )

    return pd.DataFrame(rows)


def subject_disjoint_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int
) -> pd.DataFrame:
    subjects = sorted(df["subject_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train + n_val])
    test_subjects = set(subjects[n_train + n_val:])

    assert train_subjects.isdisjoint(val_subjects), "train/val subject overlap tespit edildi!"
    assert train_subjects.isdisjoint(test_subjects), "train/test subject overlap tespit edildi!"
    assert val_subjects.isdisjoint(test_subjects), "val/test subject overlap tespit edildi!"
    assert train_subjects | val_subjects | test_subjects == set(subjects), "bazi subject'ler hicbir split'e atanmadi!"

    def assign(sid: str) -> str:
        if sid in train_subjects:
            return "train"
        if sid in val_subjects:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["subject_id"].map(assign)

    # Ikinci bir dogrulama: goruntu seviyesinde de subject overlap olmadigini kontrol et.
    split_subjects = df.groupby("split")["subject_id"].apply(set)
    assert split_subjects["train"].isdisjoint(split_subjects["val"])
    assert split_subjects["train"].isdisjoint(split_subjects["test"])
    assert split_subjects["val"].isdisjoint(split_subjects["test"])

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject-disjoint train/val/test split uretir.")
    parser.add_argument("--processed_dir", type=str, default=None,
                         help="02'nin cikti klasoru (verilmezse Drive'daki varsayilan yol kullanilir).")
    parser.add_argument("--metas_dir", type=str, default=None,
                         help="Ham CelebA-Spoof indirmesindeki metas/ klasoru (opsiyonel, spoof_type zenginlestirme icin).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Split CSV'lerinin yazilacagi klasor (verilmezse Drive'daki varsayilan yol kullanilir).")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio 1.0 olmali"

    drive_root = None
    if args.processed_dir is None or args.output_dir is None:
        drive_root = mount_drive()

    processed_dir = Path(args.processed_dir) if args.processed_dir else default_processed_dir(str(drive_root))
    output_dir = Path(args.output_dir) if args.output_dir else default_splits_dir(str(drive_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    spoof_type_lookup = {}
    if args.metas_dir:
        spoof_type_lookup = load_spoof_type_lookup(Path(args.metas_dir))

    print(f"Taraniyor: {processed_dir}")
    df = build_metadata(processed_dir, spoof_type_lookup)
    print(f"Toplam goruntu: {len(df)}, toplam subject: {df['subject_id'].nunique()}")

    df = subject_disjoint_split(df, args.train_ratio, args.val_ratio, args.seed)

    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split]
        out_path = output_dir / f"{split}.csv"
        split_df.drop(columns=["split"]).to_csv(out_path, index=False)
        print(f"{split}: {len(split_df)} goruntu, {split_df['subject_id'].nunique()} subject -> {out_path}")

    print("\nSubject-disjoint dogrulama assert'leri basarili.")
    print(f"Split CSV'leri: {output_dir}")


if __name__ == "__main__":
    main()
