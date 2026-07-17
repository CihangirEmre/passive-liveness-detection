"""
03_build_splits.py

02'nin ciktisi (yuz-kirpilmis CelebA-Spoof goruntuleri) uzerinden
SUBJECT-DISJOINT train/val/test split uretir (varsayilan oran 70/15/15).
Split, subject_id kumesi uzerinde yapilir ve her subject'in TUM goruntuleri
ayni split'e atanir — boylece ayni kisi hem train hem test'te olamaz.


Kullanim (Colab):
    python scripts/03_build_splits.py
    python scripts/03_build_splits.py --metas_dir /content/celeba_spoof_raw/metas
"""

import argparse
import json
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


def load_label_lookup(metas_dir: Path) -> dict:
    """metas/ altindaki train_label.json / test_label.json dosyalarini bulup
    parse eder. Anahtar: "<subject_id>/<live|spoof>/<filename>" (son 3 path
    parcasi) -> (spoof_type_name, binary_label_id [0=live, 1=spoof]).
    Dosya bulunamaz/parse edilemezse bos dict doner.

    NOT: Bazi Kaggle mirror'larinda (orn. mabdullahsajid/celeba-spoofing)
    train_label.txt/test_label.txt SADECE "<path> <binary_label>" (2 kolon)
    iceriyor — resmi 44 kolonlu format degil. Resmi 44-elemanli label vektoru
    (spoof_type index 40, binary label index 43) sadece .json dosyalarinda
    (path -> list[44] dict) bulunuyor, bu yuzden .json tercih edilir;
    .txt sadece (44 kolonlu olmasi durumunda) fallback'tir.
    """
    lookup = {}
    json_files = list(metas_dir.rglob("train_label.json")) + list(metas_dir.rglob("test_label.json"))

    if json_files:
        for label_file in json_files:
            with open(label_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rel_path, vec in data.items():
                if len(vec) < 44:
                    continue
                try:
                    spoof_type_id = int(vec[40])
                    binary_id = int(vec[43])
                except (ValueError, TypeError):
                    continue
                key = "/".join(rel_path.replace("\\", "/").split("/")[-3:])
                lookup[key] = (SPOOF_TYPE_NAMES.get(spoof_type_id, f"unknown_{spoof_type_id}"), binary_id)
        print(f"label lookup: {len(json_files)} json dosyasindan {len(lookup)} kayit yuklendi.")
        return lookup

    label_files = list(metas_dir.rglob("train_label.txt")) + list(metas_dir.rglob("test_label.txt"))
    if not label_files:
        print(f"[UYARI] {metas_dir} altinda train_label.json/txt bulunamadi. "
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
                    binary_id = int(parts[43])
                except ValueError:
                    continue
                key = "/".join(rel_path.split("/")[-3:])
                lookup[key] = (SPOOF_TYPE_NAMES.get(spoof_type_id, f"unknown_{spoof_type_id}"), binary_id)

    print(f"label lookup: {len(label_files)} txt dosyasindan {len(lookup)} kayit yuklendi.")
    return lookup


def build_metadata(processed_dir: Path, label_lookup: dict) -> tuple:
    """Doner: (df, corrections_df). corrections_df, klasor-tabanli etiketin
    resmi JSON binary_label'iyla celistigi (ve JSON'a gore duzeltilen)
    goruntuleri listeler — bkz. modul docstring'indeki bilinen veri hatasi.
    """
    rows, corrections = [], []
    for _split, subject_id, folder_label, img_path in iter_split_subject_label_images(processed_dir):
        key = f"{subject_id}/{folder_label}/{img_path.name}"
        folder_label_id = 0 if folder_label == "live" else 1
        entry = label_lookup.get(key)

        if entry is not None:
            spoof_type, label_id = entry
            if label_id != folder_label_id:
                corrections.append({
                    "image_path": str(img_path),
                    "folder_label": folder_label,
                    "corrected_label": "live" if label_id == 0 else "spoof",
                    "spoof_type": spoof_type,
                })
        else:
            spoof_type = "Live" if folder_label == "live" else "unknown"
            label_id = folder_label_id

        rows.append({
            "image_path": str(img_path),
            "subject_id": subject_id,
            "label": "live" if label_id == 0 else "spoof",
            "label_id": label_id,
            "spoof_type": spoof_type,
        })

    if not rows:
        raise RuntimeError(
            f"{processed_dir} altinda hic goruntu bulunamadi. "
            f"02_extract_faces.py'nin cikti klasoru dogru mu, klasor yapisi "
            f"<split>/<subject_id>/<live|spoof>/<image> seklinde mi kontrol et."
        )

    return pd.DataFrame(rows), pd.DataFrame(corrections)


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

    label_lookup = {}
    if args.metas_dir:
        label_lookup = load_label_lookup(Path(args.metas_dir))

    print(f"Taraniyor: {processed_dir}")
    df, corrections_df = build_metadata(processed_dir, label_lookup)
    print(f"Toplam goruntu: {len(df)}, toplam subject: {df['subject_id'].nunique()}")

    if len(corrections_df) > 0:
        print(f"[DUZELTME] {len(corrections_df)} goruntude klasor-tabanli etiket resmi JSON "
              f"metadata'siyla celisiyordu; JSON binary_label'i esas alinarak duzeltildi.")
        corrections_path = output_dir / "label_corrections.csv"
        corrections_df.to_csv(corrections_path, index=False)
        print(f"Duzeltilen goruntulerin detayi: {corrections_path}")

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
