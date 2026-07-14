"""
02_extract_faces.py

01'de indirilen ham CelebA-Spoof goruntulerinde yuz kutusunu bulur, %20
margin ile crop'lar, 224x224'e resize eder ve SONUCU GOOGLE DRIVE'A yazar
(Colab session storage kalici degil — bkz. src/colab_utils.py). Boylece
Colab oturumu kapansa/koptugunda islenmis veri seti kaybolmaz, sonraki
oturumda --resume ile kaldigi yerden devam edilebilir.

Yuz kutusu bulma: ONCE her goruntunun yanindaki CelebA-Spoof'un hazir
'<isim>_BB.txt' dosyasi kullanilir (varsa) — bu durumda hicbir detector
modeli calismaz, ek paket (retina-face/facenet-pytorch) gerekmez, cok
hizlidir. BB.txt bulunamayan goruntuler icin (--detector none DEGILSE)
RetinaFace/MTCNN fallback'i devreye girer — bu, BB.txt saglamayan harici
veri setleri (orn. Faz A.3'teki LCC-FASD) icin onemli.

Beklenen girdi yapisi (resmi CelebA-Spoof, bkz. 01'in verify_extracted_structure
ciktisi):
    <input_dir>/Data/<split>/<subject_id>/<live|spoof>/<image>.jpg
    <input_dir>/Data/<split>/<subject_id>/<live|spoof>/<image>_BB.txt   (varsa)

Cikti yapisi (girdiyle birebir ayna, sadece goruntuler yuz-kirpilmis):
    <output_dir>/<split>/<subject_id>/<live|spoof>/<image>.jpg

Kullanim (Colab):
    python scripts/02_extract_faces.py --input_dir /content/celeba_spoof_raw
    # --output_dir verilmezse otomatik olarak Drive'a yazar
    # (/content/drive/MyDrive/passive-liveness-dinov2/processed)

Kullanim (local, Colab disi):
    python scripts/02_extract_faces.py --input_dir H:\\celeba_spoof_extracted --output_dir H:\\celeba_processed

    # Hizli smoke test (ilk 50 goruntu):
    python scripts/02_extract_faces.py --input_dir H:\\celeba_spoof_extracted --output_dir H:\\celeba_processed --limit 50
"""

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colab_utils import mount_drive, default_processed_dir
from src.face_crop import FaceDetector, preprocess_face
from src.scan_utils import iter_split_subject_label_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CelebA-Spoof yuz crop + resize pipeline'i")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="01'in cikti klasoru (icinde 'Data/' bulunmali).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Islenmis goruntulerin yazilacagi klasor (verilmezse Drive'a yazar).")
    parser.add_argument("--use_bb", type=lambda v: v.lower() != "false", default=True,
                         help="true/false — CelebA-Spoof'un hazir '_BB.txt' dosyalarini kullan (varsayilan: true, "
                              "cok daha hizli, detector modeli gerektirmez).")
    parser.add_argument("--detector", choices=["retinaface", "mtcnn", "none"], default="none",
                         help="BB.txt bulunamayan goruntuler icin fallback detector. 'none' ise "
                              "(varsayilan) BB.txt'siz goruntuler basarisiz sayilip atlanir — "
                              "CelebA-Spoof'ta bu, ekstra paket kurulumuna hic gerek kalmadigi anlamina gelir.")
    parser.add_argument("--device", default="cuda", help="mtcnn backend icin: cuda|cpu")
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test icin ilk N goruntuyle sinirla.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    data_root = input_dir / "Data"
    if not data_root.exists():
        data_root = input_dir / "data"
    if not data_root.exists():
        raise FileNotFoundError(
            f"'{input_dir}' altinda 'Data/' veya 'data/' klasoru bulunamadi. "
            f"01_download_celeba_spoof.py'nin verify_extracted_structure ciktisini kontrol et."
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        drive_root = mount_drive()
        output_dir = default_processed_dir(str(drive_root))
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Girdi: {data_root}")
    print(f"Cikti: {output_dir}")

    detector = None if args.detector == "none" else FaceDetector(backend=args.detector, device=args.device)

    all_images = list(iter_split_subject_label_images(data_root))
    if args.limit:
        all_images = all_images[: args.limit]
    print(f"Toplam goruntu: {len(all_images)}")
    print(f"BB.txt kullanimi: {args.use_bb}, fallback detector: {args.detector}")

    fail_log_path = output_dir / "extract_faces_failures.txt"
    n_skipped, n_ok, n_failed, n_used_bb = 0, 0, 0, 0

    with open(fail_log_path, "a") as fail_log:
        for split, subject_id, label, img_path in tqdm(all_images, desc="Yuz crop"):
            out_path = output_dir / split / subject_id / label / img_path.name
            if out_path.exists():
                n_skipped += 1
                continue

            bb_path = img_path.parent / f"{img_path.stem}_BB.txt" if args.use_bb else None
            if bb_path is not None and bb_path.exists():
                n_used_bb += 1

            try:
                face_img = preprocess_face(
                    str(img_path), detector, margin=args.margin, size=args.size, bb_path=bb_path
                )
            except Exception as e:  # noqa: BLE001 — tek bir bozuk goruntu tum run'i durdurmamali
                face_img = None
                fail_log.write(f"{img_path}\tEXCEPTION: {e}\n")

            if face_img is None:
                n_failed += 1
                fail_log.write(f"{img_path}\tNO_FACE_DETECTED\n")
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            face_img.save(out_path, quality=95)
            n_ok += 1

    print("\n--- Tamamlandi ---")
    print(f"Basarili:        {n_ok}  (BB.txt ile: {n_used_bb})")
    print(f"Atlandi (resume): {n_skipped}")
    print(f"Basarisiz (yuz bulunamadi/hata): {n_failed}  -> detaylar: {fail_log_path}")


if __name__ == "__main__":
    main()
