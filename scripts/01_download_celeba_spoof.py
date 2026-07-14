"""
01_download_celeba_spoof.py

CelebA-Spoof veri setini Kaggle API uzerinden indirir ve acar (Kaggle mirror,
resmi Google Drive/Baidu dagitiminin yerine kullaniliyor). Colab'da (veya
local'de smoke test icin) calistirilabilir.

Ham veri BUYUK oldugu icin (10GB+) Colab session storage'a (/content) yazilir,
Drive'a degil — Drive'a sadece 02_extract_faces.py'nin islenmis (yuz-kirpilmis)
ciktisi yazilir (bkz. src/colab_utils.py).

Kaynak: https://www.kaggle.com/datasets/mabdullahsajid/celeba-spoofing
Resmi veri seti / label semasi referansi: https://github.com/ZhangYuanhan-AI/CelebA-Spoof

Onemli: Bu bir topluluk mirror'i oldugu icin klasor yapisinin resmi
CelebA-Spoof yapisiyla (Data/<split>/<subject_id>/<live|spoof>/*.jpg +
metas/.../train_label.txt) birebir ayni oldugu garanti degil.
verify_extracted_structure() indirme sonrasi gercek yapiyi yazdirir —
02/03 script'lerini calistirmadan once bu ciktiyi kontrol et.

On kosullar:
1. Kaggle hesabindan API token indirilmis olmali (kaggle.json)
   Kaggle -> Account -> API -> Create New Token
2. kaggle.json dosyasi ASLA repo'ya commit edilmemeli (.gitignore'da).

Kullanim (Colab):

    # kaggle.json'i Colab'a yukledikten sonra:
    python scripts/01_download_celeba_spoof.py \
        --kaggle_json /content/kaggle.json \
        --output_dir /content/celeba_spoof_raw
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colab_utils import default_raw_dir

DATASET_SLUG = "mabdullahsajid/celeba-spoofing"
KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
KAGGLE_JSON_TARGET = KAGGLE_CONFIG_DIR / "kaggle.json"


def setup_kaggle_credentials(kaggle_json_path: Path) -> None:
    if not kaggle_json_path.exists():
        raise FileNotFoundError(
            f"kaggle.json bulunamadi: {kaggle_json_path}\n"
            f"Kaggle hesabindan (Account -> API -> Create New Token) indirip "
            f"bu yola yuklediginden emin ol."
        )

    KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(kaggle_json_path, KAGGLE_JSON_TARGET)
    KAGGLE_JSON_TARGET.chmod(0o600)
    print(f"kaggle.json yerlestirildi: {KAGGLE_JSON_TARGET}")


def check_kaggle_credentials_exist() -> None:
    if not KAGGLE_JSON_TARGET.exists():
        raise FileNotFoundError(
            f"{KAGGLE_JSON_TARGET} bulunamadi.\n"
            f"Ya --kaggle_json parametresiyle dosya yolunu ver, "
            f"ya da kaggle.json'i manuel olarak {KAGGLE_CONFIG_DIR} altina koy."
        )


def ensure_kaggle_cli_installed() -> None:
    try:
        import kaggle  # noqa: F401
        return
    except ImportError:
        pass

    print("kaggle paketi bulunamadi, kuruluyor...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and "externally-managed-environment" in result.stderr:
        print("Externally-managed ortam tespit edildi, --break-system-packages ile tekrar deneniyor...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle", "--quiet", "--break-system-packages"],
            check=True,
        )
    elif result.returncode != 0:
        raise RuntimeError(f"kaggle paketi kurulamadi:\n{result.stderr}")


def download_dataset(output_dir: Path, dataset_slug: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_name = dataset_slug.split("/")[-1] + ".zip"
    zip_path = output_dir / zip_name

    if zip_path.exists():
        print(f"Zip zaten mevcut, indirme atlaniyor: {zip_path}")
        return zip_path

    print(f"Indiriliyor: {dataset_slug} -> {output_dir}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(output_dir)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        hint = ""
        if "403" in error_msg:
            hint = "\n[IPUCU] 403 hatasi genelde kaggle.json kimlik dogrulamasinin eksik/yanlis oldugunu gosterir."
        raise RuntimeError(f"Kaggle indirme basarisiz oldu:\n{error_msg}{hint}")

    print(result.stdout)

    if not zip_path.exists():
        zips = list(output_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"Indirme sonrasi zip dosyasi bulunamadi: {output_dir}")
        zip_path = zips[0]

    return zip_path


def compute_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"Aciliyor: {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Acma tamamlandi.")


def verify_extracted_structure(extract_to: Path, max_depth: int = 4) -> None:
    """Cikan klasor agacinda 'Data'/'data' ve 'metas' klasorlerini arar,
    bulunduklari yolu yazdirir. Bulunamazsa uyarir — bu durumda 02/03
    script'lerindeki path varsayimlari (Data/<split>/<subject_id>/<live|spoof>)
    gercek yapiya gore guncellenmeli.
    """
    print("\n--- Klasor yapisi kesfi ---")
    found_data, found_metas = [], []

    for p in extract_to.rglob("*"):
        if not p.is_dir():
            continue
        depth = len(p.relative_to(extract_to).parts)
        if depth > max_depth:
            continue
        if p.name.lower() == "data":
            found_data.append(p)
        elif p.name.lower() == "metas":
            found_metas.append(p)

    if found_data:
        for d in found_data:
            n_subdirs = sum(1 for _ in d.iterdir())
            print(f"Bulundu (Data): {d}  ({n_subdirs} alt oge)")
    else:
        print("[UYARI] 'Data' klasoru bulunamadi — resmi CelebA-Spoof yapisindan farkli olabilir.")

    if found_metas:
        for m in found_metas:
            print(f"Bulundu (metas): {m}")
    else:
        print("[UYARI] 'metas' klasoru bulunamadi — spoof_type/illumination/environment "
              "label'lari bu mirror'da olmayabilir; 03_build_splits.py binary "
              "live/spoof label'ina (klasor adindan) fallback yapacak.")

    print("--- Kesif tamamlandi, yukaridaki yollari gozden gecir ---\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CelebA-Spoof veri setini Kaggle'dan indirir.")
    parser.add_argument("--kaggle_json", type=str, default=None,
                         help="kaggle.json dosyasinin yolu (verilmezse ~/.kaggle/kaggle.json zaten var olmali).")
    parser.add_argument("--dataset_slug", type=str, default=DATASET_SLUG)
    parser.add_argument("--output_dir", type=str, default=str(default_raw_dir()),
                         help="Verinin indirilip acilacagi klasor.")
    parser.add_argument("--keep_zip", action="store_true",
                         help="Belirtilirse, acma isleminden sonra zip dosyasi silinmez.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ONEMLI: kimlik bilgileri ~/.kaggle/kaggle.json'a yerlestirilmeden ONCE
    # "import kaggle" CAGRILMAMALI — kaggle paketi import edilirken kendiliginden
    # kimlik dogrulamasi deniyor ve kaggle.json henuz yoksa/eskiyse burada patlar.
    if args.kaggle_json:
        setup_kaggle_credentials(Path(args.kaggle_json))
    else:
        check_kaggle_credentials_exist()

    ensure_kaggle_cli_installed()

    zip_path = download_dataset(output_dir, args.dataset_slug)
    print(f"Zip MD5: {compute_md5(zip_path)}  (kayit icin not al — sonraki indirmelerde karsilastirmak icin kullanilabilir)")

    extract_zip(zip_path, output_dir)
    verify_extracted_structure(output_dir)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Zip silindi: {zip_path}")

    print("\n--- Tamamlandi ---")
    print(f"Veri klasoru: {output_dir}")


if __name__ == "__main__":
    main()
