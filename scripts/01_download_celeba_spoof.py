"""
01_download_celeba_spoof.py

CelebA-Spoof veri setini Kaggle API uzerinden indirir ve acar (Kaggle mirror,
resmi Google Drive/Baidu dagitiminin yerine kullaniliyor). Colab'da (veya
local'de smoke test icin) calistirilabilir.

Gercek boyut ~78GB (Kaggle Data Explorer'dan dogrulandi) — bu, Colab'in
tipik yerel diskine (~60-110GB, cogunlukla zaten kismen dolu) SIGMAYABILIR.
Bu yuzden zip dosyasinin KENDISI varsayilan olarak Google Drive'a indirilir
(bkz. src/colab_utils.default_raw_zip_dir), yerel /content'e degil — zipfile
modulu Drive-mount edilmis dosyayi normal bir yol gibi okuyabiliyor. SADECE
secilen (max_per_group ile sinirli) kucuk alt kume yerel /content'e
cikarilir (hizli okuma icin). Drive'da en az ~80GB bos alan olmali.

Varsayilan olarak zip'in TAMAMI acilmaz (625K+ goruntu Colab'in yerel diskini
doldurabilir) — bunun yerine --max_per_group (varsayilan 20) ile her
(subject_id, label) grubundan en fazla N goruntu SECILEREK cikarilir. Bu hem
disk sorununu cozer hem de fine-tuning veri hacmini indirme asamasinda
azaltir. Tam veri seti isteniyorsa --max_per_group 0 verilmeli.

Kullanim (Colab):

    # kaggle.json'i Colab'a yukledikten ve Drive'i mount ettikten sonra:
    python scripts/01_download_celeba_spoof.py \
        --kaggle_json /content/kaggle.json \
        --max_per_group 20
"""

import argparse
import hashlib
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colab_utils import default_raw_dir, default_raw_zip_dir, mount_drive

DATASET_SLUG = "mabdullahsajid/celeba-spoofing"
KAGGLE_CONFIG_DIR = Path.home() / ".kaggle"
KAGGLE_JSON_TARGET = KAGGLE_CONFIG_DIR / "kaggle.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


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


def _classify_entry(name: str):
    """Zip entry adini ayristirir. 'Data/<split>/<subject_id>/<live|spoof>/<file>'
    desenine uyan bir goruntuyse (split, subject_id, label, filename) doner,
    uymuyorsa None doner (metas/ dosyalari, ust seviye dosyalar vb. icin).
    """
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    lower_parts = [p.lower() for p in parts]

    if "data" not in lower_parts:
        return None
    data_idx = lower_parts.index("data")
    remainder = parts[data_idx + 1:]
    if len(remainder) != 4:
        return None

    split, subject_id, label, filename = remainder
    if label.lower() not in ("live", "spoof"):
        return None
    if Path(filename).suffix.lower() not in IMAGE_EXTS:
        return None

    return split, subject_id, label.lower(), filename


def extract_zip(zip_path: Path, extract_to: Path, max_per_group: int = None, seed: int = 42) -> None:
    """max_per_group verilmezse (None/0) zip'in TAMAMINI acar.

    Verilirse, SECICI acma yapar: her (split, subject_id, label) grubundan en
    fazla `max_per_group` goruntuyu (seed'li rastgele secim) zip'in icinden
    diske cikarir, geri kalanina hic dokunmaz. Bu, disk dolmasini onlemenin
    yani sira fine-tuning veri hacmini de indirme asamasinda azaltir — CelebA-
    Spoof'un TAMAMINI (625K+ goruntu) diske acmaya gerek kalmaz.
    metas/ gibi goruntu-disi dosyalar (label txt'leri) her zaman TAMAMEN
    cikarilir (kucukler, kaybedilmemeli).
    """
    if not max_per_group:
        print(f"Aciliyor (TAMAMI): {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        print("Acma tamamlandi.")
        return

    print(f"Aciliyor (SECICI, grup basina en fazla {max_per_group} goruntu): {zip_path} -> {extract_to}")
    rng = random.Random(seed)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        groups = defaultdict(list)
        non_image_names = []

        for name in names:
            classified = _classify_entry(name)
            if classified is None:
                non_image_names.append(name)
                continue
            split, subject_id, label, _filename = classified
            groups[(split, subject_id, label)].append(name)

        print(f"Toplam dosya: {len(names)}, goruntu-disi/eslesmeyen: {len(non_image_names)}, "
              f"gruplanan (subject x label) sayisi: {len(groups)}")

        selected = list(non_image_names)
        for entries in groups.values():
            entries_sorted = sorted(entries)
            n_pick = min(len(entries_sorted), max_per_group)
            selected.extend(rng.sample(entries_sorted, n_pick))

        print(f"Cikarilacak toplam dosya: {len(selected)} "
              f"(goruntu-disi {len(non_image_names)} dahil, tahmini goruntu: {len(selected) - len(non_image_names)})")

        for name in selected:
            zf.extract(name, extract_to)

    print("Secici acma tamamlandi.")


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
    parser.add_argument("--zip_dir", type=str, default=None,
                         help="Zip dosyasinin (~78GB) indirilecegi klasor. Verilmezse Google Drive'daki "
                              "varsayilan yol kullanilir (yerel /content diskine sigmayabilecek kadar buyuk).")
    parser.add_argument("--output_dir", type=str, default=str(default_raw_dir()),
                         help="SECILEN (max_per_group ile sinirli) alt kumenin cikarilacagi yerel klasor.")
    parser.add_argument("--keep_zip", action="store_true",
                         help="Belirtilirse, acma isleminden sonra zip dosyasi silinmez "
                              "(farkli bir --max_per_group ile tekrar denemek icin yeniden indirmeyi onler; "
                              "zip Drive'da oldugu icin yerel diski etkilemez, sadece Drive kotasini kullanir).")
    parser.add_argument("--max_per_group", type=int, default=20,
                         help="Her (subject_id, label) grubundan diske cikarilacak MAKSIMUM goruntu sayisi "
                              "(disk tasmasini onlemek + fine-tuning veri hacmini azaltmak icin). "
                              "0 verilirse siniri kaldirir, zip'in TAMAMI acilir.")
    parser.add_argument("--seed", type=int, default=42, help="Secici acmada grup ici rastgele secim icin seed.")
    parser.add_argument("--compute_md5", action="store_true",
                         help="Belirtilirse zip'in MD5'ini hesaplar (78GB'lik dosyada Drive uzerinden yavas "
                              "olabilir, varsayilan olarak kapali).")
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

    if args.zip_dir:
        zip_dir = Path(args.zip_dir)
    else:
        drive_root = mount_drive()
        zip_dir = default_raw_zip_dir(str(drive_root))

    zip_path = download_dataset(zip_dir, args.dataset_slug)

    if args.compute_md5:
        print(f"Zip MD5: {compute_md5(zip_path)}  (kayit icin not al — sonraki indirmelerde karsilastirmak icin kullanilabilir)")

    extract_zip(zip_path, output_dir, max_per_group=args.max_per_group, seed=args.seed)
    verify_extracted_structure(output_dir)

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Zip silindi: {zip_path}")
    else:
        print(f"Zip saklandi (Drive kotasini kullaniyor, yerel diski etkilemiyor): {zip_path}")

    print("\n--- Tamamlandi ---")
    print(f"Zip klasoru (Drive): {zip_dir}")
    print(f"Secilen alt kume (yerel): {output_dir}")


if __name__ == "__main__":
    main()
