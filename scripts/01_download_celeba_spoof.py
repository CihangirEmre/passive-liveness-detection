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
import csv
import hashlib
import random
import shutil
import subprocess
import sys
import uuid
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

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


def _normalized_rel_path(name: str) -> Path:
    """Bir dosya yolunu ('CelebA_Spoof/Data/train/10001/live/x.png' gibi)
    'Data'/'data' veya 'metas' klasorunden ITIBAREN gorece yola cevirir —
    ustteki 'CelebA_Spoof/' gibi sarmalayici klasor adi ne olursa olsun
    (degisebilir, mirror'a gore farkli olabilir) SONUC HEP AYNI olur:
    <root>/Data/... veya <root>/metas/... Bu normalizasyon olmadan,
    zip-mode ve files-mode farkli klasor derinliklerinde cikti uretebilir
    ve 02_extract_faces.py'nin 'input_dir/Data' aramasi basarisiz olur.
    """
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    lower_parts = [p.lower() for p in parts]
    for anchor in ("data", "metas"):
        if anchor in lower_parts:
            idx = lower_parts.index(anchor)
            return Path(*parts[idx:])
    return Path(*parts)


def extract_zip(zip_path: Path, extract_to: Path, max_per_group: int = None, seed: int = 42) -> None:
    """max_per_group verilmezse (None/0) zip'in TAMAMINI acar.

    Verilirse, SECICI acma yapar: her (split, subject_id, label) grubundan en
    fazla `max_per_group` goruntuyu (seed'li rastgele secim) zip'in icinden
    diske cikarir, geri kalanina hic dokunmaz. Bu, disk dolmasini onlemenin
    yani sira fine-tuning veri hacmini de indirme asamasinda azaltir — CelebA-
    Spoof'un TAMAMINI (625K+ goruntu) diske acmaya gerek kalmaz.
    metas/ gibi goruntu-disi dosyalar (label txt'leri) her zaman TAMAMEN
    cikarilir (kucukler, kaybedilmemeli).

    Cikti yollari _normalized_rel_path ile normalize edilir (zip'in ust
    seviye sarmalayici klasoru ne olursa olsun cikti hep <extract_to>/Data/...
    ve <extract_to>/metas/... olur).
    """
    def _extract_one(zf: zipfile.ZipFile, name: str) -> None:
        dest = extract_to / _normalized_rel_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if not max_per_group:
        print(f"Aciliyor (TAMAMI, path normalize edilerek): {zip_path} -> {extract_to}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.endswith("/"):
                    _extract_one(zf, name)
        print("Acma tamamlandi.")
        return

    print(f"Aciliyor (SECICI, grup basina en fazla {max_per_group} goruntu): {zip_path} -> {extract_to}")
    rng = random.Random(seed)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        # '_BB.txt' dosyalari ayri tutulur — SADECE secilen goruntunun BB'si
        # cikarilir, tum veri setinin BB'leri degil (aksi halde ~600K kucuk
        # dosya gereksiz yere cikarilir).
        groups = defaultdict(list)
        always_include = []
        bb_lookup = set()

        for name in names:
            if name.endswith("_BB.txt"):
                bb_lookup.add(name)
                continue
            classified = _classify_entry(name)
            if classified is None:
                always_include.append(name)
                continue
            split, subject_id, label, _filename = classified
            groups[(split, subject_id, label)].append(name)

        print(f"Toplam dosya: {len(names)}, gruplanan (subject x label) sayisi: {len(groups)}, "
              f"her zaman dahil edilen (metas vb.): {len(always_include)}, BB.txt havuzu: {len(bb_lookup)}")

        selected = list(always_include)
        n_images = 0
        for entries in groups.values():
            entries_sorted = sorted(entries)
            n_pick = min(len(entries_sorted), max_per_group)
            for image_name in rng.sample(entries_sorted, n_pick):
                selected.append(image_name)
                n_images += 1
                stem_dir = image_name.rsplit("/", 1)[0]
                bb_name = f"{stem_dir}/{Path(image_name).stem}_BB.txt"
                if bb_name in bb_lookup:
                    selected.append(bb_name)

        print(f"Cikarilacak toplam dosya: {len(selected)} "
              f"(secilen goruntu: {n_images}, eslesen BB.txt: {len(selected) - n_images - len(always_include)}, "
              f"her zaman dahil: {len(always_include)})")

        for name in selected:
            _extract_one(zf, name)

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


def list_dataset_files(dataset_slug: str, cache_csv_path: Path) -> list:
    """Kaggle'dan (VERI INDIRMEDEN, sadece dosya listesi/meta veri) TAM dosya
    listesini ceker ve yerel bir CSV'ye onbelleklendirir — script kesilip
    tekrar calistirilirsa (resume) ayni listeyi tekrar cekmeye gerek kalmaz.
    """
    if cache_csv_path.exists():
        print(f"Dosya listesi onbellekte bulundu, tekrar cekilmiyor: {cache_csv_path}")
    else:
        print(f"Dosya listesi cekiliyor (tek seferlik meta veri istegi, veri indirmez): {dataset_slug} ...")
        result = subprocess.run(
            ["kaggle", "datasets", "files", "-d", dataset_slug, "--csv"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Dosya listesi cekilemedi:\n{result.stderr.strip()}")
        cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
        cache_csv_path.write_text(result.stdout, encoding="utf-8")

    with open(cache_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Toplam {len(rows)} dosya listelendi.")
    return rows


def build_file_selection(rows: list, max_per_group: int, seed: int, include_bb: bool):
    """Dosya listesini (split, subject_id, label) gruplarina ayirir, her
    gruptan en fazla max_per_group goruntuyu secer. include_bb=True ise
    secilen her goruntunun eslesen '<isim>_BB.txt' (CelebA-Spoof'un onceden
    hesapladigi yuz kutusu) dosyasini da secime ekler — boylece 02'de kendi
    yuz tespitimizi calistirmaya gerek kalmayabilir.

    Doner: (secilen_dosya_yollari, secilen_goruntu_sayisi, grup_sayisi)
    """
    all_names = {row["name"] for row in rows}
    groups = defaultdict(list)

    for row in rows:
        name = row["name"]
        if name.endswith("_BB.txt"):
            continue
        classified = _classify_entry(name)
        if classified is None:
            continue
        split, subject_id, label, _filename = classified
        groups[(split, subject_id, label)].append(name)

    rng = random.Random(seed)
    selected = []
    n_images = 0

    for entries in groups.values():
        entries_sorted = sorted(entries)
        n_pick = min(len(entries_sorted), max_per_group)
        for image_path in rng.sample(entries_sorted, n_pick):
            selected.append(image_path)
            n_images += 1
            if include_bb:
                stem = Path(image_path).stem
                bb_path = image_path.rsplit("/", 1)[0] + f"/{stem}_BB.txt"
                if bb_path in all_names:
                    selected.append(bb_path)

    metas_paths = [row["name"] for row in rows if "metas/" in row["name"].lower()]
    selected.extend(metas_paths)

    return selected, n_images, len(groups)


def download_single_file(dataset_slug: str, file_path: str, scratch_dir: Path, final_path: Path) -> bool:
    """Kaggle'dan TEK bir dosyayi indirir, normalize edilmis final_path'e
    yerlestirir. kaggle CLI tek-dosya indirmede bazen zip'e sarabildigi icin
    (versiyona gore degisebilir) her iki durumu da ele alir — dosyayi
    indirdikten sonra ADINA gore bulup kendimiz normalize edilmis konuma
    tasiyoruz (kaggle'in -p altinda tam olarak nereye yazdigina guvenmiyoruz).
    """
    if final_path.exists():
        return True  # resume: zaten indirilmis

    job_scratch = scratch_dir / uuid.uuid4().hex
    job_scratch.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-f", file_path,
             "-p", str(job_scratch), "--force"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return False

        basename = Path(file_path).name
        candidates = list(job_scratch.rglob(f"{basename}*"))
        if not candidates:
            return False
        src = candidates[0]

        final_path.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".zip":
            with zipfile.ZipFile(src, "r") as zf:
                names = zf.namelist()
                if not names:
                    return False
                with zf.open(names[0]) as zsrc, open(final_path, "wb") as fdst:
                    shutil.copyfileobj(zsrc, fdst)
        else:
            shutil.move(str(src), str(final_path))

        return True
    finally:
        shutil.rmtree(job_scratch, ignore_errors=True)


def run_files_mode(args: argparse.Namespace) -> None:
    """--mode files: 78GB'lik zip'i HIC indirmeden, sadece secilen dosyalari
    Kaggle'dan tek tek (paralel, thread pool ile) ceker. Disk kullanimi
    secilen alt kume kadardir — zip hicbir zaman diske/Drive'a inmez.
    """
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.file_list_cache) if args.file_list_cache else output_dir.parent / "_file_list_cache"
    cache_csv_path = cache_dir / f"{args.dataset_slug.replace('/', '_')}_files.csv"

    rows = list_dataset_files(args.dataset_slug, cache_csv_path)
    selected, n_images, n_groups = build_file_selection(
        rows, max_per_group=args.max_per_group, seed=args.seed, include_bb=not args.skip_bb
    )
    print(f"Secildi: {n_images} goruntu, {n_groups} grup (subject x label), "
          f"toplam {len(selected)} dosya (BB.txt ve metas dahil).")

    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir.parent / "_download_scratch"
    fail_log_path = output_dir / "download_failures.txt"
    n_ok, n_failed = 0, 0

    with open(fail_log_path, "a") as fail_log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_single_file, args.dataset_slug, file_path, scratch_dir,
                        output_dir / _normalized_rel_path(file_path)): file_path
            for file_path in selected
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Tek tek indirme"):
            file_path = futures[future]
            try:
                ok = future.result()
            except Exception as e:  # noqa: BLE001 — tek dosyanin hatasi tum run'i durdurmamali
                ok = False
                fail_log.write(f"{file_path}\tEXCEPTION: {e}\n")
            if ok:
                n_ok += 1
            else:
                n_failed += 1
                fail_log.write(f"{file_path}\tDOWNLOAD_FAILED\n")

    shutil.rmtree(scratch_dir, ignore_errors=True)

    print("\n--- Tamamlandi (files mode) ---")
    print(f"Basarili: {n_ok} / {len(selected)}")
    print(f"Basarisiz: {n_failed}  -> detaylar: {fail_log_path}")
    print(f"Cikti klasoru: {output_dir}")

    verify_extracted_structure(output_dir)


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
    parser.add_argument("--max_per_group", type=int, default=5,
                         help="Her (subject_id, label) grubundan indirilecek/cikarilacak MAKSIMUM goruntu "
                              "sayisi (disk tasmasini onlemek + fine-tuning veri hacmini azaltmak icin). "
                              "'zip' modunda 0 verilirse siniri kaldirir, zip'in TAMAMI acilir.")
    parser.add_argument("--seed", type=int, default=42, help="Grup ici rastgele secim icin seed.")
    parser.add_argument("--compute_md5", action="store_true",
                         help="[zip modu] Belirtilirse zip'in MD5'ini hesaplar (78GB'lik dosyada Drive "
                              "uzerinden yavas olabilir, varsayilan olarak kapali).")
    parser.add_argument("--mode", choices=["files", "zip"], default="files",
                         help="'files' (varsayilan): 78GB'lik zip'i HIC indirmeden, sadece secilen dosyalari "
                              "Kaggle'dan tek tek ceker (disk-guvenli ama COK dosya oldugu icin yavas). "
                              "'zip': tum zip'i indirip SECICI acar (hizli ama Drive'da ~80GB bos alan gerekir "
                              "ve bazi ortamlarda zip indirmesi yine yerel diski doldurabiliyor).")
    parser.add_argument("--workers", type=int, default=8,
                         help="[files modu] Paralel indirme icin thread sayisi.")
    parser.add_argument("--skip_bb", action="store_true",
                         help="[files modu] Belirtilirse, goruntulerin '_BB.txt' (onceden hesaplanmis yuz "
                              "kutusu) dosyalarini indirmez. Varsayilan: indirilir (kucuk, faydali).")
    parser.add_argument("--file_list_cache", type=str, default=None,
                         help="[files modu] Dosya listesi CSV'sinin onbelleklenecegi klasor "
                              "(verilmezse output_dir'in yaninda otomatik olusturulur).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ONEMLI: kimlik bilgileri ~/.kaggle/kaggle.json'a yerlestirilmeden ONCE
    # "import kaggle" CAGRILMAMALI — kaggle paketi import edilirken kendiliginden
    # kimlik dogrulamasi deniyor ve kaggle.json henuz yoksa/eskiyse burada patlar.
    if args.kaggle_json:
        setup_kaggle_credentials(Path(args.kaggle_json))
    else:
        check_kaggle_credentials_exist()

    ensure_kaggle_cli_installed()

    if args.mode == "files":
        run_files_mode(args)
        return

    output_dir = Path(args.output_dir)

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
