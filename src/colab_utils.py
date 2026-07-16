"""Colab / Google Drive yardimci fonksiyonlari.

scripts/01-04, islenmis (indirilmis, yuz-kirpilmis, split'lenmis) veri setini
kalici olmasi icin Google Drive'a yazar. Colab session storage (/content)
her oturum kapanisinda silinir; Drive kalici oldugu icin veri hazirlama
adimlarinin tekrar tekrar kosulmasi gerekmez.

Local'de (Colab disinda) calistirildiginda mount_drive() no-op'tur ve
DEFAULT_DRIVE_PROJECT_ROOT ile ayni gorece yapida bir local klasor doner
(smoke test / script gelistirme icin).

ONEMLI: drive.mount() SADECE dogrudan bir notebook hucresinde calisir.
"!python scripts/xxx.py" ile calistirilan script'ler ayri bir alt-process'te
kosuyor ve Colab kernel'inin frontend'e mesaj gonderme kanalina erisemiyor —
bu yuzden mount_drive() BURADA drive.mount() CAGIRMAZ, sadece Drive'in
ONCEDEN (bir notebook hucresinde) mount edilmis olup olmadigini kontrol eder.
"""

from pathlib import Path

DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/passive-liveness-dinov2"
DRIVE_MOUNT_POINT = "/content/drive/MyDrive"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    """Google Drive'in mount edilmis oldugunu dogrular ve proje kok klasorunu
    (yoksa) olusturur. Local'de calisiyorsa sadece klasoru olusturur.

    Drive henuz mount edilmemisse RuntimeError firlatir — cozum icin script'i
    calistirmadan ONCE, ayri bir notebook hucresinde su iki satiri calistirin:

        from google.colab import drive
        drive.mount('/content/drive')
    """
    root = Path(project_root)

    if not is_colab():
        print(f"[colab_utils] Colab disinda calisiyor, Drive mount kontrolu atlaniyor. Local kok: {root}")
        root.mkdir(parents=True, exist_ok=True)
        return root

    root.mkdir(parents=True, exist_ok=True)
    print(f"[colab_utils] Drive bagli. Proje kok klasoru: {root}")
    return root


def default_processed_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    return Path(project_root) / "processed"


def default_splits_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    return Path(project_root) / "splits"


def default_processed_dedup_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    return Path(project_root) / "processed_dedup"


def _warn_if_local_content_path(path: Path) -> Path:
    """default_* fonksiyonlari /content/... gibi Colab'a ozel mutlak yollar
    donduruyor. Colab disinda (Windows/local) bu yol sessizce sürücü kokune
    (orn. C:\\content\\...) yazar — script'i --output_dir vermeden local
    calistirmanin bilinen bir tuzagi (bkz. dinov2_liveness_plan.md RISKLER).
    Bu fonksiyon sadece is_colab()=False iken bir uyari basar, davranisi
    degistirmez (var olan cagrilari bozmamak icin)."""
    if not is_colab():
        print(f"[colab_utils] UYARI: Colab disindasin ama '{path}' gibi /content'e "
              f"ozel bir varsayilan yol kullaniliyor — local'de bu, mevcut surucu "
              f"kokune yazar (orn. C:\\content\\...). Yanlislikla sistem surucusunu "
              f"doldurmamak icin script'i --output_dir/--zip_dir ile ACIKCA "
              f"gecilen bir yerel yolla calistir.")
    return path


def default_raw_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    """SECILEREK cikarilan (max_per_group ile sinirli) ham goruntuler buraya,
    Colab session storage'a (/content) yazilir — kucuk bir alt kume oldugu
    icin (tum veri seti degil) yerel disk sorun cikarmaz, hizli okuma saglar.
    """
    return _warn_if_local_content_path(Path("/content/celeba_spoof_raw"))


def default_raw_zip_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    """CelebA-Spoof zip dosyasinin (~78GB) TAMAMI Colab'in yerel diskine
    (tipik olarak ~60-110GB) sigmiyor. Bu yuzden zip, yerel /content yerine
    Drive'a indirilir (Drive kotanizda yeterli alan olmasi gerekir) —
    zipfile modulu Drive-mount edilmis dosyayi normal bir yol gibi
    okuyabildigi icin secici extraction buradan calisir.
    """
    return Path(project_root) / "raw_zip"
