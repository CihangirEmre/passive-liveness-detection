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


def default_raw_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    """Ham indirilmis (henuz face-crop uygulanmamis) veri Drive'a degil,
    Colab session storage'a (/content) yazilir — buyuk (~10GB+) ve tekrar
    uretilebilir oldugu icin Drive kotasini gereksiz yere doldurmamak icin.
    """
    return Path("/content/celeba_spoof_raw")
