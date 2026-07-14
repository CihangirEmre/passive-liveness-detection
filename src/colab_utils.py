"""Colab / Google Drive yardimci fonksiyonlari.

scripts/01-04, islenmis (indirilmis, yuz-kirpilmis, split'lenmis) veri setini
kalici olmasi icin Google Drive'a yazar. Colab session storage (/content)
her oturum kapanisinda silinir; Drive kalici oldugu icin veri hazirlama
adimlarinin tekrar tekrar kosulmasi gerekmez.

Local'de (Colab disinda) calistirildiginda mount_drive() no-op'tur ve
DEFAULT_DRIVE_PROJECT_ROOT ile ayni gorece yapida bir local klasor doner
(smoke test / script gelistirme icin).
"""

from pathlib import Path

DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/passive-liveness-dinov2"


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    """Colab'da calisiyorsa Google Drive'i /content/drive'a baglar ve proje
    kok klasorunu (yoksa) olusturur. Local'de calisiyorsa sadece klasoru
    olusturur, mount islemi atlanir.
    """
    root = Path(project_root)

    if not is_colab():
        print(f"[colab_utils] Colab disinda calisiyor, Drive mount atlaniyor. Local kok: {root}")
        root.mkdir(parents=True, exist_ok=True)
        return root

    from google.colab import drive
    drive.mount("/content/drive")
    root.mkdir(parents=True, exist_ok=True)
    print(f"[colab_utils] Drive baglandi. Proje kok klasoru: {root}")
    return root


def default_processed_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    return Path(project_root) / "processed"


def default_splits_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    return Path(project_root) / "splits"


def default_raw_dir(project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> Path:
    """Ham indirilmis (henuz face-crop uygulanmamis) veri Drive'a degil,
    Colab session storage'a (/content) yazilir — buyuk (~10GB+) ve tekrar
    uretilebilir oldugu icin Drive kotasini gereksiz yere doldurmamak icin.
    """
    return Path("/content/celeba_spoof_raw")
