"""02_extract_faces.py ve 03_build_splits.py'nin ORTAK dizin tarama mantigi.

Ikisi de ayni klasor sozlesmesine (convention) gore calisir:
    <root>/<split>/<subject_id>/<live|spoof>/<image>
Bu dosya tek bir yerden tanimlanip iki script'te de import edilir ki
sozlesme sapmasi (biri degisip digeri degismemesi) riski olmasin.
"""

from pathlib import Path
from typing import Iterator, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def iter_split_subject_label_images(root: Path) -> Iterator[Tuple[str, str, str, Path]]:
    """<root>/<split>/<subject_id>/<live|spoof>/<image> desenini tarar.
    (split, subject_id, label, image_path) tuple'lari uretir.
    """
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for label_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                if label_dir.name.lower() not in ("live", "spoof"):
                    continue
                for img_path in sorted(label_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTS:
                        yield split_dir.name, subject_dir.name, label_dir.name.lower(), img_path
