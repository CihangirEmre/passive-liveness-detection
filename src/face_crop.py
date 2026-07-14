"""Yuz tespiti + marginli crop + resize — CelebA-Spoof ve (Faz A.3'te)
LCC-FASD icin ortak, dataset'ten bagimsiz preprocessing fonksiyonu.

CelebA-Spoof icin HIZLI YOL: CelebA-Spoof zaten her goruntu icin onceden
hesaplanmis bir yuz kutusu (`<isim>_BB.txt`) saglıyor — bu varsa
read_bb_file() ile dogrudan kullanilir, bir detector modeli CALISTIRMAYA
GEREK KALMAZ (ne ek paket kurulumu ne GPU inference).

Detector (RetinaFace/MTCNN) SADECE BB.txt bulunamayan durumlar icin
fallback olarak devrede — orn. Faz A.3'teki LCC-FASD gibi BB.txt
saglamayan harici veri setleri icin.

Iki detector backend destekler:
- "retinaface" (varsayilan, `retina-face` paketi)
- "mtcnn"       (facenet-pytorch, TensorFlow'suz hafif alternatif)
"""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


def read_bb_file(bb_path: Path, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """CelebA-Spoof'un onceden hesapladigi '_BB.txt' dosyasini okur.

    Format (dogrulandi, gercek ornek uzerinde gorsel kontrol edildi):
        "<x> <y> <w> <h> <score>"
    Koordinatlar 224x224'luk SABIT bir referans cerceveye gore olcekli —
    gercek goruntu boyutuna gore (img_w/224, img_h/224 ile, eksen basina
    ayri ayri) yeniden olceklenmesi gerekiyor.

    image_size: (width, height) — orijinal goruntunun GERCEK boyutu.
    Doner: (x1, y1, x2, y2) mutlak piksel koordinatlari, dosya
    okunamazsa/gecersizse None.
    """
    try:
        with open(bb_path, "r") as f:
            line = f.readline().strip()
        if not line:
            return None
        parts = line.split()
        if len(parts) < 4:
            return None
        bx, by, bw, bh = (float(v) for v in parts[:4])
    except (OSError, ValueError):
        return None

    img_w, img_h = image_size
    x1 = bx * img_w / 224
    y1 = by * img_h / 224
    x2 = x1 + bw * img_w / 224
    y2 = y1 + bh * img_h / 224

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


class FaceDetector:
    def __init__(self, backend: str = "retinaface", device: str = "cpu"):
        if backend not in ("retinaface", "mtcnn"):
            raise ValueError(f"Desteklenmeyen backend: {backend} (retinaface|mtcnn)")
        self.backend = backend
        self._device = device
        self._detector = None

    def _lazy_init(self) -> None:
        if self._detector is not None:
            return
        if self.backend == "retinaface":
            from retinaface import RetinaFace
            self._detector = RetinaFace
        else:
            from facenet_pytorch import MTCNN
            self._detector = MTCNN(keep_all=False, device=self._device)

    def detect_largest_face_bbox(self, image_path: str) -> Optional[Tuple[int, int, int, int]]:
        """En buyuk (alan) yuz kutusunu (x1, y1, x2, y2) olarak doner.
        Yuz bulunamazsa None doner.
        """
        self._lazy_init()

        if self.backend == "retinaface":
            resp = self._detector.detect_faces(image_path)
            if not isinstance(resp, dict) or len(resp) == 0:
                return None
            boxes = [face["facial_area"] for face in resp.values()]
        else:
            img = Image.open(image_path).convert("RGB")
            boxes, _ = self._detector.detect(img)
            if boxes is None:
                return None
            boxes = boxes.tolist()

        def area(b):
            return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

        best = max(boxes, key=area)
        return tuple(int(round(v)) for v in best)


def crop_with_margin(image: Image.Image, bbox: Tuple[int, int, int, int], margin: float = 0.2) -> Image.Image:
    """bbox = (x1, y1, x2, y2). Her kenara bbox genislik/yuksekliginin
    `margin` orani kadar pay birakir, goruntu sinirlarina clip eder.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin

    x1 = max(0, int(round(x1 - mx)))
    y1 = max(0, int(round(y1 - my)))
    x2 = min(image.width, int(round(x2 + mx)))
    y2 = min(image.height, int(round(y2 + my)))

    return image.crop((x1, y1, x2, y2))


def preprocess_face(
    image_path: str,
    detector: Optional[FaceDetector] = None,
    margin: float = 0.2,
    size: int = 224,
    bb_path: Optional[Path] = None,
) -> Optional[Image.Image]:
    """Tam pipeline: yuz kutusu bulma -> marginli crop -> size x size resize.

    Yuz kutusu bulma once bb_path (varsa, CelebA-Spoof'un hazir '_BB.txt'si)
    dener; orada bulunamaz/gecersizse `detector` ile tespit yapar (verildiyse).
    Ikisi de basarisiz olursa None doner (cagiran taraf loglayip atlamali).
    """
    image = Image.open(image_path).convert("RGB")

    bbox = None
    if bb_path is not None and Path(bb_path).exists():
        bbox = read_bb_file(Path(bb_path), image.size)

    if bbox is None:
        if detector is None:
            return None
        bbox = detector.detect_largest_face_bbox(image_path)
        if bbox is None:
            return None

    cropped = crop_with_margin(image, bbox, margin=margin)
    if cropped.width == 0 or cropped.height == 0:
        return None

    return cropped.resize((size, size), Image.BILINEAR)
