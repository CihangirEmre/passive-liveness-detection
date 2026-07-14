"""Yuz tespiti + marginli crop + resize — CelebA-Spoof ve (Faz A.3'te)
LCC-FASD icin ortak, dataset'ten bagimsiz preprocessing fonksiyonu.

Iki detector backend destekler:
- "retinaface" (varsayilan, `retina-face` paketi)
- "mtcnn"       (facenet-pytorch, TensorFlow'suz hafif alternatif)
"""

from typing import Optional, Tuple

from PIL import Image


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
    detector: FaceDetector,
    margin: float = 0.2,
    size: int = 224,
) -> Optional[Image.Image]:
    """Tam pipeline: yuz tespiti -> marginli crop -> size x size resize.
    Yuz bulunamazsa None doner (cagiran taraf loglayip atlamali).
    """
    image = Image.open(image_path).convert("RGB")
    bbox = detector.detect_largest_face_bbox(image_path)
    if bbox is None:
        return None

    cropped = crop_with_margin(image, bbox, margin=margin)
    if cropped.width == 0 or cropped.height == 0:
        return None

    return cropped.resize((size, size), Image.BILINEAR)
