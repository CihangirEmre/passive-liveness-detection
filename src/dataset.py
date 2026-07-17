"""CelebA-Spoof split CSV'lerinden (03_build_splits.py ciktisi) PyTorch
Dataset olusturur.

CSV'lerdeki 'image_path' kolonu, CSV'nin URETILDIGI makineye ozel MUTLAK
yoldur (orn. yerelde H:\\celeba_processed_dedup\\...). Bu yuzden goruntu
yolu dogrudan kullanilmaz — sadece dosya adi alinip --images_root + split
adi + subject_id + label ile YENIDEN insa edilir. Boylece ayni CSV hem
local'de hem Colab'da (Drive mount edilmis farkli bir kok altinda)
degistirilmeden calisir.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(train: bool, size: int = 224) -> transforms.Compose:
    ops = [transforms.Resize((size, size))]
    if train:
        # Plan kisidi: agresif ColorJitter/grayscale KULLANMA (liveness sinyali
        # texture/renkte tasinir) — sadece hafif flip + hafif renk sarsintisi.
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class CelebASpoofSplitDataset(Dataset):
    """NOT: 'image_path' kolonundaki klasor, bizim train/val/test split'imiz
    DEGIL — CelebA-Spoof'un orijinal (Data/<split>/...) klasor adidir, cunku
    02/02b bu klasor yapisini degistirmeden tasir. Ayrica 03_build_splits.py
    JSON-tabanli etiket duzeltmesi yaptiginda (bkz. label_corrections.csv)
    'label' kolonu degisebilir ama goruntu fiziksel olarak TASINMAZ — bu
    yuzden diskteki gercek konum icin image_path'in kendi son parcasi
    kullanilir, egitim etiketi icin ise (duzeltilmis) 'label_id' kolonu.
    """

    def __init__(
        self,
        csv_path: str,
        images_root: str,
        transform: Optional[transforms.Compose] = None,
        limit: Optional[int] = None,
    ):
        df = pd.read_csv(csv_path, dtype={"subject_id": str})
        if limit:
            df = df.iloc[:limit].reset_index(drop=True)
        self.df = df
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        normalized = str(row["image_path"]).replace("\\", "/")
        orig_split, _subject_from_path, physical_label, filename = normalized.split("/")[-4:]
        img_path = self.images_root / orig_split / row["subject_id"] / physical_label / filename

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_id"])


class ManifestPairDataset(Dataset):
    """LCC-FASD tarzi veri setleri icin: live ve spoof goruntulerin yollari
    AYRI iki metin dosyasinda (her satirda bir yol) listelenir — CelebA-Spoof
    gibi klasor/CSV yapisi YOK.

    NOT (kritik): LCC-FASD'in Kaggle sayfasindaki ornek kod client(genuine)=1,
    imposter(spoof)=0 kullaniyor — bu projenin BASTAN BERI kullandigi
    sozlesmenin (0=live, 1=spoof; bkz. 03_build_splits.py, metrics.py) TAM
    TERSI. Bu yuzden burada bilerek live_manifest->0, spoof_manifest->1
    olarak eslenir; Kaggle'in kendi ornek kodu DOGRUDAN kopyalanmamalidir.

    Manifest'teki yollar genelde baska bir ortama (orn. Kaggle notebook'unun
    /kaggle/input/... mount noktasi) ait MUTLAK yollardir ve Colab'da
    calismaz — bu yuzden sadece dosya ADI alinip --images_root ile YENIDEN
    insa edilir (CelebASpoofSplitDataset'teki portabilite yaklasimiyla ayni).
    """

    def __init__(
        self,
        live_manifest: str,
        spoof_manifest: str,
        images_root: str,
        transform: Optional[transforms.Compose] = None,
        limit: Optional[int] = None,
    ):
        live_paths = [
            line.strip() for line in Path(live_manifest).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        spoof_paths = [
            line.strip() for line in Path(spoof_manifest).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.items = [(p, 0) for p in live_paths] + [(p, 1) for p in spoof_paths]
        if limit:
            self.items = self.items[:limit]
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        raw_path, label = self.items[idx]
        filename = Path(raw_path.replace("\\", "/")).name
        img_path = self.images_root / filename

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
