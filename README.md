# Passive Liveness Detection — DINOv2 → Edge Distillation

Yüz görüntüsünden (tek frame, kullanıcı etkileşimi yok) **live vs. spoof** ikili sınıflandırması. İki fazlı mimari:

- **Faz A (aktif):** DINOv2 with Registers (`dinov2_vitb14_reg`) fine-tuning — araştırma/accuracy modeli.
- **Faz B (ertelenmiş):** Faz A çıktısını teacher olarak kullanan Head-aware Knowledge Distillation ile 5MB'lık edge/mobil student model.

Detaylı plan için bkz. `dinov2_liveness_plan.md`.

## Eğitim Ortamı

Model eğitimi **Google Colab (A100)** üzerinde yapılır. Bu repo Colab'a `git clone` ile çekilip
`requirements.txt` kurulduktan sonra script'ler CLI üzerinden çalıştırılır; veri ve checkpoint'ler
Colab session storage / Google Drive'da tutulur, repo'ya dahil edilmez (`.gitignore`).

```bash
# Colab hücresi
!git clone https://github.com/CihangirEmre/passive-liveness-detection.git
%cd passive-liveness-detection
!pip install -r requirements.txt
!python scripts/00_check_dinov2_setup.py
```

**Önemli — Drive mount:** `drive.mount()` sadece doğrudan bir notebook hücresinde çalışır;
`!python scripts/...` ile çalıştırılan script'ler ayrı bir alt-process'te koştuğu için Colab
kernel'inin mesajlaşma kanalına erişemez. Bu yüzden Drive gerektiren script'leri (`02_extract_faces.py`
ve sonrası) çalıştırmadan önce **ayrı bir hücrede** şunu çalıştırın:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Veri Kalıcılığı (Drive)

Ham indirilen veri (~10GB+) Colab session storage'a (`/content`) yazılır — büyük ve
`01_download_celeba_spoof.py` ile yeniden üretilebilir olduğu için Drive kotasını
doldurmaz. **İşlenmiş (yüz-kırpılmış, split'lenmiş) veri seti** ise `src/colab_utils.py`
üzerinden otomatik olarak Google Drive'a yazılır (`/content/drive/MyDrive/passive-liveness-dinov2/`),
böylece Colab oturumu kapansa/kopsa bile veri hazırlama adımları tekrarlanmaz:

```
/content/drive/MyDrive/passive-liveness-dinov2/
├── processed/         # 02'nin çıktısı: yüz-kırpılmış görüntüler
├── processed_dedup/   # 02b'nin çıktısı: near-duplicate elenmiş alt küme
└── splits/            # 03'ün çıktısı: train.csv / val.csv / test.csv
```

## Proje Yapısı

```
.
├── configs/                        # Eğitim konfigürasyonları (Faz A.2'de doldurulur)
├── data/                           # Veri seti (git'e dahil değil, scriptlerle oluşturulur)
├── notebooks/                      # Colab notebook'ları
├── scripts/
│   ├── 00_check_dinov2_setup.py    # Faz A.0: model yükleme + dummy forward pass doğrulama
│   ├── 01_download_celeba_spoof.py # Faz A.1: Kaggle'dan indirme + klasör yapısı doğrulama
│   ├── 02_extract_faces.py         # Faz A.1: yüz crop (%20 margin, 224x224) + Drive'a yazma
│   ├── 02b_dedupe_phash.py         # Faz A.1: pHash ile near-duplicate eleme, Drive'a yazma
│   ├── 03_build_splits.py          # Faz A.1: subject-disjoint train/val/test split (70/15/15)
│   └── 04_stats_report.py          # Faz A.1: istatistik raporu + örnek batch görselleştirme
├── src/
│   ├── model_dinov2.py             # DINOv2Backbone: CLS + patch token çıktısı, freeze/unfreeze kontrolü
│   ├── face_crop.py                # Yüz tespiti (RetinaFace/MTCNN) + marginli crop (CelebA-Spoof ve LCC-FASD ortak)
│   ├── scan_utils.py               # <split>/<subject_id>/<live|spoof>/<image> dizin tarama (02 ve 03 ortak)
│   └── colab_utils.py              # Google Drive mount + kalıcı path yardımcıları
├── docs/                           # Değerlendirme raporları (data_stats_report.md, sample_batch.png, ...)
├── requirements.txt
├── .gitignore
└── dinov2_liveness_plan.md
```

## Faz A.0 — Model ve Ortam Kurulumu

```bash
python scripts/00_check_dinov2_setup.py
python scripts/00_check_dinov2_setup.py --freeze-backbone false --resolution 518
```

**Kabul kriteri:** Model yükleniyor, dummy input `(1,3,224,224)` ile forward pass hatasız çalışıyor,
CLS token `(1, 768)` ve patch tokens `(1, num_patches, 768)` boyutları doğrulanmış.

## Faz A.1 — Veri Pipeline (CelebA-Spoof, Colab üzerinde)

Veri seti [Kaggle mirror](https://www.kaggle.com/datasets/mabdullahsajid/celeba-spoofing) üzerinden
indirilir. Bu bir topluluk mirror'ı olduğu için resmi
[CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) klasör yapısıyla
(`Data/<split>/<subject_id>/<live|spoof>/*.jpg`) birebir aynı olduğu garanti değil —
**01'in `verify_extracted_structure` çıktısını indirme sonrası mutlaka kontrol edin.**

**Disk notu:** `01_download_celeba_spoof.py` zip'in TAMAMINI açmaz — bu, CelebA-Spoof'un
tam boyutuyla (625K+ görüntü) Colab'ın yerel diskini (`/content`) doldurabildiği için
kasıtlı bir tasarım. Bunun yerine `--max_per_group` (varsayılan **20**) ile her
`(subject_id, label)` grubundan seed'li rastgele seçimle en fazla N görüntü ZIP'in
içinden doğrudan seçilip çıkarılır — geri kalanına hiç dokunulmaz. Bu hem disk sorununu
çözer hem de fine-tuning veri hacmini indirme aşamasında azaltır (~10.000 subject × 2
label × 20 ≈ üst sınır, gerçek toplam gruplardaki gerçek görüntü sayısına göre daha az
çıkar). Tam veri seti isteniyorsa `--max_per_group 0` verilmeli. `metas/` klasörü
(label dosyaları) boyut fark etmeksizin her zaman eksiksiz çıkarılır.

```bash
# Colab hücreleri (sırayla)
!python scripts/01_download_celeba_spoof.py --kaggle_json /content/kaggle.json
!python scripts/02_extract_faces.py --input_dir /content/celeba_spoof_raw
!python scripts/02b_dedupe_phash.py
!python scripts/03_build_splits.py --processed_dir /content/drive/MyDrive/passive-liveness-dinov2/processed_dedup
!python scripts/04_stats_report.py
```

- `02_extract_faces.py` **resume destekler**: yarıda kesilen bir Colab oturumundan sonra
  aynı komutla tekrar çalıştırıldığında, zaten işlenmiş görüntüleri atlar. Hızlı
  smoke test için `--limit 50` kullanılabilir.
- `02b_dedupe_phash.py`, fine-tuning veri hacmini azaltmak için eklendi: aynı
  `(subject_id, label)` grubu içinde perceptual hash (pHash) ile neredeyse
  birebir aynı görüntüleri (aynı çekim oturumunun art arda kareleri gibi) eler.
  Sadece "1. aşama" (ucuz, GPU'suz) dedup — orijinal `processed/` klasörünü
  değiştirmez, tutulan alt kümeyi ayrı bir Drive klasörüne (`processed_dedup/`)
  kopyalar, böylece sonucu beğenmezsen orijinal veri seti hep geri dönülebilir
  durumda kalır. `--threshold` (varsayılan 5) ile agresiflik ayarlanır — daha
  düşük değer daha az eler.
- `03_build_splits.py`, subject kümesi üzerinde 70/15/15 böler ve her subject'in
  tüm görüntülerini aynı split'e atar; assert'lerle train/val/test arasında
  subject overlap olmadığını doğrular.
- `metas/` klasörü (spoof_type/illumination/environment zenginleştirmesi için)
  bu mirror'da bulunmazsa `spoof_type` kolonu `"unknown"` ile doldurulur —
  binary live/spoof label (klasör adından alınan) bundan etkilenmez.

**Kabul kriteri:** `04_stats_report.py` çıktısı (`docs/data_stats_report.md`) toplam görüntü,
live/spoof oranı, spoof tipi dağılımı ve subject sayısını raporluyor; subject overlap
kontrolü PASS; `docs/sample_batch.png` örnek batch görselleştirmesi kaydedilmiş.
