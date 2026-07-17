# Passive Liveness Detection — DINOv2 → Edge Distillation

Yüz görüntüsünden (tek frame, kullanıcı etkileşimi yok) **live vs. spoof** ikili sınıflandırması. İki fazlı mimari:

- **Faz A (tamamlandı):** DINOv2 with Registers (`dinov2_vitb14_reg`) fine-tuning — araştırma/accuracy modeli.
- **Faz B (ertelenmiş):** Faz A çıktısını teacher olarak kullanan Head-aware Knowledge Distillation ile 5MB'lık edge/mobil student model.

Detaylı plan için bkz. `dinov2_liveness_plan.md`.

## Durum Özeti

| Faz | Adım | Durum |
|---|---|---|
| A.0 | Model + ortam kurulumu | ✅ |
| A.1 | Veri pipeline (indirme, yüz crop, dedup, split, rapor) — ~103K görüntü | ✅ |
| A.2.1 | Linear probing (backbone frozen) | ✅ |
| A.2.2 | Kademeli unfreeze fine-tuning (son 2 blok) | ✅ |
| A.3 | Internal değerlendirme (CelebA-Spoof test split) | ✅ |
| A.3 | External değerlendirme (LCC-FASD, zero-shot) | ✅ |
| — | Canlı webcam demo | ✅ |
| B | Knowledge distillation (edge model) | ⏳ ertelendi |

**Nihai sonuçlar** (checkpoint: `finetune_a2_2_u2.pt`, ~103K veri, sabit eşik=0.4724 — internal val split EER'inden türetildi, bkz. `docs/generalization_report.md`):

| Metrik | Internal (CelebA-Spoof test) | External (LCC-FASD, zero-shot) |
|---|---|---|
| ACER | %0.44 | %12.85 |
| EER | %0.45 | %11.81 |
| AUC | 0.9999 | 0.9514 |

Internal/external arasındaki büyük fark, modelin CelebA-Spoof'un kendi çekim/recapture
pipeline'ına özgü izlere kısmen fazla uyum sağladığını (literatürdeki tipik
intra-dataset/cross-dataset PAD farkı) gösteriyor — detaylı yorum için
`docs/generalization_report.md`.

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
├── raw_zip/            # 01'in indirdiği CelebA-Spoof zip'i (~78GB, yerel diske sığmaz)
├── processed/          # 02'nin çıktısı: yüz-kırpılmış görüntüler
├── processed_dedup/    # 02b'nin çıktısı: near-duplicate elenmiş alt küme
├── splits/             # 03'ün çıktısı: train.csv / val.csv / test.csv
└── checkpoints/        # train.py'nin ürettiği .pt checkpoint'ler + history CSV
```

## Proje Yapısı

```
.
├── configs/                        # Eğitim konfigürasyonları
├── data/                           # Veri seti (git'e dahil değil, scriptlerle oluşturulur)
├── model/                          # Yerel checkpoint kopyası (git'e dahil değil)
├── notebooks/                      # Colab notebook'ları
├── scripts/
│   ├── 00_check_dinov2_setup.py    # Faz A.0: model yükleme + dummy forward pass doğrulama
│   ├── 01_download_celeba_spoof.py # Faz A.1: Kaggle'dan indirme + klasör yapısı doğrulama
│   ├── 02_extract_faces.py         # Faz A.1: yüz crop (%20 margin, 224x224) + Drive'a yazma
│   ├── 02b_dedupe_phash.py         # Faz A.1: pHash ile near-duplicate eleme, Drive'a yazma
│   ├── 03_build_splits.py          # Faz A.1: subject-disjoint train/val/test split (70/15/15)
│   ├── 04_stats_report.py          # Faz A.1: istatistik raporu + örnek batch görselleştirme
│   ├── 05_evaluate_internal.py     # Faz A.3: CelebA-Spoof test split'inde ACER/EER/AUC + spoof tipi kırılımı
│   ├── 06_evaluate_external.py     # Faz A.3: LCC-FASD üzerinde zero-shot değerlendirme
│   └── 07_webcam_demo.py           # Canlı webcam ile görsel test (Haar Cascade detection + DINOv2 classification)
├── src/
│   ├── model_dinov2.py             # DINOv2Backbone: CLS + patch token çıktısı, freeze/kademeli unfreeze kontrolü
│   ├── train.py                    # Faz A.2: DinoLivenessModel + A.2.1/A.2.2 eğitim döngüsü, best-checkpoint takibi
│   ├── dataset.py                  # CelebASpoofSplitDataset + ManifestPairDataset (LCC-FASD icin)
│   ├── metrics.py                  # EER / AUC / APCER / BPCER / ACER hesaplama
│   ├── eval_utils.py               # Checkpoint'ten model kurma + inference yardımcıları
│   ├── face_crop.py                # Yüz tespiti (RetinaFace/MTCNN) + marginli crop (CelebA-Spoof ve LCC-FASD ortak)
│   ├── scan_utils.py               # <split>/<subject_id>/<live|spoof>/<image> dizin tarama (02 ve 03 ortak)
│   └── colab_utils.py              # Google Drive mount + kalıcı path yardımcıları
├── docs/                           # Değerlendirme raporları (data_stats_report.md, internal_eval_report.md,
│                                    # external_eval_report.md, generalization_report.md, sample_batch.png, ...)
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
indirilir. Klasör yapısı (`CelebA_Spoof/Data/<split>/<subject_id>/<live|spoof>/*.jpg`,
`CelebA_Spoof/metas/intra_test/train_label.txt`) Kaggle Data Explorer'dan doğrulandı —
resmi [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) yapısıyla birebir aynı.

**Disk notu (önemli):** Veri seti **~78GB** — Colab'ın tipik yerel diskine (`/content`,
~60-110GB, genelde zaten kısmen dolu) sığmıyor; Drive'a indirmek de bazı ortamlarda
yine yerel diski dolduruyor (kaggle CLI/Drive FUSE geçici olarak yerelde tamponluyor
gibi görünüyor). Bu yüzden **varsayılan indirme modu `files`**: zip'i HİÇ indirmeden,
Kaggle'ın dosya listesi API'sinden (`kaggle datasets files`) alınan tam dosya
listesinden sadece seçilen görüntüleri (+ eşleşen `_BB.txt` — CelebA-Spoof'un önceden
hesapladığı yüz kutusu — ve `metas/` label dosyaları) **tek tek, paralel** indirir.
Disk kullanımı sadece seçilen alt küme kadardır.

```bash
# Colab hücreleri (sırayla)
!python scripts/01_download_celeba_spoof.py --kaggle_json /content/kaggle.json --max_per_group 6 --workers 8
!python scripts/02_extract_faces.py --input_dir /content/celeba_spoof_raw
!python scripts/02b_dedupe_phash.py
!python scripts/03_build_splits.py --processed_dir /content/drive/MyDrive/passive-liveness-dinov2/processed_dedup
!python scripts/04_stats_report.py
```

- `--max_per_group`: her `(subject_id, label)` grubundan seed'li (seed=42) rastgele
  seçimle en fazla N görüntü — TÜM ~9193 subject'ten örneklenir, yani N arttıkça
  subject/kimlik çeşitliliği bozulmadan görüntü sayısı artar. Proje şu an
  `max_per_group=6` (~103K görüntü) ile çalışıyor; ilk doğrulama `max_per_group=5`
  (~30K, sadece A.2.1 linear probing için) ile yapılmıştı. **Not:** aynı seed ile
  farklı bir `max_per_group` değeriyle tekrar çalıştırmak farklı bir rastgele seçim
  üretir (üst küme değil) — hacim değiştirilecekse pipeline'ın tamamı (01→02→02b→03→04)
  yeniden çalıştırılmalı.
- `--workers` (varsayılan 8): paralel indirme thread sayısı.
- `--skip_bb`: `_BB.txt` (önceden hesaplanmış yüz kutusu) dosyalarını indirmeyi kapatır
  (varsayılan: indirilir, küçük ve `02`'de kendi yüz tespitimizi atlamamıza yardımcı
  olabilir).
- Dosya listesi (`kaggle datasets files`) yerelde CSV olarak önbelleklenir — script
  kesilip tekrar çalıştırılırsa listeyi tekrar çekmez.
- Daha eski/hızlı `--mode zip` yolu da mevcut (tüm zip'i indirip seçici açar) —
  yeterli Drive/disk alanı olan ortamlarda (~80GB+) kullanılabilir, ama disk
  sorunlarına karşı garantili değil.

```bash
# Alternatif: zip modu (yeterli disk/Drive alanı varsa daha hizli)
!python scripts/01_download_celeba_spoof.py --kaggle_json /content/kaggle.json --mode zip --max_per_group 20
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
  subject overlap olmadığını doğrular. Subject→split ataması sadece subject
  listesine ve sabit seed'e bağlı olduğu için, veri hacmi (`max_per_group`)
  değiştirilse bile split oranları ve subject dağılımı stabil kalır.
- `metas/` klasörü (spoof_type/illumination/environment zenginleştirmesi için)
  bu mirror'da bulunmazsa `spoof_type` kolonu `"unknown"` ile doldurulur —
  binary live/spoof label (klasör adından alınan) bundan etkilenmez.

**Kabul kriteri:** `04_stats_report.py` çıktısı (`docs/data_stats_report.md`) toplam görüntü,
live/spoof oranı, spoof tipi dağılımı ve subject sayısını raporluyor; subject overlap
kontrolü PASS; `docs/sample_batch.png` örnek batch görselleştirmesi kaydedilmiş.

## Faz A.2 — Fine-tuning (`src/train.py`)

Tek script, `--unfreeze_blocks` parametresiyle iki aşamayı da kapsar:

- **A.2.1 — Linear probing** (`--unfreeze_blocks 0`, varsayılan): backbone tamamen
  frozen, sadece `Linear(768→2)` head eğitilir (~1538 eğitilebilir parametre).
  Overfitting riski pratikte yok, hızlı bir baseline verir.
- **A.2.2 — Kademeli unfreeze fine-tuning** (`--unfreeze_blocks N`, N>0): backbone'un
  son N transformer bloğu da eğitilir, discriminative learning rate ile (head için
  `--lr`, varsayılan 1e-3; backbone için `--backbone_lr`, varsayılan 1e-5).
  `CosineAnnealingLR` scheduler'ı ve `val_acer`'a göre erken durdurma (`--patience`)
  sadece `unfreeze_blocks>0` iken devrede.

```bash
# Colab — A.2.2, son 2 blok, discriminative LR, 5 epoch sabır ile early stopping
!python src/train.py \
    --train_csv /content/drive/MyDrive/passive-liveness-dinov2/splits/train.csv \
    --val_csv /content/drive/MyDrive/passive-liveness-dinov2/splits/val.csv \
    --images_root /content/drive/MyDrive/passive-liveness-dinov2/processed_dedup \
    --output_dir /content/drive/MyDrive/passive-liveness-dinov2/checkpoints \
    --unfreeze_blocks 2 --epochs 20 --patience 5
```

- Her epoch sonunda `val_acer` iyileşirse ağırlıklar (head + eğitilebilir backbone
  blokları) `.detach().clone().cpu()` ile klonlanıp bellekte tutulur (patience
  değerinden bağımsız, her zaman çalışır). Eğitim bitince **iki checkpoint** yazılır:
  `<name>.pt` (son epoch) ve `<name>_best.pt` (en iyi `val_acer` epoch'u, `best_epoch`/
  `best_val_acer` metadata'sıyla).
  Değerlendirme script'lerinde (`05`/`06`) `_best.pt` kullanılması önerilir.
- Ayrıca `<name>_history.csv` yazılır (epoch başına train/val loss, acc, APCER,
  BPCER, ACER) — overfitting analizi için kullanılabilir.

**Gözlemlenen sonuç (100K veri, A.2.2, son 2 blok):** en iyi `val_acer` %0.305
(epoch 18), A.2.1 linear-probe baseline'ına göre belirgin iyileşme; 30K→100K veri
artışı, ezberlemenin başladığı epoch'u da geciktirdi (epoch7→epoch12).

### Öneri (henüz uygulanmadı)

`plan.md`'de belgelendiği gibi, gözlemlenen overfitting eğilimini azaltmak için
literatürden iki teknik önerildi ama kasıtlı olarak henüz koda eklenmedi:
`label_smoothing=0.1` (CrossEntropyLoss) ve `weight_decay=0.01-0.05` (AdamW).

## Faz A.3 — Değerlendirme (`05_evaluate_internal.py`, `06_evaluate_external.py`)

**Metodoloji:** Karar eşiği (threshold) **sadece val split**'in EER noktasından
türetilir, sonra hem internal test split'e hem external veri setine **sabit** olarak
uygulanır — asla test/external verisine göre yeniden kalibre edilmez (leakage/optimistik
sonuç riskini önlemek ve gerçek zero-shot genelleme performansını ölçmek için).

**A.3a — Internal (CelebA-Spoof test split):**
```bash
!python scripts/05_evaluate_internal.py \
    --checkpoint /content/drive/MyDrive/passive-liveness-dinov2/checkpoints_v2/finetune_a2_2_u2_best.pt \
    --val_csv /content/drive/MyDrive/passive-liveness-dinov2/splits_v2/val.csv \
    --test_csv /content/drive/MyDrive/passive-liveness-dinov2/splits_v2/test.csv \
    --images_root /content/celeba_processed_dedup_v2
```
ACER/APCER/BPCER (val EER eşiğinde), eşik-bağımsız test EER/AUC ve spoof-tipi bazlı
APCER kırılımını konsola basar ve `docs/internal_eval_report.md`'ye yazar.

**A.3b — External (LCC-FASD, zero-shot):** Genuine/imposter yolları `CLIENT_*.txt` /
`IMPOSTER_*.txt` manifest dosyalarında ayrı ayrı listelenen [LCC-FASD](https://www.kaggle.com/datasets/faber24/lcc-fasd)
veri setiyle test edilir (`src/dataset.py::ManifestPairDataset`). **Dikkat:** LCC-FASD'in
Kaggle referans kodu `client(genuine)=1, imposter(spoof)=0` kullanır — bu projenin
`0=live, 1=spoof` sözleşmesinin tam tersi; `ManifestPairDataset` bunu bilerek düzeltir.

```bash
!python scripts/06_evaluate_external.py \
    --checkpoint /content/drive/MyDrive/passive-liveness-dinov2/checkpoints_v2/finetune_a2_2_u2_best.pt \
    --live_manifest /content/LCC_FASD/LCC_FASD/CLIENT_TEST.txt \
    --spoof_manifest /content/LCC_FASD/LCC_FASD/IMPOSTER_TEST.txt \
    --images_root /content/LCC_FASD/LCC_FASD \
    --threshold 0.4724 \
    --internal_acer 0.0044 --internal_eer 0.0045 --internal_auc 0.9999
```
(`--internal_*` değerleri `05`'in konsol çıktısından kopyalanır; verilirse
internal/external karşılaştırma tablosu da rapora eklenir.) Eşik-bağımsız external
EER/AUC ve sabit eşikte APCER/BPCER/ACER konsola basılır, `docs/external_eval_report.md`'ye
yazılır.

**Sonuç ve yorum:** `docs/generalization_report.md` — internal/external metrik
tablosu, internal spoof-tipi kırılımı, external için kırılımın neden mevcut
olamadığının açıklaması (LCC-FASD'de saldırı-tipi metadata'sı yok) ve "çekim imzası"
(capture/domain signature) riskinin bu sonuçla nasıl doğrulandığına dair yorum içerir.

## Canlı Webcam Demo (`07_webcam_demo.py`)

Eğitilmiş bir checkpoint'i yerel webcam ile görsel olarak test etmek için:

```bash
python scripts/07_webcam_demo.py --checkpoint model/finetune_a2_2_u2.pt --threshold 0.4724
```

- Yüz **tespiti** OpenCV'nin klasik `haarcascade_frontalface_default.xml` (Haar Cascade)
  algoritmasıyla yapılır — DINOv2 modeliyle ilgisi yok, sadece "yüz nerede" sorusuna
  cevap verir. Her karede sıfırdan çalışır (gerçek bir track ID / hafıza yok).
- Bulunan yüz kırpılıp (`--margin`, varsayılan 0.2) 224x224'e resize edildikten
  **sonra** DINOv2 tabanlı sınıflandırma modeline verilir.
  Ekrandaki yüzde her zaman **spoof olasılığını** gösterir (`softmax(logits)[1]`);
  LIVE etiketinde düşük skor beklenen/doğru davranıştır.
- `--infer_every N`: CPU'da akıcılık için her N karede bir inference çalıştırır.
- **Bilinen sınırlama:** Demo şu an **frame-bazlı** karar veriyor — tek bir gürültülü
  kare, kısa süreliğine yanlış etiket gösterebilir. Production sistemlerde bu,
  temporal smoothing (kayan pencere ortalaması/EMA), çoğunluk oylaması + histerezis
  veya sekans-sonu tek karar gibi tekniklerle çözülür; bu demo'da henüz uygulanmadı.

## Sonraki Adımlar

- Faz B'ye (Head-aware Knowledge Distillation) geçmeden önce, internal/external
  genelleme farkının (ACER %0.44 → %12.85) azaltılıp azaltılmayacağının
  değerlendirilmesi öneriliyor (bkz. `docs/generalization_report.md`'nin kapanışı) —
  örn. daha çeşitli/çok-kaynaklı eğitim verisi veya domain generalization teknikleri.
- `label_smoothing` / `weight_decay` (bkz. yukarıdaki Öneri) uygulanıp A.2.2'nin
  tekrar çalıştırılması değerlendirilebilir.
