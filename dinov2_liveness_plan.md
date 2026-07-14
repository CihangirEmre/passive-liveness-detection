# Passive Liveness Detection — DINOv2 → Edge Distillation Planı

> **Bu dosyanın amacı:** Bu plan bir LLM asistanına (Claude, Claude Code vb.) context olarak verilmek üzere yazılmıştır. İki ana faz vardır: **Faz A** (araştırma/accuracy modeli — DINOv2 fine-tuning) ve **Faz B** (deployment modeli — Head-aware Knowledge Distillation ile 5MB student). Faz B, Faz A tamamlandıktan sonra, Faz A'nın çıktısı olan modeli TEACHER olarak kullanarak başlar. Faz B şimdilik ERTELENMİŞTİR (future work) — sadece plan/iskelet olarak bu dosyada durur, Faz A bitmeden başlanmaz.

---

## PROJE ÖZETİ

```yaml
proje_adi: passive-liveness-dinov2-to-edge
gorev_tipi: Binary image classification (live vs. spoof)
yaklasim: Passive (kullanıcı etkileşimi yok, tek frame)
dataset_primary: CelebA-Spoof
dataset_external: LCC-FASD (birincil)
framework: PyTorch + Hugging Face transformers / torch.hub
dil: Python 3.12
deney_ortami: Google Colab (A100)
degerlendirme_metrikleri: [ACER, APCER, BPCER, EER, AUC]

iki_fazli_mimari:
  faz_A:
    rol: "Araştırma modeli — en yüksek doğruluk/generalization"
    model: DINOv2 with Registers (dinov2_vitb14_reg)
    durum: AKTİF — şimdi başlanacak
  faz_B:
    rol: "Deployment modeli — cihaza giden asıl model"
    model: Head-aware Knowledge Distillation, 5MB student
    teacher: Faz A çıktısı (fine-tuned DINOv2)
    durum: ERTELENMİŞ — Faz A tamamlanmadan başlanmaz
```

### Neden bu mimari (LLM için context)
- DINOv2, self-supervised ön-eğitimi sayesinde cross-dataset generalization'da CNN ve supervised ViT'lerden daha güçlü sonuç veriyor; Registers varyantı attention artifact problemini çözüp ince taneli spoofing ipuçlarını daha temiz yakalıyor.
- Ancak DINOv2 (86M+ parametre) doğrudan cihaza (mobil/edge) deploy edilemeyecek kadar büyük ve yavaş; self-attention'ın O(n²) maliyeti edge NPU'larda CNN kadar optimize değil.
- Çözüm: DINOv2'yi hiç cihaza göndermeden, sadece "öğretmen" olarak kullanıp bilgisini 5MB'lık bir "öğrenci" modele damıtmak (distillation). Cihaza giden model HER ZAMAN Faz B çıktısıdır, Faz A çıktısı değildir.

---

## FAZ A — DINOv2 Fine-Tuning (AKTİF FAZ, şimdi başla)

### A.0 — Model ve Ortam Kurulumu

```yaml
model_id: dinov2_vitb14_reg   # "reg" = with Registers, ZORUNLU (attention artifact fix)
kaynak_secenek_1: torch.hub("facebookresearch/dinov2", "dinov2_vitb14_reg")
kaynak_secenek_2: transformers ("facebook/dinov2-with-registers-base")
patch_size: 14
input_resolution: 224   # 14'ün katı olmalı; 518 alternatif (daha yüksek texture detayı, daha yüksek maliyet)
parametre_sayisi: ~86M (ViT-B/14)
```

**Kabul Kriteri:** Model yükleniyor, dummy input (1,3,224,224) ile forward pass hatasız çalışıyor, çıktı embedding boyutu doğrulanmış.

**LLM'e Görev Tanımı:**
```
"torch.hub üzerinden dinov2_vitb14_reg modelini yükleyen, dummy input ile forward
pass testi yapan ve çıktı embedding boyutunu (CLS token + patch tokens) yazdıran
bir setup script'i yaz. Backbone'un frozen/unfrozen durumunu kontrol eden bir flag ekle."
```

---

### A.1 — Veri Pipeline (CelebA-Spoof)

```
scripts/
  01_download_celeba_spoof.py   # dataset indirme + checksum kontrolü
  02_extract_faces.py           # yüz crop (RetinaFace/MTCNN), margin ~%20, output 224x224
  03_build_splits.py            # SUBJECT-DISJOINT train/val/test split
  04_stats_report.py            # sınıf dağılımı, spoof tipi dağılımı, subject sayısı
```

### Kritik Kurallar
1. **Subject-disjoint split zorunlu** — aynı kişi hem train hem test'te olamaz (assert testi ile doğrula).
2. Crop boyutu **224×224** (patch_size=14'ün katı, 16 patch/kenar).
3. Class imbalance varsa raporla; gerekirse weighted sampler (CelebA-Spoof spoof-ağırlıklı olabilir).
4. **Augmentasyon kısıtı:** Agresif ColorJitter/grayscale KULLANMA — liveness sinyali texture/renkte taşınır. Hafif RandomHorizontalFlip + hafif ColorJitter + JPEG compression augmentation (quality 60-95) yeterli.

### Kabul Kriteri
- [ ] `04_stats_report.py` çıktısı: toplam görüntü, live/spoof oranı, spoof tipi dağılımı, subject sayısı
- [ ] Subject overlap assert testi geçiyor
- [ ] Örnek batch görselleştirmesi kaydedildi

### LLM'e Görev Tanımı
```
"CelebA-Spoof metadata dosyasından (image_path, subject_id, label, spoof_type)
subject-disjoint train/val/test split üreten script yaz (oran 70/15/15).
Assert ile subject_id overlap kontrolü ekle. RetinaFace ile yüz crop + 224x224
resize + %20 margin uygulayan ayrı bir preprocessing fonksiyonu yaz."
```

---

### A.2 — Training Stratejisi (Kademeli)

DINOv2 iki alt-adımda eğitilir; ikinci adıma birinci adım tamamlanmadan geçilmez.

#### A.2.1 — Linear Probing (hızlı baseline)
```yaml
backbone: FROZEN (tüm parametreler dondurulmuş)
head: Linear(768 -> 2)   # ViT-B/14 CLS token boyutu 768
loss: CrossEntropyLoss
optimizer: AdamW, lr=1e-3 (sadece head)
epochs: 10-15
beklenen_sure_A100: birkaç saat (tüm CelebA-Spoof üzerinde)
amac: "Hızlı sağlık kontrolü — feature'lar liveness ayrımı için kullanışlı mı?"
```

**Kabul Kriteri:** Val ACER değeri makul bir referans noktası oluşturuyor (kesin eşik yok, sadece "sinyal var mı" kontrolü).

#### A.2.2 — Kademeli Unfreeze (asıl performans artışı)
```yaml
backbone: SON 2-4 transformer bloğu unfreeze edilir, geri kalan frozen
head: Linear(768 -> 2)  # veya MLP head
optimizer: AdamW
  backbone_lr: 1e-5   # düşük — pretrained bilgiyi bozma
  head_lr: 1e-3        # yüksek — head sıfırdan öğreniyor
scheduler: CosineAnnealingLR
epochs: 15-20, early stopping (val ACER, patience=5)
input_variant_test: 224 vs 518 çözünürlük karşılaştırması (opsiyonel, compute izin verirse)
```

**Neden discriminative LR:** Backbone zaten güçlü genel özellikler öğrenmiş; büyük LR ile bunu bozmamak gerekir. Head ise sıfırdan spoofing-özel ayrımı öğrenmeli.

### Kabul Kriteri (A.2 geneli)
- [ ] Val ACER, linear probing sonucundan belirgin şekilde daha iyi
- [ ] Spoof tipi bazlı APCER breakdown üretildi (print/replay/mask ayrı ayrı)
- [ ] Overfitting kontrolü: train/val ACER farkı makul aralıkta (büyük fark varsa unfreeze edilen blok sayısı azaltılır)

### LLM'e Görev Tanımı
```
"dinov2_vitb14_reg üzerine iki aşamalı training script'i yaz: (1) backbone frozen,
sadece linear head eğitimi, (2) son N transformer bloğunu unfreeze edip
discriminative learning rate (backbone 1e-5, head 1e-3) ile fine-tuning.
AdamW + CosineAnnealingLR + early stopping (val ACER bazlı, patience=5).
Her iki aşama için ayrı checkpoint kaydet."
```

---

### A.3 — Değerlendirme (Internal + External)

```
scripts/
  05_evaluate_internal.py   # CelebA-Spoof test split üzerinde
  06_evaluate_external.py   # LCC-FASD ve üzerinde ZERO-SHOT (fine-tuning YOK)
```

### Protokol
1. Faz A.2'de eğitilen en iyi checkpoint **dondurulur**.
2. LCC-FASD preprocessing'den geçirilir.
3. Rapor: internal ACER vs. external ACER + gap analizi.


### Kabul Kriteri
- [ ] `docs/generalization_report.md`: internal/external metrik tablosu
- [ ] Spoof tipi bazlı external performans breakdown (LCC-FASD'de hangi saldırı tipinde düşüş var?)


### Kabul Kriterleri tamamlandıktan sonra kullanıcı ister ise yapılıcak ek iş ve Kabul kriteri
4. Grad-CAM veya attention map görselleştirmesi ile modelin nereye "baktığı" incelenir (DINOv2+Registers'ın attention'ı temiz olduğu için bu analiz daha güvenilir olmalı).
- [ ] Attention/Grad-CAM görselleştirmesi ile hata analizi

### LLM'e Görev Tanımı
```
"Eğitilmiş DINOv2 checkpoint'ini LCC-FASD üzerinde zero-shot
değerlendiren bir script yaz. APCER/BPCER/ACER/EER hesapla (threshold val set
EER noktasından). Internal vs external sonuçları karşılaştıran bir tablo üret."
```

---

## FAZ A ÇIKTISI (Faz B'nin girdisi)

```yaml
faz_a_ciktisi:
  checkpoint: "en iyi val ACER'a sahip fine-tuned DINOv2 modeli"
  format: ".pt / .safetensors"
  icerik: "backbone (kısmi unfreeze edilmiş) + head ağırlıkları"
  rol_faz_b_icin: "TEACHER modeli"
```

**Faz A tamamlanma koşulu:** Internal ACER kabul edilebilir seviyede VE external (LCC-FASD) generalization gap raporlanmış olmalı. Bu iki koşul sağlanmadan Faz B'ye geçilmez.

---

## FAZ B — Head-aware Knowledge Distillation (ERTELENMİŞ — Faz A sonrası başlar)

> **DURUM: Bu faz şimdi UYGULANMAYACAK.** Aşağıdaki içerik, Faz A tamamlandığında kullanılacak bir iskelet/yol haritasıdır. Amaç: cihaza (mobil/edge) gidecek asıl modeli üretmek.

### B.0 — Kavramsal Çerçeve

```yaml
teacher: "Faz A çıktısı (fine-tuned DINOv2 with Registers)"
student: "5MB parametre bütçeli lightweight model (örn. MobileNetV3 tabanlı backbone)"
yontem: Head-aware Knowledge Distillation
distillation_bilesenleri:
  - feature_level_distillation: "teacher ve student ara katman feature'ları arasında loss"
  - logits_level_distillation: "teacher ve student çıktı logitleri arasında KL-divergence / soft label loss"
  - head_aware_strategy: "teacher/student arasındaki boyut uyumsuzluğunu çözmek için attention head correlation matrix"
neden_gerekli: >
  Teacher (DINOv2) ile student (MobileNetV3-benzeri) arasında ara katman
  boyutları (hidden dim, head sayısı) uyuşmuyor. Head-aware strateji, bu
  boyut uyumsuzluğunu bir korelasyon matrisi ile köprüleyerek feature-level
  distillation'ı mümkün kılar.
```

### B.1 — Student Mimari Seçimi (öneri, kesinleşmemiş)

| Aday | Parametre | Not |
|---|---|---|
| MobileNetV3-Small | ~2.5M | En hafif, ilk deneme için uygun |
| MobileNetV3-Large | ~5.4M | Hedef bütçeye (5MB) en yakın |
| EfficientFormer-L1 | ~12M | Transformer-benzeri ama mobil-optimize; büyükse küçültülür |

**Karar kuralı:** MobileNetV3-Large ile başla, 5MB hedefini (quantization sonrası) tutturamıyorsa pruning veya MobileNetV3-Small'a düş.

### B.2 — Training Protokolü (taslak)

```yaml
asama_1: "Student'ı sadece hard label (gerçek etiket) ile eğit → baseline"
asama_2: "Teacher'ın soft label'ları (logits) ile distillation loss ekle"
asama_3: "Feature-level distillation ekle (head-aware correlation matrix ile)"
asama_4: "Quantization (INT8) veya pruning ile 5MB hedefine indir"
degerlendirme: "Her aşamada internal + external ACER karşılaştır — hangi bileşen ne kadar katkı sağlıyor ablation olarak raporla"
```

### B.3 — Kabul Kriterleri (Faz B başladığında geçerli olacak)
- [ ] Student model boyutu ≤ 5MB (quantization sonrası)
- [ ] Student ACER, teacher ACER'a "kabul edilebilir" yakınlıkta (performans kaybı raporlanır, gizlenmez)
- [ ] Inference hızı ölçülmüş (CPU/mobil simülasyonu, örn. ONNX Runtime ile)
- [ ] HF Space veya webcam demo üzerinden gerçek zamanlı çalıştığı doğrulanmış

### LLM'e Görev Tanımı (Faz B başladığında kullanılacak)
```
"Faz A'da eğitilmiş DINOv2 checkpoint'ini teacher olarak kullanan bir Head-aware
Knowledge Distillation pipeline'ı yaz. Student: MobileNetV3-Large. Distillation
loss: feature-level (head-aware correlation matrix ile boyut uyumu) + logits-level
(KL-divergence, temperature parametreli). Her aşamayı (hard-label-only, +logits,
+feature) ayrı ayrı eğitip ablation tablosu üret."
```

---

## RİSKLER VE BİLİNEN TUZAKLAR (her fazda kontrol edilecek)

1. **Subject leakage (Faz A):** Split değişikliğinden sonra assert testi mutlaka çalıştırılmalı.
2. **DINOv2 giriş çözünürlük kısıtı:** 224 veya 518 — 14'ün katı olmayan boyutlar hata verir.
3. **Overfitting (Faz A.2.2):** Çok fazla blok unfreeze edilirse tek dataset'e (CelebA-Spoof) aşırı uyum riski artar; external gap büyür.
4. **Teacher-student boyut uyumsuzluğu (Faz B):** Head-aware correlation matrix olmadan feature-level distillation doğrudan uygulanamaz — bu adımı atlamaya çalışma.
5. **Faz B'ye erken geçiş:** Faz A'nın generalization raporu tamamlanmadan Faz B'ye başlanırsa, kötü bir teacher'ın hatalarını student'a damıtmış olursun. Sıra kesinlikle korunmalı.
6. **Lisans/etik:** CelebA-Spoof, LCC-FASD — hiçbiri demo'da ham görüntü olarak gösterilmemeli, sadece kendi çektiğin test görüntüleri demo'da kullanılmalı.

---

## KLASÖR YAPISI (hedef)

```
passive-liveness-dinov2/
├── configs/
│   ├── faz_a_linear_probe.yaml
│   ├── faz_a_finetune.yaml
│   └── faz_b_distillation.yaml     # Faz B başladığında doldurulur
├── data/                # (gitignore) raw + processed
├── scripts/              # 01-06 pipeline scriptleri (Faz A)
├── src/
│   ├── dataset.py
│   ├── model_dinov2.py
│   ├── model_student.py            # Faz B başladığında doldurulur
│   ├── distillation.py             # Faz B başladığında doldurulur
│   ├── metrics.py                  # APCER/BPCER/ACER/EER
│   └── train.py
├── docs/
│   ├── notes.md
│   ├── ablation_table.md
│   └── generalization_report.md    # Faz A çıktısı, Faz B'nin başlangıç referansı
└── README.md
```

---

## ZAMAN ÇİZELGESİ ÖZETİ

| Faz | Alt-Adım | Süre | Durum |
|---|---|---|---|
| A.0 | Model/ortam kurulumu | 0.5 gün | Şimdi başla |
| A.1 | Veri pipeline (CelebA-Spoof) | 3-5 gün | Şimdi başla |
| A.2.1 | Linear probing | 1-2 gün | A.1 sonrası |
| A.2.2 | Kademeli unfreeze fine-tuning | 3-5 gün | A.2.1 sonrası |
| A.3 | Internal + external değerlendirme | 3-5 gün | A.2.2 sonrası |
| **B** | **Head-aware KD → 5MB student** | **1-2 hafta** | **ERTELENMİŞ — Faz A tamamlanınca** |

**Faz A toplam: ~2-3 hafta.** Faz B, Faz A'nın `generalization_report.md` çıktısı onaylandıktan sonra ayrı bir çalışma turu olarak başlatılır.
