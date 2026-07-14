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
!git clone <repo-url>
%cd passiveLivenessDetection
!pip install -r requirements.txt
!python scripts/00_check_dinov2_setup.py
```

## Proje Yapısı

```
.
├── configs/                        # Eğitim konfigürasyonları (Faz A.2'de doldurulur)
├── data/                           # Veri seti (git'e dahil değil, scriptlerle oluşturulur)
├── notebooks/                      # Colab notebook'ları
├── scripts/
│   └── 00_check_dinov2_setup.py    # Faz A.0: model yükleme + dummy forward pass doğrulama
├── src/
│   └── model_dinov2.py             # DINOv2Backbone: CLS + patch token çıktısı, freeze/unfreeze kontrolü
├── docs/                           # Değerlendirme raporları (Faz A.3'te doldurulur)
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
