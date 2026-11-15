# Breast MRI Classification with CNN and Vision Transformers

Bu depo, meme MRI görüntülerini **Benign** ve **Malignant** olarak ikili sınıflandırmada CNN ve vision transformer tabanlı modelleri karşılaştırmak için hazırlanmıştır.

Bütün deneyler **tek bir standart protokol** altında yürütülür. Dört deney grubunun her biri için ayrı bir notebook vardır ve bu dört notebook'un kodu birebir aynıdır; tek fark ilk hücredeki sabit `GROUP` satırıdır.

| Notebook | Grup (`GROUP`) | Modeller |
|---|---|---|
| `1_baseline_cnn.ipynb` | `baseline_cnn` | ResNet34, DenseNet121, EfficientNet-B0, MobileNetV2, InceptionV3 |
| `2_baseline_vit.ipynb` | `baseline_vit` | ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32, Swin-T |
| `3_optimized_cnn.ipynb` | `optimized_cnn` | ResNet34, DenseNet121, EfficientNet-B0, MobileNetV2, InceptionV3 |
| `4_optimized_vit.ipynb` | `optimized_vit` | ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32, Swin-T |
| `5_tum_sonuclar.ipynb` | — | Dört grubun sonuçlarını `ALL_RESULTS.xlsx` dosyasında birleştirir (GPU gerekmez) |

Ortak ayarlardan birini değiştirmek gerekirse (örneğin `COMMON` veya `RECIPES`) değişiklik dört notebook'a da aynı şekilde yapılmalıdır; aksi halde gruplar arasındaki standart bozulur.

Swin-T eğitilip kaydedilir. Makaleye dahil edilip edilmeyeceği ayrı bir karardır (`INCLUDE_SWIN`).

## Veri Seti

[Breast MRI Tumor Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/abenjelloun/breast-mri-tumor-classification-dataset). Veri seti TCIA "Breast Diagnosis" koleksiyonundan türetilmiştir.

| Kaggle klasörü | Benign | Malignant | Toplam |
|---|---|---|---|
| train | 8.413 | 8.413 | 16.826 |
| validation | 1.797 | 1.797 | 3.594 |
| test | 1.802 | 1.802 | 3.604 |
| **Toplam** | **12.012** | **12.012** | **24.024** |

Notebook veriyi `kagglehub` ile indirir ya da Drive'daki rar dosyasını açar. `train`, `validation` ve `test` klasörlerini kendisi bulur ve görüntü sayılarını yukarıdaki değerlerle karşılaştırır. Sayılar tutmazsa çalışmayı durdurur.

## 5 Katlı Doğrulama

Test kümesi (3.604) hiçbir zaman eğitime ya da model seçimine karışmaz. Kaggle'daki train (16.826) ve validation (3.594) klasörleri birleştirilerek 20.420 görüntülük bir havuz oluşturulur ve 5 fold bu havuzdan üretilir.

| `CV_MODE` | Her fold'da train / validation / test | Açıklama |
|---|---|---|
| `holdout5` (varsayılan) | **16.826 / 3.594 / 3.604** | 5 tekrarlı tabakalı (stratified) hold-out, yani Monte Carlo çapraz doğrulama. Her fold'daki sayılar orijinal Kaggle ayrımıyla birebir aynıdır. Validation kümeleri fold'lar arasında kısmen örtüşebilir. |
| `kfold` | ~16.336 / ~4.084 / 3.604 | Klasik stratified 5-fold. Her görüntü tam bir kez validation'da yer alır. |

Fold ataması bir kez üretilir ve `folds_<CV_MODE>.csv` dosyasına kaydedilir. **Bütün gruplar ve bütün modeller aynı fold dosyasını kullanır.** Her fold'un sınıf bazlı sayıları `fold_counts_<CV_MODE>.csv` dosyasına yazılır; makaledeki veri bölünmesi tablosu bu dosyadan alınmalıdır.

## Ortak Protokol (dört grupta da aynı)

| Ayar | Değer |
|---|---|
| Başlangıç ağırlıkları | ImageNet-1K ön eğitimli (transfer learning) |
| Görüntü boyutu / normalizasyon | 224×224 / ImageNet ortalama ve std |
| Epoch | Her fold'da 10, early stopping yok |
| Etkin batch | 32 |
| Model seçimi | En düşük validation loss'u veren epoch |
| Değerlendirme kaybı | Düz cross-entropy (label smoothing yok), bu yüzden gruplar karşılaştırılabilir |
| Mixed precision | Açık |
| Seed | 42 |
| Class weighting | Yok (veri seti dengeli) |
| Sınıflandırma başı | Dropout + Linear (2 sınıf) |

## Baseline ve Optimize Edilmiş Reçeteler

Aynı gruptaki CNN ve ViT modelleri **birebir aynı reçeteyi** kullanır.

| Teknik | Baseline | Optimize |
|---|---|---|
| Eğitilen katmanlar | Yalnızca sınıflandırma başı (backbone donuk, BatchNorm eval modunda) | CNN: tüm katmanlar · ViT: son bloklar (aşağıda) |
| Optimizer | Adam | AdamW |
| Learning rate | 1e-3 | 1e-4 |
| Weight decay | 0 | 1e-4 |
| Scheduler | Yok | Cosine annealing (1e-4'ten 1e-6'ya) |
| Data augmentation | Yok | Var |
| Label smoothing | 0 | 0,1 |
| Head dropout | 0 | 0,3 |
| Gradient clipping | Yok | 1,0 |

**Augmentation (yalnızca optimize deneylerde ve yalnızca train görüntülerine):** ±15° rastgele döndürme, yatay ve dikey çevirme (p = 0,5), %5 kaydırma ve 0,95–1,05 ölçekleme, parlaklık ve kontrast ±0,15, random erasing (p = 0,15).

**Optimize deneyde ince ayar kapsamı.** CNN ile ViT arasında bilerek bırakılan tek fark budur:

| Model | Eğitime açılan kısım |
|---|---|
| CNN'ler | Tüm katmanlar (full fine-tuning) |
| ViT-B/16, ViT-B/32 | Son 4/12 encoder bloğu, son LayerNorm ve head |
| ViT-L/16, ViT-L/32 | Son 4/24 encoder bloğu, son LayerNorm ve head |
| Swin-T | Son stage (`features.7`), son norm ve head |

Bütün ayarlar notebook'un ilk hücresinde (`COMMON`, `RECIPES`, `FT_SCOPE`, `AUGMENT`) tanımlıdır. Her grup çalıştırıldığında kullanılan ayarlar `config.json` dosyasına kaydedilir.

## Sızıntı Kontrolü

Eğitime başlamadan önce bir kez şu kontroller yapılır ve sonuç `leakage_check.csv` dosyasına yazılır:

- test ile train+val havuzu arasında birebir aynı (md5) ya da çok benzer (pHash, Hamming ≤ 2) görüntü var mı,
- train+val havuzunun kendi içinde aynı ya da çok benzer görüntü var mı (fold'lar arası sızıntı).

## Çıktılar

Sonuçlar Drive'a yazılır: `/content/drive/MyDrive/breast_mri_standard_results/<CV_MODE>/<GROUP>/<Model>/Fold_<k>/`

Her model ve her fold için:

- train, validation ve test için ayrı ayrı confusion matrix (başlıkta N yazar), ROC eğrisi ve metrik çubuk grafiği,
- accuracy ve loss eğrileri, epoch geçmişi (`history.csv`),
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1, ROC-AUC, Loss ve TN/FP/FN/TP sayıları (`metrics.csv`, `result.json`),
- istatistiksel testler (ör. McNemar) için örnek bazlı test tahminleri (`test_predictions.csv`).

Her grup notebook'u kendi `<GROUP>_results.xlsx` dosyasını üretir; bu dosya fold bazlı sonuçları, ortalama ± std değerlerini ve makale tablosunu içerir. `5_tum_sonuclar.ipynb` bütün grupları `ALL_RESULTS.xlsx` dosyasında birleştirir. Bu dosyada test, validation ve train için ortalama ± std tabloları, fold'lara göre veri bölünmesi ve hangi modelin kaç fold'u tamamladığı yer alır.

> Makalede sonuçlar **5 fold'un ortalaması ± standart sapması** olarak raporlanmalıdır. En iyi fold seçilerek raporlanmamalıdır.

## Çalıştırma

1. `1_baseline_cnn.ipynb` → `2_baseline_vit.ipynb` → `3_optimized_cnn.ipynb` → `4_optimized_vit.ipynb` notebook'larını sırayla Colab'da GPU ile aç ve bütün hücrelerini çalıştır.
2. Colab kesilirse aynı notebook'u yeniden çalıştırman yeterli; tamamlanan fold'lar atlanır.
3. Bir grup tek oturuma sığmazsa ilk hücrede `MODELS = ["ViT-L-16"]` gibi tek bir model ver. ViT-L modelleri en uzun sürenlerdir.
4. Dört grup bitince `5_tum_sonuclar.ipynb` ile `ALL_RESULTS.xlsx` dosyasını üret.

İlk notebook'ta fold ataması (`folds_holdout5.csv`) ve sızıntı kontrolü bir kez üretilir. Sonraki notebook'lar aynı dosyaları kullanır.

## Eski Deneyler (`legacy/`)

`legacy/` klasöründeki notebook'lar önceki deney setidir ve sadece arşiv amacıyla tutulur. Bu notebook'larda:

- Kaggle'daki `validation` klasörü kullanılmamış, yalnızca `train` klasörü (16.826 görüntü) 5'e bölünmüştür. Bu yüzden her fold yaklaşık 13.461 train / 3.366 validation görüntüsünden oluşur ve orijinal veri setinin sayılarıyla tutarsızdır.
- Gruplar arasında model seçim ölçütü, learning rate, label smoothing, early stopping ve batch ayarları farklıdır.
- `vision_transformers_baseline_equal_protocol.ipynb` adına rağmen bir baseline deneyi değildir. Bu notebook'ta augmentation ve tam ince ayar kullanılmıştır.

Makaledeki sonuçlar yukarıdaki dört grup notebook'u ile üretilen sonuçlara göre güncellenmelidir.
