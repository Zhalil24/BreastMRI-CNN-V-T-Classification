# Breast MRI Classification: CNN and Vision Transformer Comparison

Bu çalışma, meme MRI görüntülerinin **Benign** ve **Malignant** olarak ikili sınıflandırılmasında CNN ve vision transformer tabanlı modellerin performanslarını karşılaştırmak amacıyla hazırlanmıştır.

Deneyler iki grupta yürütülmektedir:

1. **Optimize edilmemiş (baseline) modeller**
2. **Eşit koşullarda optimize edilmiş modeller**

Karşılaştırmanın adil olması için aynı deney grubundaki bütün CNN ve transformer modellerine aynı veri bölme, eğitim, model seçimi ve değerlendirme protokolü uygulanmıştır.

## Kullanılan Modeller

### CNN modelleri

- ResNet34
- DenseNet121
- EfficientNet-B0
- MobileNetV2
- InceptionV3

### Vision transformer tabanlı modeller

- ViT-B/16
- ViT-B/32
- ViT-L/16
- ViT-L/32
- Swin Transformer Tiny (Swin-T)

> Swin-T klasik ViT ailesinin doğrudan bir üyesi değildir. Kaydırılmış pencere tabanlı, hiyerarşik bir vision transformer mimarisidir. Bu nedenle çalışmada genel olarak “vision transformer tabanlı modeller” ifadesi kullanılmaktadır.

## Veri Seti ve Ön İşleme

Kodlar veri setini aşağıdaki Google Drive yolundan okumaktadır:

```text
/content/drive/MyDrive/data_set_zip/breast_mri_dataset.rar
```

RAR dosyası açıldığında aşağıdaki klasör yapısı beklenmektedir:

```text
breast_mri_dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
└── test/
    ├── Benign/
    └── Malignant/
```

Bütün deneylerde görüntüler `224×224` boyutuna getirilmekte ve ImageNet ortalama/standart sapma değerleriyle normalize edilmektedir:

```text
Mean: [0.485, 0.456, 0.406]
Std:  [0.229, 0.224, 0.225]
```

Sınıf `1`, pozitif sınıf olarak kabul edilmekte ve `Malignant` sınıfını temsil etmelidir.

## Optimize Edilmemiş Modeller (Baseline)

Baseline deneylerinin amacı, CNN ve transformer mimarilerini ek optimizasyon teknikleri olmadan aynı ve yalın eğitim protokolünde karşılaştırmaktır.

### Baseline eğitim yaklaşımı

- ImageNet1K pretrained ağırlıkları kullanılır.
- Transfer learning uygulanır.
- Modelin backbone/encoder katmanları dondurulur.
- Yalnızca tek doğrusal classification head eğitilir.
- Veri artırma uygulanmaz.
- Bütün modeller `224×224` görüntülerle çalıştırılır.
- CNN backbone’larındaki BatchNorm istatistikleri değiştirilmez.
- Dondurulmuş backbone içindeki dropout ve stochastic-depth davranışları kapalı tutulur.
- Scheduler kullanılmaz.
- Early stopping kullanılmaz.
- Class weighting kullanılmaz.
- Label smoothing kullanılmaz.
- Mixed-precision training kullanılmaz.
- Gradient clipping kullanılmaz.
- Differential learning rate kullanılmaz.

### Baseline hiperparametreleri

| Parametre | Değer |
|---|---:|
| Pretrained ağırlık | ImageNet1K V1 |
| Görüntü boyutu | 224×224 |
| Fold sayısı | 5 |
| Epoch | Her fold için 10 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | `1×10⁻³` |
| Loss | CrossEntropyLoss |
| Weight decay | 0 |
| Random seed | 42 |
| Model seçim ölçütü | En yüksek validation accuracy |

### Baseline kodları

```text
cnn_baseline_equal_protocol.py
vision_transformers_baseline_equal_protocol.py
```

## Optimize Edilmiş Modeller

Optimize edilmiş deneylerde bütün CNN ve transformer modelleri aynı modern optimizasyon protokolüyle eğitilmektedir. Amaç, bir model grubuna diğerinden daha fazla eğitim avantajı sağlamadan mimarilerin optimize edilmiş performanslarını karşılaştırmaktır.

### Kullanılan optimizasyon teknikleri

#### Full fine-tuning

Pretrained modelin yalnızca classification head’i değil, bütün katmanları eğitilir.

#### Data augmentation

Train görüntülerine aşağıdaki veri artırma işlemleri uygulanır:

- Random rotation (`±15°`)
- Random horizontal flip (`p=0.5`)
- Random vertical flip (`p=0.2`)
- Küçük konum kaydırma (`±5%`)
- Küçük ölçek değişimi (`0.95–1.05`)
- Parlaklık değişimi (`0.15`)
- Kontrast değişimi (`0.15`)
- Random erasing (`p=0.15`)

Validation ve test kümelerine veri artırma uygulanmaz.

#### AdamW ve weight decay

Optimizer olarak AdamW kullanılmakta ve aşırı öğrenmeyi azaltmak için weight decay uygulanmaktadır.

#### Cosine learning-rate scheduler

Learning rate, 10 epoch boyunca `1×10⁻⁴` değerinden `1×10⁻⁶` değerine kosinüs biçiminde azaltılır.

#### Fold bazlı class weighting

Sınıf ağırlıkları her fold’da yalnızca ilgili training bölümünün sınıf dağılımından hesaplanır. Validation veya test etiketleri class-weight hesabında kullanılmaz.

#### Label smoothing

Modelin aşırı kendinden emin tahminler üretmesini azaltmak amacıyla `0.1` label smoothing uygulanır.

#### Dropout

Bütün modellerin yeni classification head’ine `0.3` dropout eklenir.

#### Mixed-precision training

GPU bellek kullanımını azaltmak ve eğitimi hızlandırmak amacıyla FP16/FP32 mixed precision kullanılır.

#### Gradient clipping

Kararsız gradient büyümelerini sınırlandırmak için gradient normu `1.0` ile sınırlandırılır.

#### Gradient accumulation

ViT-L modelleri de dâhil olmak üzere bütün modellerin aynı etkin batch size ile eğitilebilmesi için:

```text
Fiziksel batch size: 4
Accumulation steps: 8
Etkin batch size: 4 × 8 = 32
```

kullanılır.

### Optimize edilmiş deney hiperparametreleri

| Parametre | Değer |
|---|---:|
| Pretrained ağırlık | ImageNet1K V1 |
| Fine-tuning | Bütün katmanlar |
| Görüntü boyutu | 224×224 |
| Fold sayısı | 5 |
| Epoch | Her fold için 10 |
| Fiziksel batch size | 4 |
| Gradient accumulation | 8 adım |
| Etkin batch size | 32 |
| Optimizer | AdamW |
| Learning rate | `1×10⁻⁴` |
| Minimum learning rate | `1×10⁻⁶` |
| Weight decay | `1×10⁻⁴` |
| Scheduler | CosineAnnealingLR |
| Loss | Weighted CrossEntropyLoss |
| Label smoothing | 0.1 |
| Classification head dropout | 0.3 |
| Gradient clipping | 1.0 |
| Mixed precision | Aktif |
| Random seed | 42 |
| Model seçim ölçütü | En yüksek validation accuracy |

### Optimize edilmiş model kodları

```text
cnn_optimized_equal_protocol.py
vision_transformers_optimized_equal_protocol.py
```

## 5-Fold Eğitim ve Best Epoch Protokolü

Hem baseline hem de optimize edilmiş deneylerde aynı model seçme ve test protokolü uygulanır.

Her model için:

1. Training verisi sınıf oranları korunarak stratified 5-fold'a ayrılır.
2. Her fold 10 epoch eğitilir.
3. Her epoch sonunda train ve validation loss/accuracy değerleri hesaplanır.
4. Validation accuracy değeri en yüksek epoch, ilgili fold'un **best epoch** değeri olarak seçilir.
5. On epoch tamamlandıktan sonra best epoch ağırlıkları geri yüklenir.
6. Seçilen modelle train, validation ve bağımsız test kümeleri değerlendirilir.
7. Test kümesi her fold için yalnızca bir kez değerlendirilir.
8. Beş fold'dan beş test sonucu elde edilir.
9. Sonuçlar ortalama ± standart sapma biçiminde raporlanır.

```text
Fold 1 → 10 epoch → Best epoch → Test sonucu 1
Fold 2 → 10 epoch → Best epoch → Test sonucu 2
Fold 3 → 10 epoch → Best epoch → Test sonucu 3
Fold 4 → 10 epoch → Best epoch → Test sonucu 4
Fold 5 → 10 epoch → Best epoch → Test sonucu 5
```

Test kümesi epoch veya model seçmek için kullanılmaz. Best epoch yalnızca validation accuracy üzerinden belirlenir.

## Değerlendirme Metrikleri

Her fold için aşağıdaki metrikler hesaplanır:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

Her model için beş fold sonucunun ortalama ve standart sapması raporlanır.

## Üretilen Çıktılar

Kodlar sonuçları Google Drive'a yazmaz. Drive yalnızca veri setini okumak için bağlanır. Çıktılar Colab'ın geçici `/content` dizininde oluşturulur.

```text
/content/cnn_baseline_equal_protocol_outputs
/content/vision_transformers_baseline_equal_protocol_outputs
/content/cnn_optimized_equal_protocol_outputs
/content/vision_transformers_optimized_equal_protocol_outputs
```

Her model ve fold için aşağıdaki grafikler hem PNG olarak oluşturulur hem de Colab hücresinde gösterilir:

- Train–validation accuracy grafiği
- Train–validation loss grafiği
- Train confusion matrix
- Validation confusion matrix
- Test confusion matrix
- Train ROC curve
- Validation ROC curve
- Test ROC curve
- Train metrik grafiği
- Validation metrik grafiği
- Test metrik grafiği

Her model için oluşturulan Excel dosyaları şu sayfaları içerir:

- `Epoch History`
- `Train Results`
- `Validation Results`
- `Test Results`
- `Mean and Std`

Fold atamaları ayrıca `Fold Assignments.csv` dosyasında tutulur.

## Google Colab ve GitHub Kullanımı

Notebook GitHub'a kaydedilirken **çıktıları dâhil et (include outputs)** seçeneği işaretlenirse hücre altında gösterilen grafik ve tablolar GitHub'daki notebook içerisinde görünür.

`/content` altında oluşturulan PNG, CSV ve Excel dosyaları notebook ile birlikte otomatik yüklenmez. Bu dosyaların repoda ayrı dosyalar olarak bulunması isteniyorsa Colab oturumu sona ermeden indirilmeli ve repoya ayrıca eklenmelidir.

Tam fine-tuning sonucunda oluşabilecek büyük model ağırlıkları normal GitHub dosya sınırlarını aşabileceğinden optimize edilmiş kodlar `.pth` ağırlıklarını kaydetmez.

## Bilimsel Karşılaştırma Notları

- Baseline CNN ve transformer modelleri kendi aralarında aynı protokolle eğitilir.
- Optimize edilmiş CNN ve transformer modelleri kendi aralarında aynı protokolle eğitilir.
- Baseline ve optimize edilmiş gruplarda aynı fold ve best-epoch yaklaşımı kullanılır.
- Test kümesi model seçimi veya hiperparametre ayarı için kullanılmaz.
- “En iyi optimize edilmiş” yerine **eşit optimize edilmiş** veya **ortak optimize edilmiş protokol** ifadesi daha doğrudur; mutlak en iyi hiperparametrelerin gösterilmesi ayrıca hiperparametre taraması gerektirir.
- Aynı hastaya ait birden fazla görüntü varsa görüntü seviyesinde fold ayrımı veri sızıntısına neden olabilir. Bu durumda hasta kimliğine göre `StratifiedGroupKFold` kullanılmalı ve aynı hasta yalnızca tek bir fold'da bulunmalıdır.

## Deney Özeti

| Deney grubu | Eğitilen katmanlar | Optimizer | LR | Augmentation | Etkin batch |
|---|---|---|---:|---|---:|
| Baseline CNN | Yalnızca head | Adam | `1×10⁻³` | Yok | 32 |
| Baseline transformer | Yalnızca head | Adam | `1×10⁻³` | Yok | 32 |
| Optimize CNN | Bütün model | AdamW | `1×10⁻⁴` | Var | 32 |
| Optimize transformer | Bütün model | AdamW | `1×10⁻⁴` | Var | 32 |

Bu yapı sayesinde aynı deney grubundaki performans farklarının eğitim protokolündeki eşitsizliklerden değil, büyük ölçüde model mimarilerinden kaynaklanması hedeflenmektedir.
