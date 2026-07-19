# Breast MRI Classification with CNN and Vision Transformers

Bu depo, meme MRI görüntülerinin **Benign** ve **Malignant** olarak ikili sınıflandırılmasında CNN ve vision transformer tabanlı modellerin karşılaştırılması amacıyla hazırlanmıştır.

Çalışmanın temel amacı:

- CNN ve vision transformer modellerini ortak bir baseline protokolünde karşılaştırmak,
- optimize edilmemiş modellerin referans performanslarını belirlemek,
- vision transformer modellerini mimariye daha uygun tekniklerle optimize etmek,
- bütün modellerin fold bazlı train, validation ve bağımsız test sonuçlarını aynı metriklerle raporlamaktır.

## Depodaki Notebook'lar

### `BaseLineCNNModels.ipynb`

Optimize edilmemiş CNN modellerinin ortak baseline protokolüyle eğitildiği notebook'tur.

İçerdiği modeller:

- ResNet34
- DenseNet121
- EfficientNet-B0
- MobileNetV2
- InceptionV3

Bu notebook'ta ImageNet ön eğitimli modellerin feature extractor/backbone bölümleri dondurulur ve yalnızca yeni classification head eğitilir. Amaç, ek optimizasyon teknikleri uygulanmadan CNN mimarilerinin yalın transfer learning performanslarını ölçmektir.

### `BaseLineVITModels.ipynb`

Optimize edilmemiş vision transformer tabanlı modellerin baseline eğitimini içerir.

İçerdiği modeller:

- ViT-B/16
- ViT-B/32
- ViT-L/16
- ViT-L/32
- Swin Transformer Tiny (Swin-T)

CNN baseline ile adil karşılaştırma yapabilmek için aynı veri bölme, epoch, etkin batch size, optimizer, learning rate, loss, model seçimi ve değerlendirme yaklaşımı kullanılır. Pretrained encoder dondurulur ve yalnızca classification head eğitilir.

### `vision_transformers_baseline_equal_protocol.ipynb`

Vision transformer baseline deneylerini CNN baseline ile eşit ve açık biçimde tanımlanmış deney protokolünde çalıştıran notebook'tur. Beş modelin 5-fold eğitimini, best-epoch seçimini, fold bazlı değerlendirmesini ve standart çıktı üretimini tek akışta yürütür.

Bu notebook, transformer baseline sonuçlarının CNN sonuçlarıyla doğrudan karşılaştırılabilmesini sağlayan ana eşit-protokol uygulamasıdır.

### `vision_transformers_baseline_from_vit_l16_onward.ipynb`

`vision_transformers_baseline_equal_protocol.ipynb` çalışmasının yarıda kesilmesi nedeniyle oluşturulan devam notebook'udur.

Bu dosya:

- daha önce tamamlanan modelleri yeniden eğitmeden çalışmaya devam etmek,
- ViT-L/16 modelini Fold 1'den Fold 5'e kadar yeniden/tam olarak çalıştırmak,
- ardından ViT-L/32 ve Swin-T baseline deneylerini tamamlamak

amacıyla kullanılır.

Bu notebook yeni veya farklı bir deney protokolü değildir. Ana baseline notebook'uyla aynı koşulları kullanır ve eksik kalan deneyleri tamamlar. Sonuçlar yorumlanırken iki notebook'tan gelen çıktılar tek bir baseline deney dizisinin parçaları olarak ele alınmalıdır.

### `vision_transformers_vit_specific_optimized.ipynb`

Vision transformer modellerinin, ortak veri ve değerlendirme düzeni korunarak transformer mimarisine daha uygun tekniklerle optimize edildiği notebook'tur.

İçerdiği modeller:

- ViT-B/16
- ViT-B/32
- ViT-L/16
- ViT-L/32
- Swin-T

Bu deneyde yalnızca classification head eğitimi yerine modelin son transformer blokları da fine-tuning işlemine açılır. Böylece pretrained temsil korunurken modelin meme MRI veri setine uyarlanması hedeflenir.

## Veri Seti

Notebook'lar veri setini Google Drive'daki RAR arşivinden okur:

```text
/content/drive/MyDrive/data_set_zip/breast_mri_dataset.rar
```

Beklenen klasör yapısı:

```text
breast_mri_dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
└── test/
    ├── Benign/
    └── Malignant/
```

`train` klasörü stratified 5-fold cross-validation için kullanılır. `test` klasörü eğitimden bağımsız tutulur ve her fold'un seçilen modeliyle yalnızca nihai değerlendirme amacıyla kullanılır.

> ROC-AUC, sensitivity ve diğer pozitif sınıf metriklerinin doğru yorumlanabilmesi için `Malignant` sınıfının pozitif sınıf (`1`) olduğu doğrulanmalıdır.

## Baseline Deney Protokolü

CNN ve vision transformer baseline deneylerinde aşağıdaki ortak koşullar uygulanır:

| Özellik | Baseline ayarı |
|---|---|
| Başlangıç ağırlıkları | ImageNet1K pretrained |
| Transfer learning | Var |
| Eğitilen bölüm | Yalnızca classification head |
| Backbone/encoder | Dondurulmuş |
| Görüntü boyutu | 224 × 224 |
| Cross-validation | Stratified 5-fold |
| Epoch | Her fold için 10 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Loss | CrossEntropyLoss |
| Data augmentation | Yok |
| Scheduler | Yok |
| Early stopping | Yok |
| Class weighting | Yok |
| Label smoothing | Yok |
| Model seçim ölçütü | En yüksek validation accuracy |
| Random seed | 42 |

Baseline modeller “sıfırdan eğitim” değildir. ImageNet ön eğitimli özellik çıkarıcılar kullanıldığı için transfer learning uygulanır; ancak yalnızca classification head güncellendiğinden full fine-tuning yapılmaz.

## ViT'e Özel Optimize Edilmiş Deney Protokolü

Optimize edilmiş transformer deneylerinde veri seti, fold yapısı, bağımsız test kümesi ve raporlanan metrikler baseline deneyleriyle uyumludur. Eğitim stratejisi ise transformer mimarilerine daha uygun hale getirilmiştir.

| Özellik | ViT'e özel optimize ayarı |
|---|---|
| Başlangıç ağırlıkları | ImageNet1K pretrained |
| Fine-tuning | Classification head + son transformer blokları |
| ViT-B modelleri | Son 4/12 encoder bloğu |
| ViT-L modelleri | Son 4/24 encoder bloğu |
| Swin-T | Son hiyerarşik stage (`features.7`) |
| Görüntü boyutu | 224 × 224 |
| Cross-validation | Stratified 5-fold |
| En fazla epoch | 10 |
| Fiziksel batch size | 4 |
| Gradient accumulation | 8 adım |
| Etkin batch size | 32 |
| Optimizer | AdamW |
| Learning rate | `3e-5` |
| Weight decay | `1e-4` |
| Loss | Fold bazlı weighted CrossEntropyLoss |
| Label smoothing | `0.05` |
| Scheduler | Cosine annealing |
| Mixed precision | Var |
| Gradient clipping | `1.0` |
| Early stopping | Patience = 5 |
| Model seçim ölçütü | En düşük validation loss |
| Random seed | 42 |

### Kullanılan data augmentation işlemleri

Yalnızca training görüntülerine aşağıdaki işlemler uygulanır:

- Random rotation
- Random horizontal flip
- Random vertical flip
- Küçük konum ve ölçek değişimleri
- Parlaklık ve kontrast değişimleri
- Random erasing

Validation ve test görüntülerine augmentation uygulanmaz.

### Baseline ile temel farkı

Baseline deneyinde encoder tamamen dondurulur ve yalnızca head eğitilir. ViT'e özel optimize deneyde ise düşük learning rate ile son transformer katmanları kısmi olarak fine-tune edilir. Ayrıca AdamW, weight decay, augmentation, class weighting, label smoothing, cosine scheduler, mixed precision, gradient clipping ve early stopping kullanılır.

Bu nedenle optimize edilmiş deney, baseline ile aynı eğitim reçetesinin tekrarı değil; aynı veri ve değerlendirme çerçevesi üzerinde ViT mimarisine uygun bir optimizasyon senaryosudur.

## 5-Fold Eğitim, Best Epoch ve Test Protokolü

Her model için genel süreç şöyledir:

1. Eğitim verisi, sınıf oranları korunarak beş fold'a ayrılır.
2. Her iterasyonda dört fold training, bir fold validation için kullanılır.
3. Model yalnızca ilgili fold'un training bölümüyle eğitilir.
4. Her epoch sonunda train ve validation değerleri hesaplanır.
5. Baseline deneylerinde en yüksek validation accuracy; ViT'e özel optimize deneyde en düşük validation loss sağlayan epoch seçilir.
6. Seçilen best-epoch ağırlıkları geri yüklenir.
7. Seçilen modelle fold'un train, validation ve bağımsız test sonuçları hesaplanır.
8. Bu işlem beş fold için tekrarlanır ve her modelden beş ayrı test sonucu elde edilir.
9. Fold sonuçları ayrı ayrı ve ortalama ± standart sapma biçiminde raporlanır.

Test kümesi model veya epoch seçmek için kullanılmaz. Her epoch test edilmez; bağımsız test değerlendirmesi yalnızca validation sonucuyla seçilmiş fold modelinde yapılır.

## Değerlendirme Metrikleri

Her fold için ortak olarak aşağıdaki metrikler hesaplanır:

- Accuracy
- Precision
- Recall / Sensitivity
- F1-score
- Specificity
- ROC-AUC
- Confusion matrix
- ROC curve

Train, validation ve test sonuçları fold bazında kaydedilir. Model düzeyindeki özet tablolarda beş fold'un ortalama ve standart sapma değerleri raporlanır.

## Grafikler ve Tablolar

Notebook'lar model ve fold bazında aşağıdaki çıktıları üretir:

- Epoch bazlı train–validation accuracy grafiği
- Epoch bazlı train–validation loss grafiği
- Train confusion matrix
- Validation confusion matrix
- Test confusion matrix
- Test ROC curve
- Fold bazlı metrik tabloları
- Model bazlı ortalama ± standart sapma özetleri

Grafikler Colab hücrelerinin altında gösterilir ve çıktı klasörüne PNG olarak kaydedilir. Tablolar hücre altında görüntülenir; ayrıca CSV ve/veya Excel biçiminde `/content` altındaki ilgili çıktı klasörüne yazılır.

## Dosya Kaydetme Politikası

Google Drive yalnızca veri setini okumak için kullanılır. Sonuçlar Google Drive'a kaydedilmez.

Tüm deney çıktıları Colab'ın geçici dosya sistemi olan `/content` altında oluşturulur. Bu nedenle Colab oturumu kapanmadan önce notebook ve çıktı dosyaları GitHub'a veya yerel bilgisayara aktarılmalıdır. Colab oturumu sonlandırıldığında `/content` altındaki dosyalar silinebilir.

## Önerilen Çalıştırma Sırası

1. `BaseLineCNNModels.ipynb`
2. `vision_transformers_baseline_equal_protocol.ipynb`
3. Ana ViT baseline çalışması yarıda kaldıysa `vision_transformers_baseline_from_vit_l16_onward.ipynb`
4. `vision_transformers_vit_specific_optimized.ipynb`

`BaseLineVITModels.ipynb` önceki/alternatif ViT baseline notebook'u olarak saklanıyorsa, aynı modellerin sonuçlarını iki kez ana karşılaştırmaya eklememek gerekir. Makaledeki nihai baseline sonuçları için tek bir protokol ve tek bir sonuç seti esas alınmalıdır.

## Bilimsel Karşılaştırma Notu

Baseline CNN–ViT karşılaştırması, aynı eğitim ve değerlendirme protokolü kullanıldığı için doğrudan mimari karşılaştırma olarak sunulabilir.

ViT'e özel optimize sonuçlar ise “mimariye özel optimize edilmiş transformer performansı” olarak ayrı bir deney grubu şeklinde raporlanmalıdır. Baseline CNN sonuçlarıyla karşılaştırılabilir; ancak eğitim teknikleri farklı olduğu için bu karşılaştırma yalnızca mimari üstünlük iddiası olarak yorumlanmamalıdır.

Makale içinde sonuç tablolarında deney türü açıkça belirtilmelidir:

- **Baseline CNN**
- **Baseline Vision Transformer**
- **ViT-Specific Optimized Vision Transformer**

Bu ayrım, deneylerin tekrarlanabilirliğini ve sonuçların bilimsel olarak doğru yorumlanmasını sağlar.
