# Çıktılar

Bu klasör, `1_baseline_cnn`, `2_baseline_vit`, `3_optimized_cnn` ve `4_optimized_vit` notebook'larının içindeki bütün grafik ve tabloların dışa aktarılmış halidir. Hepsi notebook'lardaki çıktılardan otomatik olarak çıkarılmıştır; yeniden eğitim yapılmamıştır.

```
çıktılar/
├── veri_bolunmesi/
│   ├── split_sayilari.csv        Kaggle train / validation / test sayıları (sınıf bazında)
│   ├── fold_sayilari.csv         Her fold'da train / validation / test sayıları (16.826 / 3.594 / 3.604)
│   └── sizinti_kontrolu.csv      Split'ler arası aynı / çok benzer görüntü kontrolü
├── 1_baseline_cnn/
│   ├── grup_ozeti_test.csv       Gruptaki modellerin test sonuçları (5 fold ort. ± std)
│   ├── ResNet34/
│   │   ├── fold_sonuclari.csv    5 fold × Train / Validation / Test bütün metrikler
│   │   ├── ozet_ort_std.csv      Train / Validation / Test için ortalama ± std (%)
│   │   ├── Fold_1/
│   │   │   ├── accuracy_curve.png, loss_curve.png
│   │   │   ├── cm_train.png, cm_validation.png, cm_test.png             (confusion matrix)
│   │   │   ├── roc_train.png, roc_validation.png, roc_test.png
│   │   │   ├── metrics_train.png, metrics_validation.png, metrics_test.png
│   │   │   ├── metrics.csv       Bu fold'un Train / Validation / Test metrikleri (N, TN, FP, FN, TP dahil)
│   │   │   └── egitim_log.txt    Epoch epoch train / validation loss ve doğruluk
│   │   └── Fold_2 … Fold_5/
│   └── DenseNet121/, EfficientNet-B0/, MobileNetV2/, InceptionV3/
├── 2_baseline_vit/               ViT-B-16, ViT-B-32, ViT-L-16, ViT-L-32, Swin-T
├── 3_optimized_cnn/              5 CNN (optimize)
└── 4_optimized_vit/              4 ViT + Swin-T (optimize)
```

Her fold klasöründe 11 grafik vardır. Toplamda 4 grup × 5 model × 5 fold = 100 fold klasörü ve 1.100 grafik bulunur.

Kapsamlı karşılaştırma grafikleri (Baseline ve Optimize karşılaştırması, ortalama ROC, ortalama confusion matrix) için `5_tum_sonuclar.ipynb` notebook'u kullanılır.
