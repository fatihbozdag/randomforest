# Su Kalitesi Analiz Sistemi (RandomForest Projesi)

Bu depo; su kalitesi verilerini işleyen, çeşitli makine öğrenmesi modellerini (Random Forest, Gradient Boosting, SVR, Sinir Ağı, Ridge Regresyon ve Ensemble) eğiten, modelleri değerlendirip görselleştiren kapsamlı bir Python projesidir.

## İçindekiler
1. [Proje Yapısı](#proje-yapısı)
2. [Kurulum](#kurulum)
3. [Kullanım](#kullanım)
4. [Çıktı Dizini](#çıktı-dizini)
5. [Modeller ve Metotlar](#modeller-ve-metotlar)
6. [Lisans](#lisans)

## Proje Yapısı
```
RandomForest/
├─ config.py                # Genel yapılandırma
├─ data_processor.py        # Veri ön işleme ve özellik müh.
├─ main.py                  # Tam boru hattı çalıştırıcısı
├─ models/                  # Model sınıfları
├─ optimization.py          # Hiperparametre optimizasyonu
├─ evaluation.py            # ModelEvaluator
├─ visualization.py         # Görselleştirme fonksiyonları
├─ train.xlsx               # Eğitim verisi
├─ test.xlsx                # Test verisi
└─ requirements.txt         # Python bağımlılıkları
```

## Kurulum
```bash
# Sanal ortam (önerilen)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

## Kullanım
Tam boru hattını çalıştırmak için:
```bash
python main.py --full-pipeline
```
Bu komut; veriyi işler, tüm modelleri eğitir, değerlendirir, birleşik tahmin dosyalarını ve görselleri üretir.

### Komut Satırı Seçenekleri (Örnekler)
| Amaç | Komut |
|------|-------|
| Sadece belirli modelleri eğit | `python main.py --full-pipeline --models random_forest svr` |
| Hiperparametre optimizasyonu | `python main.py --full-pipeline --optimize --n-trials 50` |

## Çıktı Dizini
Eğitim sonrası oluşturulan klasör yapısı:
```
output/
├─ models/
│   └─ <model_adı>/        # Model dosyası (.joblib), metrikler (.json), grafikler (.png)
├─ predictions/            # Bütün modelleri içeren predictions_<timestamp>.csv/xlsx
├─ bcg_analysis/           # Resmî BCG analiz çıktıları
└─ reports/                # Genel raporlar (txt/json)
```

## Modeller ve Metotlar
- **Random Forest**: Ölçek duyarsız, outlier'lara dayanıklı.
- **Gradient Boosting (XGBoost fallback)**: Erken durma ile ayarlanmış.
- **SVR**: RBF kernel, öznitelik ölçeklendirme ile.
- **Sinir Ağı**: Küçük mimari `(64, 16)`, erken durma destekli.
- **Ridge Regresyon**: α=10, polinom & etkileşim özellikleri ile.
- **Ensemble**: Oy veren regresör; ağırlık optimizasyonu yapılabilir.

## Lisans
MIT 