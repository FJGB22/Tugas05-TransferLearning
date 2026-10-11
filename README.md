# Tugas 05 — Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Jonathan Kuniardi (2802461832) — Computer Science, BINUS University

Perbandingan tiga paradigma pelatihan ResNet-18 pada subset 10 kelas Oxford-IIIT Pet (±2.000 citra):
pelatihan dari nol, feature extraction, dan progressive fine-tuning, ditambah uji kontrol augmentasi dan
perbandingan dengan EfficientNet-B0 / MobileNetV3-Small.

## Kondisi eksperimen

| Kode | Kondisi | Parameter dilatih |
|---|---|---|
| B1 | From scratch (`weights=None`) | semua |
| B2 | Feature extraction | `fc` saja |
| B3 | Progressive fine-tuning, differential LR (layer4 1e-4, fc 1e-3) | `layer4` + `fc` |
| B4 | B3 tanpa augmentasi (pembanding untuk B3) | `layer4` + `fc` |
| B5 | Strategi terbaik (dipilih dari validasi) pada EfficientNet-B0 dan MobileNetV3-Small | blok terakhir + head |

Protokol: input 224×224, normalisasi ImageNet, batch 32, AdamW (weight decay 1e-2), CrossEntropyLoss,
early stopping pada validation loss (patience 5, maks. 30 epoch), seed 42, split stratified 70/15/15.

## Cara menjalankan

1. Buka `Tugas05_TransferLearning.ipynb` di Google Colab, pilih runtime **T4 GPU**.
2. Runtime → Run all. Dataset diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet`.
3. Semua hasil tersimpan di `results/`.

Untuk menjalankan lokal: `pip install -r requirements.txt`, lalu jalankan notebook dengan Jupyter.

## Isi `results/`

| Berkas | Isi |
|---|---|
| `summary.csv`, `summary.json` | metrik akhir per kondisi |
| `epoch_log.csv` | loss dan akurasi train/val per epoch |
| `consolidated_table.csv` | tabel konsolidasi C1 |
| `data_split.csv` | pembagian train/val/test per berkas |
| `test_predictions_<kondisi>.csv` | probabilitas prediksi tiap citra test |
| `misclassified_<terbaik>.csv` | sampel salah prediksi untuk failure analysis |
| `environment.json` | versi library, GPU, dan hyperparameter |
| `*.png` | kurva loss, perbandingan augmentasi, confusion matrix, failure cases, Grad-CAM |

## Referensi

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and Dogs. CVPR.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
- Howard, A., et al. (2019). Searching for MobileNetV3. ICCV.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
