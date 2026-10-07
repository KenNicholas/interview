## Executive Summary & Research Findings

### 1. Justifikasi Metrik Evaluasi
* Dataset ini dievaluasi terutama menggunakan metrik **F1-Score (Weighted/Macro)** dan **ROC-AUC**.
* *Alasan:* Metrik Accuracy tidak digunakan sebagai acuan utama karena [sebutkan jika ada class imbalance / ketidakseimbangan kelas target]. F1-Score memberikan keseimbangan antara Precision dan Recall, yang krusial untuk meminimalkan dampak *False Positives* dan *False Negatives*.

### 2. Analisis Model Trade-off
* **Model Terbaik:** [Nama Model, misal: LightGBM] mencapai F1-score tertinggi (**[Nilai, misal: 0.87]**) dengan waktu pelatihan **[Nilai] detik**.
* **Trade-off:** 
  * Jika fokus bisnis adalah **Akurasi & Performa Tertinggi**, maka [LightGBM/Random Forest] adalah pilihan utama.
  * Jika fokus bisnis adalah **Latensi Inferensi Kilat & Explainability**, maka [Logistic Regression/Decision Tree] dapat dipertimbangkan karena selisih performanya relatif kecil ([Sebutkan selisih F1-score]), namun jauh lebih ringan di lingkungan *production*.

### 3. Rencana Pengembangan Lanjutan (Next Steps)
Jika diberikan alokasi waktu tambahan, langkah eksperimen selanjutnya meliputi:
1. **Hyperparameter Tuning:** Melakukan `RandomizedSearchCV` atau `Optuna` pada model [Nama Model Terbaik] untuk mengoptimalkan hyperparameter kunci.
2. **Advanced Feature Engineering:** Membentuk fitur interaksi baru berdasarkan domain knowledge domain keuangan/otomotif/perilaku pengguna.
3. **Imbalanced Handling:** Menerapkan teknik resampling (*SMOTE* atau *Class Weighting*) jika rasio kelas target sangat timpang.
4. **Model Explainability:** Mengintegrasikan library `SHAP` untuk menganalisis kontribusi setiap fitur terhadap keputusan model secara individual.