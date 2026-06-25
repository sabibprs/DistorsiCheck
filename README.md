# DistorsiCheck

DistorsiCheck adalah aplikasi berbasis **Natural Language Processing (NLP)** yang digunakan untuk mendeteksi **distorsi informasi pada berita online berbahasa Indonesia** menggunakan metode **Complement Naive Bayes**, **TF-IDF**, dan **analisis semantik**. Aplikasi ini dibangun menggunakan **Python** dan **Streamlit** sehingga pengguna dapat melakukan klasifikasi berita secara langsung melalui antarmuka web.

🌐 **Demo Aplikasi:** https://distorsicheck.streamlit.app/

---

## Tentang Penelitian

Repository ini merupakan implementasi dari penelitian skripsi berjudul:

> **Deteksi Distorsi Informasi pada Berita Online Menggunakan Metode Naive Bayes dengan Pendekatan Analisis Semantik**

Penelitian ini bertujuan untuk mengklasifikasikan berita ke dalam dua kategori, yaitu:

* **Normal** : Berita yang tidak mengandung indikasi distorsi informasi.
* **Distorsi** : Berita yang mengandung indikasi distorsi berdasarkan fitur semantik yang telah ditentukan.

Tahapan penelitian meliputi:

1. Web Scraping
2. Data Labeling
3. Text Preprocessing
4. TF-IDF Feature Extraction
5. Semantic Feature Extraction
6. Complement Naive Bayes Classification
7. Model Evaluation

---

## Fitur

* Klasifikasi berita **Normal** dan **Distorsi**
* Text preprocessing Bahasa Indonesia
* TF-IDF dengan n-gram (1–3)
* Analisis fitur semantik berbasis leksikon
* Confidence score hasil prediksi
* Antarmuka berbasis Streamlit

---

## Dataset

### Dataset Hasil Web Scraping

Simpan dataset hasil scraping pada folder berikut:

```text
data/dataset_mentah
```

### Dataset Hasil Labeling

Simpan dataset yang telah melalui proses pelabelan pada folder berikut:

```text
data/dataset_balanced
```

---

## Menjalankan Aplikasi

Clone repository:

```bash
git clone https://github.com/sabibprs/DistorsiCheck.git
```

Masuk ke folder project:

```bash
cd DistorsiCheck
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

---

## Deploy

Aplikasi dapat diakses secara online melalui:

👉 **https://distorsicheck.streamlit.app/**

---

## Author

**Sabib Prastio**
Sistem Informasi - Universitas AMIKOM Yogyakarta

GitHub: https://github.com/sabibprs
