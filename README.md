# Laporan Proyek Machine Learning - Guruh Sukmo
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Pertanian**, dengan judul **Classification: Mushroom (Edible vs. Poisonous**  

### Latar Belakang

Gambar 1. Mushroom ![image](https://github.com/user-attachments/assets/5050a667-ef24-4e0c-b889-e5f46b21fdc8)


Jamur merupakan organisme yang umum ditemukan dan sering dimanfaatkan sebagai bahan pangan. Namun, tidak semua jenis jamur aman untuk dikonsumsi karena ada yang beracun dan bisa menyebabkan keracunan serius bahkan kematian. Identifikasi jamur yang akurat sangat penting, tetapi sulit dilakukan karena tidak ada aturan sederhana atau rumus pasti untuk membedakan jamur yang dapat dimakan dan yang beracun. Seiring meningkatnya kegiatan berburu jamur, kebutuhan akan metode identifikasi yang cepat dan akurat menjadi sangat penting. Teknologi machine learning menawarkan solusi yang menjanjikan untuk klasifikasi jamur berdasarkan fitur fisiknya.

## Business Understanding
Dalam konteks keselamatan pangan dan kesehatan masyarakat, mengembangkan alat bantu identifikasi jamur yang dapat dengan cepat dan tepat menentukan apakah jamur itu aman dikonsumsi sangat penting. Model prediktif yang akurat dapat digunakan oleh pemburu jamur, penjual, dan konsumen untuk menghindari konsumsi jamur beracun yang berpotensi fatal. Hal ini juga dapat membuka peluang pengembangan aplikasi berbasis AI yang mendukung edukasi dan pengamanan pangan.

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Problem Statement 1:
Bagaimana cara mengembangkan model machine learning yang dapat mengklasifikasikan jenis jamur sebagai edible (bisa dimakan) atau poisonous (beracun) dengan akurasi tinggi menggunakan fitur fisik yang tersedia pada dataset jamur?
- Problem Statement 2:
Mengingat tidak adanya aturan universal untuk membedakan jamur beracun dan yang aman dikonsumsi, bagaimana membangun sistem identifikasi otomatis yang cepat dan andal untuk membantu pemburu dan konsumen jamur menghindari risiko keracunan?

### Goals
Tujuan dari proyek ini adalah:
- Membangun model klasifikasi jamur yang akurat menggunakan dataset jamur dari UCI Machine Learning Repository.
- Mengidentifikasi fitur-fitur fisik jamur yang paling berpengaruh dalam menentukan edibilitas.
- Menghasilkan model yang dapat membantu pengguna secara praktis dalam membedakan jamur aman dan beracun.
- Meningkatkan kesadaran dan keselamatan masyarakat dalam konsumsi jamur.

### Solution Statements
- Mengumpulkan dan membersihkan dataset jamur yang berisi lebih dari 8.000 sampel.
- Melakukan eksplorasi data untuk memahami fitur-fitur fisik jamur yang ada.
- Menerapkan teknik machine learning seperti Logistic Regression, Naive Bayes, atau Random Forest untuk membangun model klasifikasi.
- Melakukan evaluasi dan tuning model untuk mendapatkan akurasi terbaik.

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | Mushroom Classification |
| Source | [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification/data) |
| License | Public Domain |
| Visibility | Public |
| Tags | Earth and Nature, Biology, Public Safety, Benchmark |
| Usability | 8.53 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data dari UCI Repository, data ini didapat dari sebuah perusahaan pertanian Amerika Utara, yang disediakan secara publik di kaggle dengan nama datasets yaitu: Mushroom Classification

Tabel 1. EDA Deskripsi Variabel

Dilihat dari _Tabel 1. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 8124 baris dengan 23 kolom.
- Dataset bertipe object semua.
- Tidak terdapat missing value dalam dataset.
### Variable - variable pada dataset
- `class` : Kelas jamur — edible (bisa dimakan, e), poisonous (beracun, p),
- `cap-shape` : Bentuk tudung jamur — bell (b), conical (c), convex (x), flat (f), knobbed (k), sunken (s),
- `cap-surface` : Permukaan tudung — fibrous (f), grooves (g), scaly (y), smooth (s),
- `cap-color` : Warna tudung — brown (n), buff (b), cinnamon (c), gray (g), green (r), pink (p), purple (u), red (e), white (w), yellow (y),
- `bruises` : Apakah jamur memiliki memar — bruises (t), no (f),
- `odor` : Bau jamur — almond (a), anise (l), creosote (c), fishy (y), foul (f), musty (m), none (n), pungent (p), spicy (s),
- `gill-attachment` : Cara melekatnya insang — attached (a), descending (d), free (f), notched (n),
- `gill-spacing` : Jarak antar insang — close (c), crowded (w), distant (d),
- `gill-size` : Ukuran insang — broad (b), narrow (n),
- `gill-color` : Warna insang — black (k), brown (n), buff (b), chocolate (h), gray (g), green (r), orange (o), pink (p), purple (u), red (e), white (w), yellow (y),
- `stalk-shape` : Bentuk batang — enlarging (e), tapering (t),
- `stalk-root` : Akar batang — bulbous (b), club (c), cup (u), equal (e), rhizomorphs (z), rooted (r), missing (?),
- `stalk-surface-above-ring` : Permukaan batang di atas cincin — fibrous (f), scaly (y), silky (k), smooth (s),
- `stalk-surface-below-ring` : Permukaan batang di bawah cincin — fibrous (f), scaly (y), silky (k), smooth (s),
- `stalk-color-above-ring` : Warna batang di atas cincin — brown (n), buff (b), cinnamon (c), gray (g), orange (o), pink (p), red (e), white (w), yellow (y),
- `stalk-color-below-ring` : Warna batang di bawah cincin — brown (n), buff (b), cinnamon (c), gray (g), orange (o), pink (p), red (e), white (w), yellow (y),
- `veil-type` : Tipe tirai — partial (p), universal (u),
- `veil-color` : Warna tirai — brown (n), orange (o), white (w), yellow (y),
- `ring-number` : Jumlah cincin — none (n), one (o), two (t),
- `ring-type` : Tipe cincin — cobwebby (c), evanescent (e), flaring (f), large (l), none (n), pendant (p), sheathing (s), zone (z),
- `spore-print-color` : Warna spora — black (k), brown (n), buff (b), chocolate (h), green (r), orange (o), purple (u), white (w), yellow (y),
- `population` : Pola populasi jamur — abundant (a), clustered (c), numerous (n), scattered (s), several (v), solitary (y),
- `habitat` : Habitat tumbuh jamur — grasses (g), leaves (l), meadows (m), paths (p), urban (u), waste (w), woods (d),


## Data Preparation
### Encoding
Pada dataset jamur ini, seluruh fitur direpresentasikan dalam bentuk kategori (huruf-huruf). Machine learning model umumnya tidak bisa bekerja langsung dengan data kategorikal, sehingga perlu dilakukan proses encoding, yaitu mengubah nilai kategori menjadi nilai numerik.
Label Encoding: Setiap kategori diubah menjadi angka unik. Cocok untuk data kategorikal ordinal.
Untuk dataset jamur ini, Label Encoding sering digunakan karena semua fitur kategorikal, dan jumlah kategorinya relatif banyak sehingga one-hot encoding bisa menyebabkan dimensionalitas tinggi.

### Feature Selection
Feature Selection adalah proses memilih fitur-fitur paling relevan yang berkontribusi terhadap prediksi model. Tujuan utamanya adalah:
- Mengurangi kompleksitas model.
- Meningkatkan akurasi dan generalisasi model.
- Mengurangi risiko overfitting.

#### Feature Selection menggunakan Chi-Square
Metode Chi-Square (χ²) adalah teknik statistik yang digunakan untuk mengukur ketergantungan antara dua variabel kategorikal — dalam konteks ini, antara setiap fitur dan label (target).

Tujuannya adalah untuk menentukan fitur mana yang paling berpengaruh terhadap target (misalnya: edible atau poisonous).

Cara Kerja:
- Menghitung nilai chi-square untuk setiap fitur terhadap target.
- Semakin tinggi nilai chi-square, semakin besar kontribusi fitur tersebut terhadap prediksi kelas.
- Fitur dengan nilai chi-square rendah dianggap tidak relevan dan bisa dihapus.

Gambar 2. Fitur terpilih ![image](https://github.com/user-attachments/assets/f77df58a-3361-4f38-9e3d-347f2093c459)


## Modeling
Setelah proses preprocessing dan feature selection selesai, tahap selanjutnya adalah membangun dan mengevaluasi model klasifikasi. Tiga algoritma yang digunakan dalam proyek ini adalah:

- Logistic Regression
- Ridge Classifier
- Naive Bayes (GaussianNB)

Kelebihan dan Kekurangan Model
### 1. Logistic Regression
Kelebihan:
- Sederhana, cepat, dan mudah diimplementasikan.
- Memberikan hasil prediksi dalam bentuk probabilitas.
- Cocok untuk kasus klasifikasi biner seperti edible vs poisonous.
- Mudah untuk diinterpretasikan secara statistik.

Kekurangan:
- Asumsi hubungan linier antara fitur dan target membatasi performa pada data kompleks.
- Tidak optimal jika fitur memiliki multikolinearitas tinggi.
- Tidak menangani data non-linear tanpa transformasi tambahan.

### 2. Ridge Classifier
Kelebihan:
- Memiliki regularisasi L2 (Ridge) untuk mengurangi overfitting.
- Lebih stabil dalam menangani fitur yang saling berkorelasi.
- Cocok untuk data berdimensi tinggi.

Kekurangan:
- Tetap berbasis model linier sehingga kurang efektif pada data non-linear.
- Interpretasi model lebih kompleks dibanding logistic regression karena adanya penalti regularisasi.
- Tidak sefleksibel model non-linear seperti decision tree atau SVM.

### 3. Naive Bayes (GaussianNB)
Kelebihan:
- Proses pelatihan dan prediksi sangat cepat, efisien untuk dataset besar.
- Performa cukup baik meskipun asumsi independensi fitur tidak sepenuhnya terpenuhi.
- Sangat cocok untuk data kategorikal atau yang telah melalui proses encoding.

Kekurangan:
- Asumsi independensi antar fitur sering tidak realistis.
- Tidak mampu menangani interaksi antar fitur.
- Kurang akurat jika distribusi data tidak sesuai dengan asumsi Gaussian (untuk GaussianNB).

## Evaluation

- Logistic Regression memberikan performa terbaik dengan akurasi 90.46%, diikuti oleh Ridge Classifier (88.62%) dan Naive Bayes (88.37%).
- Model cukup seimbang dalam mengklasifikasikan kedua kelas (edible vs poisonous), dengan nilai precision, recall, dan f1-score yang tinggi.
- Logistic Regression memiliki keseimbangan terbaik antara kelas 0 (edible) dan kelas 1 (poisonous).


Gambar 3. Perbandingan Akurasi Model ![image](https://github.com/user-attachments/assets/f64a6d78-9e3d-47b3-8df8-7fac39528533)







## Referensi
1. Machine Learning for Mushroom Classification – Towards Data Science - https://towardsdatascience.com/machine-learning-on-mushroom-dataset-6d9853435d3e
2. Zhang, Y., Liu, Q., Wang, X., & Zhao, H. (2022). Mushroom classification using deep convolutional neural networks. arXiv preprint arXiv:2210.10351. https://arxiv.org/abs/2210.10351
3. Arifianto, Y. A., Djamaluddin, M., & Rahman, A. (2022). Improving Mushroom Classification Accuracy Using Transfer Learning with AlexNet. Applied Sciences, 12(7), 3409. https://www.mdpi.com/2076-3417/12/7/3409
4. Tianmitrapap, J., & Viyanon, S. (2025). Comparison of Five Supervised Machine Learning Algorithms for Mushroom Classification. Thailand Statistician, 23(1), 45–62. https://li01.tci-thaijo.org/index.php/tstj/article/view/258636
5. Rianti, A., Anwar, A., & Desanti, F. A. (2023). Klasifikasi Jamur Layak Konsumsi Menggunakan Algoritma Decision Tree. Jurnal Artificial Intelligence dan Komputasi, 3(2), 85–94. https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/4087
6. GeeksforGeeks. (2021). Feature Selection using Chi-Square Test. GeeksforGeeks.org. https://www.geeksforgeeks.org/feature-selection-using-chi-square-test

_
