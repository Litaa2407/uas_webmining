# Dashboard Daerah Kembar Pendidikan Indonesia

Proyek Web Mining ini membandingkan pola kesenjangan pendidikan pada 38 provinsi
Indonesia berdasarkan data BPS tahun 2025. Setiap provinsi direpresentasikan
dengan sembilan indikator pendidikan. Pipeline menghitung gap terhadap nilai
nasional, melakukan robust scaling, memberi bobot dengan Entropy Weight Method,
menghitung kemiripan dengan weighted cosine, membentuk graf k-nearest neighbors
(k-NN), lalu mendeteksi komunitas menggunakan Louvain.

Analisis ini menggambarkan kemiripan pola, bukan hubungan sebab-akibat.

## Tujuan analisis

Proyek ini dirancang untuk menjawab pertanyaan berikut:

- Provinsi mana yang memiliki pola kesenjangan pendidikan paling mirip?
- Apakah provinsi yang dekat secara geografis juga mirip secara pendidikan?
- Indikator apa yang paling membedakan pola antarprovinsi?
- Apakah pola kemiripan pendidikan berkaitan dengan IPM dan kemiskinan?

## Data yang digunakan

Data sumber berada di folder [`data/`](data/). Pipeline inti membaca lima tabel:

| Tabel | Penggunaan |
|---|---|
| Rata-rata lama sekolah menurut provinsi, 2025 | Fitur pendidikan |
| APS menurut provinsi dan kelompok umur, 2025 | Fitur pendidikan |
| Jumlah desa yang memiliki fasilitas sekolah, 2025 | Dasar untuk fitur rasio fasilitas |
| IPM menurut provinsi, 2025 | Pembanding untuk validasi, bukan fitur similarity |
| Jumlah dan persentase penduduk miskin, 2025 | Pembanding untuk validasi, bukan fitur similarity |

Baris `INDONESIA` pada tabel yang memilikinya digunakan sebagai nilai acuan
nasional dan tidak dihitung sebagai provinsi. CSV sumber tidak diubah oleh
pipeline.

File lain yang ada di `data/` berisi konteks nasional atau data koordinat daerah.
File-file tersebut belum digunakan oleh script preprocessing dan mining inti
yang dijelaskan di bawah.

## Indikator utama

Setiap provinsi memiliki sembilan fitur:

1. `RLS`: rata-rata lama sekolah penduduk umur 15 tahun ke atas (tahun).
2. `APS_7_12`: angka partisipasi sekolah umur 7–12 tahun (persen).
3. `APS_13_15`: angka partisipasi sekolah umur 13–15 tahun (persen).
4. `APS_16_18`: angka partisipasi sekolah umur 16–18 tahun (persen).
5. `APS_19_23`: angka partisipasi sekolah umur 19–23 tahun (persen).
6. `Rasio_SMP_SD`: jumlah desa dengan fasilitas SMP dibagi jumlah desa dengan
   fasilitas SD.
7. `Rasio_SMU_SD`: jumlah desa dengan fasilitas SMU dibagi jumlah desa dengan
   fasilitas SD.
8. `Rasio_SMK_SD`: jumlah desa dengan fasilitas SMK dibagi jumlah desa dengan
   fasilitas SD.
9. `Rasio_PT_SD`: jumlah desa dengan fasilitas perguruan tinggi dibagi jumlah
   desa dengan fasilitas SD.

Empat rasio fasilitas digunakan untuk mengurangi pengaruh skala jumlah desa
antarprovinsi. IPM dan kemiskinan sengaja tidak dimasukkan ke dalam sembilan
fitur tersebut agar dapat digunakan sebagai pembanding eksternal.

## Alur pipeline

### 1. Preprocessing data

Script: [`preprocess.py`](preprocess.py)

Script ini membaca CSV sumber, menyeragamkan nama provinsi, mengubah kolom
indikator menjadi numerik, memisahkan baris nasional, menggabungkan tabel
provinsi, membuat empat rasio fasilitas, lalu memeriksa duplikasi, missing
values, dan jumlah provinsi. Hasilnya disimpan di
[`output_preprocessing/`](output_preprocessing/).

Output:

| File | Isi |
|---|---|
| `01_dataset_pendidikan_38_provinsi.csv` | 38 provinsi dan sembilan fitur utama |
| `02_dataset_validasi.csv` | IPM dan persentase kemiskinan September, terpisah dari fitur |
| `03_referensi_nasional.csv` | Nilai nasional untuk lima indikator dan empat rasio |
| `04_fasilitas_mentah.csv` | Jumlah fasilitas mentah sebelum dihitung rasio |

### 2. Vektor gap dan robust scaling

Script: [`gap_vector.py`](gap_vector.py)

Untuk provinsi \(i\) dan fitur \(j\), script menghitung:

```text
gap(i,j) = (nilai_provinsi(i,j) - nilai_nasional(j)) / IQR(j)
```

IQR dihitung dari nilai 38 provinsi untuk fitur tersebut. Nilai gap negatif
menunjukkan nilai provinsi berada di bawah acuan nasional; nilai positif
menunjukkan berada di atasnya. Pencilan tidak dihapus.

Output:

| File | Isi |
|---|---|
| `05_gap_vector_raw.csv` | Selisih nilai provinsi terhadap nilai nasional sebelum dibagi IQR |
| `06_gap_vector_robust.csv` | Vektor gap setelah dibagi IQR; input untuk weighted cosine |
| `07_laporan_iqr.csv` | Q1, Q3, IQR, dan nilai nasional per fitur |

### 3. Pemeriksaan kualitas preprocessing

Script: [`quality_assessment.py`](quality_assessment.py)

Script memeriksa jumlah provinsi dan fitur, missing values, duplikasi nama,
konsistensi daftar provinsi, dan apakah gap robust sama dengan gap raw dibagi
IQR. Script ini adalah quality check dasar; bukan evaluasi model atau validasi
geografis.

Output:

| File | Isi |
|---|---|
| `08_quality_assessment.csv` | Hasil pemeriksaan dengan status PASS/FAIL |
| `09_statistik_gap_robust.csv` | Statistik deskriptif dan skewness gap robust |
| `10_distribusi_gap.csv` | Jumlah provinsi di bawah, sama dengan, atau di atas nilai nasional |

### 4. Entropy Weight Method

Script: [`mining/entropy_weight.py`](mining/entropy_weight.py)

Script menerapkan normalisasi min–max pada sembilan fitur mentah antarprovinsi,
mengubah nilai normalisasi menjadi proporsi per fitur, menghitung entropy
ternormalisasi, lalu menghitung diversifikasi dan bobot:

```text
e(j) = -(1 / ln(n)) * sum_i[p(i,j) * ln(p(i,j))]
diversifikasi(j) = 1 - e(j)
bobot(j) = diversifikasi(j) / sum_k[diversifikasi(k)]
```

Pada perhitungan `p * ln(p)`, nilai proporsi nol diabaikan. Bobot yang lebih
besar menunjukkan fitur lebih bervariasi dalam data yang digunakan untuk
entropy; bobot bukan penilaian sebab-akibat atau kepentingan kebijakan.

Output berada di
[`mining/output/mining/01_entropy_weight/`](mining/output/mining/01_entropy_weight/):

- `minmax_normalized.csv`
- `probability_distribution.csv`
- `entropy_weights.csv`

### 5. Weighted cosine similarity

Script: [`mining/weighted_cosine.py`](mining/weighted_cosine.py)

Script menggunakan vektor gap robust dan bobot entropy untuk menghitung
kemiripan setiap pasangan provinsi:

```text
similarity(a,b) =
  sum_j[w(j) * gap(a,j) * gap(b,j)]
  / (sqrt(sum_j[w(j) * gap(a,j)^2]) *
     sqrt(sum_j[w(j) * gap(b,j)^2]))
```

Nilai dekat `1` menunjukkan arah pola gap yang mirip, nilai dekat `-1`
menunjukkan pola yang berlawanan, dan nilai sekitar `0` menunjukkan kemiripan
arah yang lemah. Cosine terutama membandingkan arah pola, bukan besar total
gap.

Output berada di
[`mining/output/mining/02_weighted_cosine/`](mining/output/mining/02_weighted_cosine/):

- `similarity_matrix.csv`: matriks kemiripan 38 × 38.
- `similarity_pairs.csv`: 703 pasangan provinsi unik yang diurutkan menurut similarity.

### 6. Graf k-NN

Script: [`mining/knn_graph.py`](mining/knn_graph.py)

Parameter yang digunakan saat ini adalah `k = 5`. Setiap provinsi memilih
sampai lima tetangga dengan similarity tertinggi; hanya similarity positif
yang dipakai. Pilihan tetangga digabung menjadi graf tidak berarah dengan
metode **union k-NN**: satu sisi dibuat jika salah satu dari dua provinsi
memilih provinsi lainnya.

Output berada di
[`mining/output/mining/03_knn/`](mining/output/mining/03_knn/):

- `knn_neighbors.csv`: daftar tetangga terarah dan peringkatnya.
- `knn_edges.csv`: sisi graf dan bobot similarity.
- `knn_nodes.csv`: degree dan weighted degree setiap provinsi.
- `knn_graph.graphml`: graf untuk dibuka di perangkat visualisasi graf.

### 7. Deteksi komunitas Louvain

Script: [`mining/louvain_community.py`](mining/louvain_community.py)

Louvain dijalankan pada graf k-NN berbobot positif dengan `resolution = 1` dan
`seed = 42`. Hasil komunitas dapat berubah jika input, parameter, atau seed
diubah. Label komunitas adalah label teknis, bukan kategori tingkat
pendidikan; pemberian nama berdasarkan karakteristik indikator memerlukan
interpretasi tambahan.

Output berada di
[`mining/output/mining/04_louvain/`](mining/output/mining/04_louvain/):

- `louvain_communities.csv`: label komunitas, degree, dan weighted degree per provinsi.
- `louvain_summary.csv`: jumlah anggota, sisi internal, dan ringkasan similarity.
- `louvain_metrics.csv`: metrik hasil, termasuk modularity.
- `louvain_graph.graphml`: graf dengan atribut komunitas.

## Menyiapkan dan menjalankan di Windows

Buka PowerShell pada direktori proyek:

```powershell
cd C:\Users\LITA\Downloads\webmining_UAS
```

Buat virtual environment dan pasang dependensi. `networkx` diperlukan untuk
pembuatan graf dan Louvain:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt networkx
```

Jalankan script secara berurutan karena setiap tahap menggunakan output tahap
sebelumnya:

```powershell
.\.venv\Scripts\python.exe .\preprocess.py
.\.venv\Scripts\python.exe .\gap_vector.py
.\.venv\Scripts\python.exe .\quality_assessment.py
.\.venv\Scripts\python.exe .\mining\entropy_weight.py
.\.venv\Scripts\python.exe .\mining\weighted_cosine.py
.\.venv\Scripts\python.exe .\mining\knn_graph.py
.\.venv\Scripts\python.exe .\mining\louvain_community.py
```

Script membuat direktori output jika diperlukan. Jalankan ulang dari awal
setelah mengganti data sumber agar seluruh hasil tetap berasal dari versi data
yang sama.

## Hasil run saat dokumentasi ini dibuat

Satu kali eksekusi pipeline menghasilkan:

- 38 provinsi dan 9 fitur.
- 703 pasangan similarity unik.
- Graf k-NN dengan 38 node dan 133 sisi.
- 3 komunitas Louvain.
- Modularity sekitar `0.510`.

Angka ini merupakan hasil run saat ini, bukan jaminan bahwa hasil akan sama
jika data atau parameter berubah. Laporan proyek menetapkan target modularity
lebih dari `0.3`; hasil run tersebut memenuhi target itu. Nilai modularity
sendiri tidak cukup untuk menyatakan semua aspek model tervalidasi.

## Yang belum tercakup

Script di atas telah menjalankan inti mining, tetapi beberapa evaluasi yang
direncanakan dalam laporan belum ada di pipeline:

- **Silhouette score** untuk kualitas kelompok dengan jarak yang sesuai dengan
  similarity.
- **Stabilitas top-5** dengan menghilangkan satu fitur bergantian dan
  menghitung Jaccard.
- **Sensitivitas bobot**, misalnya membandingkan bobot entropy, bobot sama
  rata, dan skenario bobot lain.
- **Validasi IPM dan kemiskinan** melalui korelasi pasangan similarity dengan
  selisih nilai indikator tersebut.
- **Validasi geografis**: pemetaan koordinat ibu kota 38 provinsi, perhitungan
  Haversine, uji Mantel berbasis permutasi, irisan tetangga terdekat, serta
  uji sensitivitas tanpa provinsi Papua.
- **Pembanding metode** seperti cosine biasa, jarak Euclidean, dan K-Means.
- **Dashboard interaktif** yang menampilkan peta, heatmap, top-5 kembar, dan
  insight otomatis.

Karena itu, hasil mining sebaiknya dibaca sebagai hasil eksplorasi sampai
evaluasi-evaluasi tersebut dijalankan. Data koordinat di `data/` belum
digunakan oleh pipeline ini.

## Catatan interpretasi

- Data hanya mencakup potret tahun 2025 pada tingkat provinsi, bukan analisis
  tren.
- Jumlah desa yang memiliki fasilitas sekolah bukan ukuran kualitas sekolah,
  jumlah murid, atau jumlah guru.
- Pencilan tetap dipertahankan karena dapat merepresentasikan kesenjangan
  nyata; robust scaling tidak menghapus pencilan.
- Similarity yang tinggi berarti pola gap searah, bukan berarti dua provinsi
  memiliki nilai indikator yang sama atau hubungan sebab-akibat.
- Jarak garis lurus antaribu kota, jika kelak dihitung, tidak mewakili waktu
  atau kemudahan perjalanan antarpulau.
