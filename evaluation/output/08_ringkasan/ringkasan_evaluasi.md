# Ringkasan Evaluasi

## Kriteria keberhasilan (Laporan 2.3)

| Aspek                            | Kriteria                                                       |     Nilai |   p_value | Status                |
|:---------------------------------|:---------------------------------------------------------------|----------:|----------:|:----------------------|
| Validitas                        | Spearman(kemiripan, selisih IPM) negatif & p < 0,05            |   -0.2563 |    0.0014 | LOLOS                 |
| Validitas                        | Spearman(kemiripan, selisih kemiskinan) negatif & p < 0,05     |   -0.0042 |    0.9446 | TIDAK LOLOS           |
| Kualitas kelompok                | Modularity Louvain > 0,3                                       |    0.5163 |  nan      | LOLOS                 |
| Kualitas kelompok                | Silhouette > 0                                                 |    0.4519 |  nan      | LOLOS                 |
| Stabilitas                       | Rata-rata Jaccard top-5 leave-one-feature-out > 0,6            |    0.7994 |  nan      | LOLOS                 |
| Stabilitas                       | Skenario terburuk: tanpa Rasio fasilitas PT/SD                 |    0.5251 |  nan      | DI BAWAH 0,6          |
| Kepekaan bobot                   | Jaccard top-5 bobot sama rata vs entropi                       |    0.5787 |  nan      | Dilaporkan            |
| Kemasukakalan                    | Sim rata-rata Jawa/kota besar > sim ke Papua Pegunungan/Tengah |    0.3934 |  nan      | LOLOS                 |
| Validasi geografis               | Mantel Spearman(kemiripan, jarak)                              |   -0.0605 |    0.2044 | Dilaporkan apa adanya |
| Validasi geografis               | Irisan top-5 kembar & top-5 terdekat (acak 13,5%)              |    0.2526 |    0.0002 | Dilaporkan apa adanya |
| Validasi geografis               | Median jarak ke 5 kembar (km)                                  | 1103.61   |    0.0006 | Dilaporkan apa adanya |
| Validasi geografis (tanpa Papua) | Mantel tanpa 6 provinsi Papua                                  |   -0.048  |    0.2858 | Dilaporkan apa adanya |
| Validasi geografis (tanpa Papua) | Irisan tanpa Papua                                             |    0.25   |    0.0074 | Dilaporkan apa adanya |

## Jawaban pertanyaan analitik (Laporan 2.2)

**1. Provinsi mana yang paling mirip dengan provinsi tertentu?**  
Lihat `07_kemasukakalan/top5_kembar_semua_provinsi.csv`.

**2. Apakah provinsi yang berdekatan otomatis mirip?**  
Mantel Spearman = -0,060 (p = 0,2044); tanpa Papua -0,048 (p = 0,2858). Irisan kembar dengan tetangga terdekat 25,3% vs harapan acak 13,5% (p = 0,0002). Median jarak ke kembar 1103,6 km (p = 0,0006). Bacaan: hubungan jarak secara global sangat lemah; efek tetangga lokal ada.

**3. Dimensi apa yang paling membedakan antarprovinsi?**

| Fitur                  |   Bobot_Entropi |   Variansi_Gap_Berbobot |   Kruskal_H_Antar_Komunitas |   p_Kruskal |   Porsi_Variansi |
|:-----------------------|----------------:|------------------------:|----------------------------:|------------:|-----------------:|
| Rasio fasilitas PT/SD  |          0.423  |                  4.905  |                     25.3627 |      0      |           0.661  |
| APS 7-12               |          0.0257 |                  1.1669 |                     17.3709 |      0.0006 |           0.1573 |
| Rasio fasilitas SMK/SD |          0.2364 |                  0.8036 |                     22.2445 |      0.0001 |           0.1083 |
| Rasio fasilitas SMU/SD |          0.1184 |                  0.19   |                     21.6446 |      0.0001 |           0.0256 |
| APS 13-15              |          0.0255 |                  0.1875 |                     18.8483 |      0.0003 |           0.0253 |
| Rasio fasilitas SMP/SD |          0.063  |                  0.0555 |                     21.2208 |      0.0001 |           0.0075 |
| APS 16-18              |          0.0296 |                  0.0487 |                     18.5725 |      0.0003 |           0.0066 |
| APS 19-23              |          0.0533 |                  0.037  |                     16.1208 |      0.0011 |           0.005  |
| Rata-rata lama sekolah |          0.0251 |                  0.0264 |                     19.8436 |      0.0002 |           0.0036 |

Tiga fitur teratas menyumbang 92,7% variansi vektor gap berbobot.

**4. Apakah kemiripan sejalan dengan kedekatan IPM dan kemiskinan?**  
IPM: r = -0,256, p = 0,0014 (LOLOS). Kemiskinan: r = -0,004, p = 0,9446 (TIDAK LOLOS).

## Kepekaan bobot

| Skema_Bobot                         |   Jaccard_Top5_vs_Utama |   Spearman_Kemiripan_vs_Utama |   ARI_Komunitas_vs_Utama |   Jumlah_Komunitas |   Modularity | Bobot_Terbesar                 |   Porsi_3_Rasio_Fasilitas_Atas |
|:------------------------------------|------------------------:|------------------------------:|-------------------------:|-------------------:|-------------:|:-------------------------------|-------------------------------:|
| Entropi (sebelum dikoreksi) [utama] |                  1      |                        1      |                   1      |                  4 |       0.5163 | Rasio fasilitas PT/SD (0.423)  |                          0.778 |
| Entropi terkoreksi redundansi       |                  0.9323 |                        0.9995 |                   0.7133 |                  3 |       0.5331 | Rasio fasilitas PT/SD (0.456)  |                          0.79  |
| Sama rata                           |                  0.5787 |                        0.8956 |                   0.5848 |                  4 |       0.5572 | Rata-rata lama sekolah (0.111) |                          0.333 |

## Pembanding metode

| Metode                                   |   Spearman_vs_Selisih_IPM |   p_Mantel_999 |   Jaccard_Top5_vs_Utama |   Jumlah_Kelompok | Ukuran_Kelompok   |   Modularity_Louvain |   ARI_vs_Louvain_Utama |
|:-----------------------------------------|--------------------------:|---------------:|------------------------:|------------------:|:------------------|---------------------:|-----------------------:|
| Weighted cosine (gap + entropi) [utama]  |                   -0.2563 |          0.001 |                  1      |                 4 | [12, 11, 11, 4]   |               0.5163 |                 1      |
| Cosine biasa (gap, tanpa bobot)          |                   -0.2187 |          0.001 |                  0.5787 |                 4 | [14, 11, 7, 6]    |               0.5572 |                 0.5848 |
| Jarak Euclidean berbobot (gap + entropi) |                   -0.6016 |          0.001 |                  0.4764 |                 4 | [12, 10, 8, 8]    |               0.4965 |                 0.5162 |
| K-Means k=2 (silhouette terbaik 0.854)   |                  nan      |        nan     |                nan      |                 2 | [37, 1]           |             nan      |                -0.0056 |
| K-Means k=4 (sama dengan Louvain)        |                  nan      |        nan     |                nan      |                 4 | [29, 6, 2, 1]     |             nan      |                 0.144  |

## Komunitas Louvain

- **Komunitas 1** (11 provinsi): Fasilitas jenjang atas di bawah nasional; fasilitas SMP di bawah nasional; capaian lama sekolah di atas nasional. Anggota: ACEH; BENGKULU; KALIMANTAN SELATAN; KALIMANTAN UTARA; MALUKU; PAPUA; PAPUA BARAT; PAPUA BARAT DAYA; SULAWESI TENGGARA; SULAWESI UTARA; SUMATERA UTARA
- **Komunitas 2** (12 provinsi): Fasilitas jenjang atas di atas nasional; fasilitas SMP di atas nasional; akses menengah di atas nasional. Anggota: BALI; BANTEN; DI YOGYAKARTA; DKI JAKARTA; JAWA BARAT; JAWA TIMUR; KALIMANTAN TIMUR; KEPULAUAN RIAU; NUSA TENGGARA BARAT; RIAU; SULAWESI SELATAN; SUMATERA BARAT
- **Komunitas 3** (11 provinsi): Fasilitas jenjang atas di bawah nasional; akses pendidikan tinggi di bawah nasional; akses menengah di bawah nasional. Anggota: GORONTALO; JAMBI; JAWA TENGAH; KALIMANTAN BARAT; KALIMANTAN TENGAH; KEPULAUAN BANGKA BELITUNG; LAMPUNG; MALUKU UTARA; NUSA TENGGARA TIMUR; SULAWESI TENGAH; SUMATERA SELATAN
- **Komunitas 4** (4 provinsi): Akses dasar jauh di bawah nasional; akses menengah jauh di bawah nasional; capaian lama sekolah jauh di bawah nasional. Anggota: PAPUA PEGUNUNGAN; PAPUA SELATAN; PAPUA TENGAH; SULAWESI BARAT