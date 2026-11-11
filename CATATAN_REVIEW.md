# Catatan Review Program (Preprocessing, Mining, Evaluation)

Repo yang dicek: `Litaa2407/uas_webmining` (commit `176b04e`, "pre processing & mining").
Cara cek: semua script dijalankan ulang dari nol, lalu tiap angka dibandingkan dengan implementasi pembanding yang dibuat terpisah dan dengan angka di Laporan Minggu 10.

## 1. Hasil cek preprocessing: BENAR

| Bagian | Status | Bukti |
|---|---|---|
| Baca 3 tabel BPS + IPM + kemiskinan | Benar | 38 provinsi, tidak ada nilai kosong |
| Standarisasi nama (KEP. → KEPULAUAN, huruf besar) | Benar | Merge `one_to_one` lolos |
| Pisahkan baris INDONESIA | Benar | Nilai nasional sama dengan laporan (RLS 9,41; APS 99,23/96,30/77,86/29,13) |
| 4 rasio fasilitas | Benar | Rasio nasional dari baris INDONESIA (0,54; 0,25; 0,15; 0,04) |
| Gap / IQR | Benar | Selisih dengan pembanding < 1e-13 |
| Quality assessment | Benar | Semua PASS |

Yang **kurang**:
- **Koordinat ibu kota belum diproses** (Laporan 3.4, 4.6, Lampiran A). Tanpa ini validasi geografis (pertanyaan analitik 2) tidak bisa dijalankan. → ditambah `preprocess_koordinat.py`.
- `data/processed/` berisi 3 file dari script lain yang tidak ada di repo (nama kolom bahasa Inggris). Isinya sama, tetapi membingungkan. **Saran: hapus folder ini.**
- `requirements.txt` hanya berisi pandas dan numpy, padahal mining memakai networkx. → diperbarui.

## 2. Hasil cek mining: BENAR, dengan 1 masalah dan 2 bagian yang belum ada

| Bagian | Status | Bukti |
|---|---|---|
| Entropy weight (min-max → p → e → w) | Benar | Selisih bobot < 1e-15 |
| Weighted cosine 38 × 38 | Benar | Selisih < 1e-14 |
| Graf k-NN (k = 5, sim > 0) | Benar | 133 edge |
| Louvain | **Masalah** | Lihat bawah |

**Masalah Louvain:** Louvain itu acak (urutan node diacak). Dengan satu seed (42), hasilnya **3 komunitas, modularity 0,5102**. Hasil ini hanya muncul di 9 dari 100 seed dan bukan yang terbaik. Modularity tertinggi adalah **0,5163 dengan 4 komunitas**: Papua Pegunungan, Papua Tengah, Papua Selatan, dan Sulawesi Barat terpisah sebagai komunitas sendiri.
→ `mining/louvain_community.py` diubah: dijalankan 100 seed, lalu dipilih modularity tertinggi (`N_RUNS`). Semua run disimpan di `louvain_seed_runs.csv`.

Yang **belum ada**:
- **Label/profil komunitas** (Laporan 5.6 langkah 4). → ditambah `mining/community_profile.py`.
- **Jarak Haversine** (Laporan 5.2 langkah 7). → ditambah `mining/haversine_distance.py`. Statistiknya sama persis dengan laporan 4.6: min 77,5 km, maks 5.236,3 km, rata-rata 1.713,0 km.

## 3. Evaluation (baru, folder `evaluation/`)

Isinya mengikuti tabel 5.8 dan kriteria 2.3 di laporan. Setiap script dimulai dengan `check_consistency()`, yang memastikan rumus evaluasi sama persis dengan hasil mining.

## 4. Hasil evaluasi yang perlu dibahas jujur di laporan

| Kriteria | Nilai | Status |
|---|---|---|
| Validitas IPM | r = −0,256, p = 0,001 | Lolos, tapi lemah |
| Validitas kemiskinan | r = −0,004, p = 0,94 | **Tidak lolos** |
| Modularity | 0,516 | Lolos |
| Silhouette | 0,452 | Lolos |
| LOFO rata-rata | 0,799 | Lolos |
| LOFO tanpa PT/SD | 0,525 | **Di bawah 0,6** |
| Bobot sama rata vs entropi | Jaccard 0,58 | **Hasil peka terhadap bobot** |
| Kemasukakalan Jawa vs Papua | 0,39 vs negatif | Lolos |
| Mantel kemiripan vs jarak | r = −0,06, p = 0,20 | Tidak ada hubungan global |
| Irisan kembar ∩ tetangga | 25,3% vs acak 13,5%, p = 0,0002 | Ada efek lokal |

Poin penting:
1. **3 rasio fasilitas jenjang atas memegang 78% bobot entropi** (PT/SD saja 0,42), sedangkan RLS dan APS masing-masing sekitar 0,03. Fitur PT/SD sendiri menyumbang 66% variansi. Jadi "kembar pendidikan" sebagian besar adalah kembar *pola fasilitas jenjang atas*. Jawaban pertanyaan analitik 3 harus menyebut ini.
2. **Louvain tidak stabil antara 3 dan 4 komunitas** (46 vs 54 dari 100 seed; modularity 0,5156 vs 0,5163, selisih sangat kecil). Yang goyah adalah apakah kelompok Papua ekstrem + Sulawesi Barat berdiri sendiri. Provinsi perbatasan: Sulawesi Tenggara, Bengkulu, Sulawesi Utara.
3. **Cosine hanya menangkap arah.** Euclidean berbobot berkorelasi −0,60 dengan selisih IPM, sedangkan cosine hanya −0,26.
4. **K-Means gagal**: k terbaik = 2 dengan ukuran [37, 1] (DKI sendirian). Ini mendukung pilihan Louvain.
5. **Pertanyaan 2**: kedekatan tidak otomatis membuat mirip (Mantel lemah), tetapi ada efek tetangga lokal (irisan dan jarak kembar signifikan).

## 5. Daftar file yang ditambah / diubah

| File | Status |
|---|---|
| `preprocess_koordinat.py` | BARU |
| `mining/louvain_community.py` | DIUBAH (multi-seed; bagian konfigurasi + fungsi `detect_communities`) |
| `mining/community_profile.py` | BARU |
| `mining/haversine_distance.py` | BARU |
| `evaluation/` (9 file) | BARU |
| `run_all.py` | BARU |
| `requirements.txt` | DIUBAH |
| `README.md`, `CATATAN_REVIEW.md` | BARU |
| `data/processed/` | DIHAPUS (sisa script lain) |

File lain (`preprocess.py`, `gap_vector.py`, `quality_assessment.py`, `entropy_weight.py`, `weighted_cosine.py`, `knn_graph.py`) **tidak diubah**.