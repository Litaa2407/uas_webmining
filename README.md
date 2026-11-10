# UAS Web Mining: Daerah Kembar Pendidikan

Entropy-Weighted Cosine Similarity pada Vektor Gap dan Deteksi Komunitas Louvain, 38 provinsi, data BPS 2025.

Alur program: **Preprocessing → Mining → Evaluation**

| Tahap | Script | Output | Laporan |
|---|---|---|---|
| Preprocessing | `preprocess.py` | `output_preprocessing/01-04` | 4.3, 5.2 (1-2) |
| | `gap_vector.py` | `output_preprocessing/05-07` | 5.3 |
| | `quality_assessment.py` | `output_preprocessing/08-10` | 4.3 |
| | `preprocess_koordinat.py` | `output_preprocessing/11_koordinat_ibu_kota.csv` | 3.4, 4.6, Lampiran A |
| Mining | `mining/entropy_weight.py` | `mining/output/mining/01_entropy_weight` | 5.4 |
| | `mining/weighted_cosine.py` | `.../02_weighted_cosine` | 5.5 |
| | `mining/knn_graph.py` | `.../03_knn` | 5.6 (1-2) |
| | `mining/louvain_community.py` | `.../04_louvain` | 5.6 (3) |
| | `mining/community_profile.py` | `.../06_profil_komunitas` | 5.6 (4) |
| | `mining/haversine_distance.py` | `.../05_jarak_geografis` | 5.2 (7), 5.7 |
| Evaluation | `evaluation/validitas_eksternal.py` | `evaluation/output/01_...` | 2.3, 5.8 (IPM & kemiskinan) |
| | `evaluation/validasi_geografis.py` | `.../02_...` | 5.7 (Mantel, irisan, tanpa Papua) |
| | `evaluation/kualitas_komunitas.py` | `.../03_...` | 2.3 (modularity, silhouette) |
| | `evaluation/stabilitas_fitur.py` | `.../04_...` | 2.3 (leave-one-feature-out) |
| | `evaluation/kepekaan_bobot.py` | `.../05_...` | 5.8 (kepekaan bobot) |
| | `evaluation/pembanding_metode.py` | `.../06_...` | 5.8 (cosine biasa, Euclidean, K-Means) |
| | `evaluation/kemasukakalan.py` | `.../07_...` | 2.3 (kemasukakalan) |
| | `evaluation/ringkasan_evaluasi.py` | `.../08_ringkasan` | 2.2 & 2.3 (tabel kriteria + jawaban 4 pertanyaan) |

`run_all.py` menjalankan semuanya berurutan.

## Cara menjalankan

### 1. Siapkan Python (3.10 atau lebih baru)

```bash
python3 --version
```

### 2. Masuk ke folder proyek

```bash
cd ~/Documents/uas_webmining
```

### 3. Buat virtual environment dan pasang library (sekali saja)

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Jalankan seluruh tahap

```bash
python run_all.py
```

Butuh sekitar 20-30 detik. Setiap script ditandai `[OK]`, dan log lengkapnya ada di folder `logs/`.

Menjalankan satu tahap saja:

```bash
python run_all.py preprocessing
python run_all.py mining
python run_all.py evaluation
```

Menjalankan satu script (misalnya saat presentasi):

```bash
python mining/louvain_community.py
python evaluation/validasi_geografis.py
```

Urutan harus dijaga: preprocessing → mining → evaluation.

### 5. Lihat hasil

- `evaluation/output/08_ringkasan/ringkasan_evaluasi.md`: tabel kriteria keberhasilan dan jawaban 4 pertanyaan analitik
- `evaluation/output/08_ringkasan/kriteria_keberhasilan.csv`: tabel siap tempel ke laporan
- `evaluation/output/*/*.png`: grafik

## Parameter penting

| Parameter | Nilai | Lokasi |
|---|---|---|
| k tetangga | 5 | `mining/knn_graph.py`, `evaluation/eval_utils.py` |
| Seed awal Louvain | 42 | `mining/louvain_community.py` |
| Jumlah run Louvain | 100 (dipilih modularity tertinggi) | `mining/louvain_community.py` (`N_RUNS`) |
| Permutasi uji Mantel | 9.999 | `evaluation/eval_utils.py` |

## Masalah umum

| Gejala | Solusi |
|---|---|
| `ModuleNotFoundError` | Aktifkan `.venv` lalu `pip install -r requirements.txt` |
| `FileNotFoundError ... output_preprocessing/...` | Jalankan tahap sebelumnya dulu (`python run_all.py`) |
| `Cek konsistensi ... FAIL` | Hasil mining sudah usang. Jalankan ulang `python run_all.py mining` |