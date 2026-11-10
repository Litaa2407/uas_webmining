
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. KONFIGURASI FOLDER
# ============================================================

# Lokasi file weighted_cosine.py
BASE_DIR = Path(__file__).resolve().parent

# Folder utama proyek webmining_UAS
PROJECT_DIR = BASE_DIR.parent

# Folder preprocessing
PREPROCESSING_DIR = (
    PROJECT_DIR / "output_preprocessing"
)

# Folder hasil Entropy Weighting
ENTROPY_DIR = (
    BASE_DIR / "output" / "mining" / "01_entropy_weight"
)

# Folder hasil Weighted Cosine Similarity
COSINE_DIR = (
    BASE_DIR / "output" / "mining" / "02_weighted_cosine"
)

# Membuat folder output secara otomatis
COSINE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. KONFIGURASI FILE INPUT
# ============================================================

FILE_GAP = (
    PREPROCESSING_DIR /
    "06_gap_vector_robust.csv"
)

FILE_WEIGHT = (
    ENTROPY_DIR /
    "entropy_weights.csv"
)


# ============================================================
# 3. KONFIGURASI FITUR
# ============================================================

FEATURES = [
    "RLS",
    "APS_7_12",
    "APS_13_15",
    "APS_16_18",
    "APS_19_23",
    "Rasio_SMP_SD",
    "Rasio_SMU_SD",
    "Rasio_SMK_SD",
    "Rasio_PT_SD"
]


# ============================================================
# 4. MEMBACA DATA GAP ROBUST
# ============================================================

def load_gap_data():

    print("\n" + "=" * 60)
    print("TAHAP 1: MEMBACA DATA GAP ROBUST")
    print("=" * 60)

    print("\nLokasi file:")
    print(FILE_GAP)

    if not FILE_GAP.exists():

        raise FileNotFoundError(
            f"File gap robust tidak ditemukan:\n{FILE_GAP}"
        )

    df = pd.read_csv(FILE_GAP)

    print("\nDataset berhasil dibaca.")
    print("Jumlah baris:", len(df))
    print("Jumlah kolom:", len(df.columns))

    # Memeriksa kolom Provinsi
    if "Provinsi" not in df.columns:

        raise ValueError(
            "Kolom 'Provinsi' tidak ditemukan."
        )

    # Memeriksa fitur
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Fitur tidak ditemukan: {missing_features}"
        )

    # Memastikan 38 provinsi
    if len(df) != 38:

        raise ValueError(
            f"Jumlah provinsi seharusnya 38, "
            f"tetapi ditemukan {len(df)}."
        )

    # Memeriksa duplikasi provinsi
    if df["Provinsi"].duplicated().any():

        raise ValueError(
            "Terdapat nama provinsi yang duplikat."
        )

    # Memastikan seluruh fitur numerik
    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if df[FEATURES].isnull().any().any():

        raise ValueError(
            "Terdapat nilai kosong atau non-numerik "
            "pada gap robust."
        )

    if not np.isfinite(df[FEATURES].to_numpy()).all():

        raise ValueError(
            "Terdapat nilai infinity pada gap robust."
        )

    print("Jumlah provinsi: 38")
    print("Jumlah fitur: 9")
    print("Pemeriksaan data gap robust: PASS")

    return df


# ============================================================
# 5. MEMBACA BOBOT ENTROPY
# ============================================================

def load_entropy_weights():

    print("\n" + "=" * 60)
    print("TAHAP 2: MEMBACA BOBOT ENTROPY")
    print("=" * 60)

    print("\nLokasi file:")
    print(FILE_WEIGHT)

    if not FILE_WEIGHT.exists():

        raise FileNotFoundError(
            f"File bobot entropy tidak ditemukan:\n{FILE_WEIGHT}"
        )

    weights_df = pd.read_csv(FILE_WEIGHT)

    print("\nFile bobot berhasil dibaca.")

    # Memeriksa kolom yang dibutuhkan
    required_columns = [
        "Fitur",
        "Bobot"
    ]

    for column in required_columns:

        if column not in weights_df.columns:

            raise ValueError(
                f"Kolom '{column}' tidak ditemukan "
                "pada file entropy_weights.csv."
            )

    # Mengambil nama fitur dan bobot
    weights_df["Bobot"] = pd.to_numeric(
        weights_df["Bobot"],
        errors="coerce"
    )

    if weights_df["Bobot"].isnull().any():

        raise ValueError(
            "Terdapat bobot kosong atau non-numerik."
        )

    # Memeriksa fitur yang tersedia
    weight_features = weights_df["Fitur"].tolist()

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in weight_features
    ]

    if missing_features:

        raise ValueError(
            f"Fitur berikut tidak memiliki bobot: "
            f"{missing_features}"
        )

    # Menyusun bobot sesuai urutan FEATURES
    weights_df = (
        weights_df
        .set_index("Fitur")
        .loc[FEATURES]
    )

    weights = weights_df["Bobot"].copy()

    # Validasi jumlah bobot
    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-9
    ):

        raise ValueError(
            "Total bobot entropy tidak sama dengan 1."
        )

    if (weights < 0).any():

        raise ValueError(
            "Ditemukan bobot negatif."
        )

    print("\nBobot Entropy yang digunakan:")

    print(weights)

    print("\nTotal bobot:", weights.sum())

    print("Pemeriksaan bobot: PASS")

    return weights


# ============================================================
# 6. MENYIAPKAN MATRIKS GAP DAN BOBOT
# ============================================================

def prepare_data(df, weights):

    print("\n" + "=" * 60)
    print("TAHAP 3: MENYIAPKAN DATA PERHITUNGAN")
    print("=" * 60)

    # Mengambil nama provinsi
    provinces = df["Provinsi"].tolist()

    # Mengambil matriks gap robust
    X = df[FEATURES].to_numpy(dtype=float)

    # Mengambil bobot sesuai urutan fitur
    W = weights.to_numpy(dtype=float)

    print("Matriks gap robust:", X.shape)
    print("Vektor bobot:", W.shape)

    print("Persiapan data selesai.")

    return provinces, X, W


# ============================================================
# 7. MENGHITUNG WEIGHTED COSINE SIMILARITY
# ============================================================

def calculate_weighted_cosine(
    provinces,
    X,
    W
):

    print("\n" + "=" * 60)
    print("TAHAP 4: WEIGHTED COSINE SIMILARITY")
    print("=" * 60)

    # Mengalikan setiap fitur dengan bobotnya
    weighted_X = X * W

    # Menghitung perkalian matriks untuk pembilang
    numerator = weighted_X @ X.T

    # Menghitung norma berbobot setiap provinsi
    squared_norms = np.sum(
        (X ** 2) * W,
        axis=1
    )

    norms = np.sqrt(squared_norms)

    # Menghitung penyebut
    denominator = np.outer(
        norms,
        norms
    )

    # Menginisialisasi matriks similarity
    similarity_matrix = np.zeros(
        (len(provinces), len(provinces)),
        dtype=float
    )

    # Menghitung similarity jika penyebut tidak nol
    np.divide(
        numerator,
        denominator,
        out=similarity_matrix,
        where=denominator > 0
    )

    # Membatasi nilai akibat error floating point
    similarity_matrix = np.clip(
        similarity_matrix,
        -1.0,
        1.0
    )

    # Mengatur diagonal:
    # similarity suatu provinsi dengan dirinya sendiri = 1
    np.fill_diagonal(
        similarity_matrix,
        1.0
    )

    # Memastikan simetris
    similarity_matrix = (
        similarity_matrix +
        similarity_matrix.T
    ) / 2

    print("Perhitungan Weighted Cosine selesai.")

    print("\nUkuran matriks:")
    print(similarity_matrix.shape)

    print("\nNilai similarity minimum:")
    print(similarity_matrix.min())

    print("\nNilai similarity maksimum:")
    print(similarity_matrix.max())

    return similarity_matrix


# ============================================================
# 8. MEMBUAT DATAFRAME MATRIKS SIMILARITY
# ============================================================

def create_similarity_dataframe(
    provinces,
    similarity_matrix
):

    print("\n" + "=" * 60)
    print("TAHAP 5: MEMBUAT MATRIKS SIMILARITY")
    print("=" * 60)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=provinces,
        columns=provinces
    )

    similarity_df.index.name = "Provinsi"

    print("Matriks similarity berhasil dibuat.")

    return similarity_df


# ============================================================
# 9. MEMBUAT DAFTAR PASANGAN PROVINSI
# ============================================================

def create_similarity_pairs(
    provinces,
    similarity_matrix
):

    print("\n" + "=" * 60)
    print("TAHAP 6: MEMBUAT DAFTAR PASANGAN PROVINSI")
    print("=" * 60)

    pairs = []

    n = len(provinces)

    # Hanya mengambil pasangan unik
    for i in range(n):

        for j in range(i + 1, n):

            pairs.append({
                "Provinsi_A": provinces[i],
                "Provinsi_B": provinces[j],
                "Similarity": similarity_matrix[i, j]
            })

    pairs_df = pd.DataFrame(pairs)

    # Mengurutkan similarity tertinggi
    pairs_df = pairs_df.sort_values(
        by="Similarity",
        ascending=False
    ).reset_index(drop=True)

    print("Jumlah pasangan provinsi:", len(pairs_df))

    print("\nLima pasangan dengan similarity tertinggi:")

    print(pairs_df.head(5).to_string(index=False))

    return pairs_df


# ============================================================
# 10. MENYIMPAN HASIL WEIGHTED COSINE
# ============================================================

def save_results(
    similarity_df,
    pairs_df
):

    print("\n" + "=" * 60)
    print("TAHAP 7: MENYIMPAN HASIL")
    print("=" * 60)

    # File 1: Matriks similarity
    matrix_file = (
        COSINE_DIR /
        "similarity_matrix.csv"
    )

    similarity_df.to_csv(
        matrix_file
    )

    print("\nMatriks similarity disimpan:")
    print(matrix_file)

    # File 2: Daftar pasangan provinsi
    pairs_file = (
        COSINE_DIR /
        "similarity_pairs.csv"
    )

    pairs_df.to_csv(
        pairs_file,
        index=False
    )

    print("\nDaftar pasangan similarity disimpan:")
    print(pairs_file)


# ============================================================
# 11. PROGRAM UTAMA
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("WEIGHTED COSINE SIMILARITY")
    print("WEB MINING - PENDIDIKAN ANTARPROVINSI")
    print("=" * 60)

    print("\nFolder preprocessing:")
    print(PREPROCESSING_DIR)

    print("\nFolder entropy:")
    print(ENTROPY_DIR)

    print("\nFolder output cosine:")
    print(COSINE_DIR)

    # Tahap 1: Membaca gap robust
    df = load_gap_data()

    # Tahap 2: Membaca bobot entropy
    weights = load_entropy_weights()

    # Tahap 3: Menyiapkan matriks perhitungan
    provinces, X, W = prepare_data(
        df,
        weights
    )

    # Tahap 4: Menghitung Weighted Cosine
    similarity_matrix = calculate_weighted_cosine(
        provinces,
        X,
        W
    )

    # Tahap 5: Membuat dataframe matriks
    similarity_df = create_similarity_dataframe(
        provinces,
        similarity_matrix
    )

    # Tahap 6: Membuat daftar pasangan
    pairs_df = create_similarity_pairs(
        provinces,
        similarity_matrix
    )

    # Tahap 7: Menyimpan hasil
    save_results(
        similarity_df,
        pairs_df
    )

    print("\n")

    print("=" * 60)
    print("WEIGHTED COSINE SIMILARITY SELESAI")
    print("=" * 60)

    print("\nJumlah provinsi:", len(provinces))

    print(
        "Jumlah pasangan unik:",
        len(pairs_df)
    )

    print("\nFile hasil tersimpan di:")
    print(COSINE_DIR)

    print("\nFile yang dihasilkan:")
    print("1. similarity_matrix.csv")
    print("2. similarity_pairs.csv")


# ============================================================
# 12. EKSEKUSI PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
