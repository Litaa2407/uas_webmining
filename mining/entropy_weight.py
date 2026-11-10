
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. KONFIGURASI FOLDER
# ============================================================

# Lokasi file entropy_weight.py
BASE_DIR = Path(__file__).resolve().parent

# Lokasi folder utama webmining_UAS
PROJECT_DIR = BASE_DIR.parent

# Folder hasil preprocessing
PREPROCESSING_DIR = (
    PROJECT_DIR / "output_preprocessing"
)

# Folder hasil mining
MINING_DIR = (
    BASE_DIR / "output" / "mining"
)

# Folder khusus Entropy Weighting
ENTROPY_DIR = (
    MINING_DIR / "01_entropy_weight"
)

# Membuat folder output secara otomatis
ENTROPY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. KONFIGURASI DATASET DAN FITUR
# ============================================================

FILE_DATA = (
    PREPROCESSING_DIR /
    "01_dataset_pendidikan_38_provinsi.csv"
)

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
# 3. MEMBACA DATASET
# ============================================================

def load_dataset():

    print("\n" + "=" * 60)
    print("TAHAP 1: MEMBACA DATASET")
    print("=" * 60)

    print("\nLokasi dataset:")
    print(FILE_DATA)

    # Memeriksa keberadaan dataset
    if not FILE_DATA.exists():

        raise FileNotFoundError(
            "\nDataset tidak ditemukan!\n"
            f"Lokasi: {FILE_DATA}\n"
            "Periksa folder output_preprocessing."
        )

    # Membaca dataset
    df = pd.read_csv(FILE_DATA)

    print("\nDataset berhasil dibaca.")
    print("Jumlah baris:", len(df))
    print("Jumlah kolom:", len(df.columns))

    # Memeriksa kolom Provinsi
    if "Provinsi" not in df.columns:

        raise ValueError(
            "Kolom 'Provinsi' tidak ditemukan."
        )

    # Memeriksa ketersediaan fitur
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Fitur tidak ditemukan: {missing_features}"
        )

    # Memastikan fitur numerik
    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Memeriksa nilai kosong
    if df[FEATURES].isnull().any().any():

        missing_count = df[FEATURES].isnull().sum()

        raise ValueError(
            "Terdapat nilai kosong atau non-numerik:\n"
            f"{missing_count[missing_count > 0]}"
        )

    # Memastikan jumlah data 38 provinsi
    if len(df) != 38:

        raise ValueError(
            f"Jumlah data seharusnya 38 provinsi, "
            f"tetapi ditemukan {len(df)}."
        )

    print("Jumlah provinsi: 38")
    print("Jumlah fitur: 9")
    print("Pemeriksaan dataset: PASS")

    return df


# ============================================================
# 4. NORMALISASI MIN-MAX
# ============================================================

def minmax_normalization(df):

    print("\n" + "=" * 60)
    print("TAHAP 2: NORMALISASI MIN-MAX")
    print("=" * 60)

    # Mengambil sembilan fitur pendidikan
    X = df[FEATURES].copy()

    # Menghitung minimum dan maksimum
    min_values = X.min()
    max_values = X.max()

    range_values = max_values - min_values

    # Menginisialisasi hasil normalisasi
    X_normalized = pd.DataFrame(
        0.0,
        index=X.index,
        columns=FEATURES
    )

    # Normalisasi fitur yang tidak konstan
    non_constant = range_values > 0

    X_normalized.loc[:, non_constant] = (
        X.loc[:, non_constant]
        - min_values[non_constant]
    ) / range_values[non_constant]

    print("Normalisasi Min-Max selesai.")

    print(
        "Nilai minimum:",
        X_normalized.min().min()
    )

    print(
        "Nilai maksimum:",
        X_normalized.max().max()
    )

    # Menambahkan nama provinsi
    normalized_output = pd.concat(
        [
            df[["Provinsi"]].reset_index(drop=True),
            X_normalized.reset_index(drop=True)
        ],
        axis=1
    )

    # Menyimpan hasil normalisasi
    output_file = (
        ENTROPY_DIR / "minmax_normalized.csv"
    )

    normalized_output.to_csv(
        output_file,
        index=False
    )

    print("\nFile berhasil disimpan:")
    print(output_file)

    return X_normalized


# ============================================================
# 5. MENGHITUNG DISTRIBUSI PROBABILITAS
# ============================================================

def calculate_probability(
    X_normalized,
    provinces
):

    print("\n" + "=" * 60)
    print("TAHAP 3: DISTRIBUSI PROBABILITAS")
    print("=" * 60)

    # Menghitung jumlah setiap kolom
    column_sums = X_normalized.sum(axis=0)

    # Menghitung probabilitas
    P = X_normalized.div(
        column_sums.replace(0, np.nan),
        axis=1
    ).fillna(0)

    print("Distribusi probabilitas selesai.")

    # Memeriksa jumlah probabilitas setiap fitur
    print("\nJumlah probabilitas setiap fitur:")

    print(P.sum(axis=0))

    # Menambahkan nama provinsi
    probability_output = P.copy()

    probability_output.insert(
        0,
        "Provinsi",
        provinces.reset_index(drop=True)
    )

    # Menyimpan hasil probabilitas
    output_file = (
        ENTROPY_DIR /
        "probability_distribution.csv"
    )

    probability_output.to_csv(
        output_file,
        index=False
    )

    print("\nFile berhasil disimpan:")
    print(output_file)

    return P


# ============================================================
# 6. MENGHITUNG ENTROPY SETIAP FITUR
# ============================================================

def calculate_entropy(P):

    print("\n" + "=" * 60)
    print("TAHAP 4: PERHITUNGAN ENTROPY")
    print("=" * 60)

    # Jumlah data
    n = len(P)

    if n <= 1:

        raise ValueError(
            "Jumlah data harus lebih dari satu."
        )

    # Konstanta entropy
    k = 1 / np.log(n)

    entropy_values = {}

    for feature in FEATURES:

        probabilities = P[feature].to_numpy()

        # Mengabaikan probabilitas nol
        nonzero_probabilities = probabilities[
            probabilities > 0
        ]

        # Menghitung entropy
        entropy = -k * np.sum(
            nonzero_probabilities *
            np.log(nonzero_probabilities)
        )

        entropy_values[feature] = entropy

    entropy_series = pd.Series(
        entropy_values,
        name="Entropy"
    )

    print("Perhitungan entropy selesai.")

    print("\nNilai entropy setiap fitur:")
    print(entropy_series)

    return entropy_series


# ============================================================
# 7. MENGHITUNG DIVERSIFIKASI
# ============================================================

def calculate_diversification(entropy_series):

    print("\n" + "=" * 60)
    print("TAHAP 5: PERHITUNGAN DIVERSIFIKASI")
    print("=" * 60)

    # Diversifikasi = 1 - entropy
    diversification = 1 - entropy_series

    diversification.name = "Diversification"

    print("Perhitungan diversifikasi selesai.")

    print("\nNilai diversifikasi:")
    print(diversification)

    return diversification


# ============================================================
# 8. MENGHITUNG BOBOT ENTROPY
# ============================================================

def calculate_entropy_weight(
    entropy_series,
    diversification
):

    print("\n" + "=" * 60)
    print("TAHAP 6: PERHITUNGAN BOBOT ENTROPY")
    print("=" * 60)

    total_diversification = diversification.sum()

    if total_diversification <= 0:

        raise ValueError(
            "Total diversifikasi bernilai nol. "
            "Bobot entropy tidak dapat dihitung."
        )

    # Menghitung bobot setiap fitur
    weights = (
        diversification /
        total_diversification
    )

    weights.name = "Bobot"

    # Menggabungkan hasil entropy,
    # diversifikasi, dan bobot
    result = pd.DataFrame({
        "Entropy": entropy_series,
        "Diversification": diversification,
        "Bobot": weights
    })

    result.index.name = "Fitur"

    print("\nHasil bobot entropy:")
    print(result)

    print("\nTotal bobot:")
    print(weights.sum())

    return result


# ============================================================
# 9. VALIDASI HASIL BOBOT
# ============================================================

def validate_weights(result):

    print("\n" + "=" * 60)
    print("TAHAP 7: VALIDASI BOBOT")
    print("=" * 60)

    weights = result["Bobot"]

    # Memeriksa total bobot
    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-9
    ):

        raise ValueError(
            "Total bobot tidak sama dengan 1."
        )

    # Memeriksa bobot negatif
    if (weights < 0).any():

        raise ValueError(
            "Terdapat bobot negatif."
        )

    # Memeriksa rentang entropy
    if (
        (result["Entropy"] < -1e-9).any()
        or
        (result["Entropy"] > 1 + 1e-9).any()
    ):

        raise ValueError(
            "Nilai entropy berada di luar rentang 0-1."
        )

    print("Total bobot = 1: PASS")
    print("Tidak ada bobot negatif: PASS")
    print("Rentang entropy 0-1: PASS")

    print("\nSeluruh validasi berhasil.")


# ============================================================
# 10. MENYIMPAN HASIL ENTROPY WEIGHTING
# ============================================================

def save_entropy_result(result):

    print("\n" + "=" * 60)
    print("TAHAP 8: MENYIMPAN HASIL")
    print("=" * 60)

    output_file = (
        ENTROPY_DIR /
        "entropy_weights.csv"
    )

    result.to_csv(output_file)

    print("Hasil bobot entropy berhasil disimpan.")

    print("\nLokasi file:")
    print(output_file)


# ============================================================
# 11. PROGRAM UTAMA
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("ENTROPY WEIGHTING")
    print("WEB MINING - PENDIDIKAN ANTARPROVINSI")
    print("=" * 60)

    print("\nLokasi proyek:")
    print(PROJECT_DIR)

    print("\nFolder preprocessing:")
    print(PREPROCESSING_DIR)

    print("\nFolder output mining:")
    print(MINING_DIR)

    print("\nFolder hasil entropy:")
    print(ENTROPY_DIR)

    # Tahap 1: Membaca dataset
    df = load_dataset()

    provinces = df["Provinsi"].copy()

    # Tahap 2: Normalisasi Min-Max
    X_normalized = minmax_normalization(df)

    # Tahap 3: Distribusi probabilitas
    P = calculate_probability(
        X_normalized,
        provinces
    )

    # Tahap 4: Menghitung entropy
    entropy_series = calculate_entropy(P)

    # Tahap 5: Menghitung diversifikasi
    diversification = calculate_diversification(
        entropy_series
    )

    # Tahap 6: Menghitung bobot entropy
    result = calculate_entropy_weight(
        entropy_series,
        diversification
    )

    # Tahap 7: Validasi bobot
    validate_weights(result)

    # Tahap 8: Menyimpan hasil
    save_entropy_result(result)

    print("\n")

    print("=" * 60)
    print("PROSES ENTROPY WEIGHTING SELESAI")
    print("=" * 60)

    print("\nFile hasil tersimpan di:")
    print(ENTROPY_DIR)

    print("\nFile yang dihasilkan:")
    print("1. minmax_normalized.csv")
    print("2. probability_distribution.csv")
    print("3. entropy_weights.csv")


# ============================================================
# 12. EKSEKUSI PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
