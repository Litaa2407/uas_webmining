
# ============================================================
# WEB MINING
# PREPROCESSING - GAP VECTOR & ROBUST SCALING
# Dashboard Daerah Kembar Pendidikan Indonesia
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output_preprocessing"

FILE_DATA = OUTPUT_DIR / "01_dataset_pendidikan_38_provinsi.csv"
FILE_NASIONAL = OUTPUT_DIR / "03_referensi_nasional.csv"

print("=" * 65)
print("GAP VECTOR CALCULATION & ROBUST SCALING")
print("=" * 65)


# ============================================================
# 2. MEMBACA DATASET
# ============================================================

print("\n[1] Membaca dataset...")

if not FILE_DATA.exists():
    raise FileNotFoundError(FILE_DATA)

if not FILE_NASIONAL.exists():
    raise FileNotFoundError(FILE_NASIONAL)

data = pd.read_csv(FILE_DATA)
nasional = pd.read_csv(FILE_NASIONAL)

print("Dataset pendidikan:", data.shape)
print("Referensi nasional:", nasional.shape)


# ============================================================
# 3. MENENTUKAN 9 FITUR
# ============================================================

feature_columns = [
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

# Validasi kolom
missing_features = [
    col for col in feature_columns
    if col not in data.columns
]

if missing_features:
    raise ValueError(
        f"Fitur tidak ditemukan: {missing_features}"
    )

if len(data) != 38:
    raise ValueError(
        f"Dataset harus memiliki 38 provinsi, ditemukan {len(data)}"
    )

if data[feature_columns].isnull().any().any():
    raise ValueError(
        "Terdapat missing value pada dataset pendidikan."
    )

print("9 fitur berhasil diidentifikasi.")


# ============================================================
# 4. MENGAMBIL NILAI NASIONAL
# ============================================================

print("\n[2] Mengambil nilai nasional...")

nasional_dict = dict(
    zip(
        nasional["Fitur"],
        nasional["Nilai_Nasional"]
    )
)

for feature in feature_columns:

    if feature not in nasional_dict:
        raise ValueError(
            f"Referensi nasional untuk {feature} tidak ditemukan."
        )

print("Referensi nasional berhasil diidentifikasi.")


# ============================================================
# 5. MENGHITUNG GAP VALUE
# ============================================================

print("\n[3] Menghitung gap setiap indikator...")

gap_raw = pd.DataFrame()

gap_raw["Provinsi"] = data["Provinsi"]

for feature in feature_columns:

    nilai_nasional = nasional_dict[feature]

    gap_raw[feature] = (
        data[feature] - nilai_nasional
    )

print("Gap value berhasil dihitung.")


# ============================================================
# 6. MENGHITUNG INTERQUARTILE RANGE (IQR)
# ============================================================

print("\n[4] Menghitung IQR...")

iqr_values = {}

for feature in feature_columns:

    Q1 = data[feature].quantile(0.25)

    Q3 = data[feature].quantile(0.75)

    IQR = Q3 - Q1

    if IQR <= 0:
        raise ValueError(
            f"IQR fitur {feature} bernilai nol atau negatif."
        )

    iqr_values[feature] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR
    }

    print(
        f"{feature}: Q1={Q1:.4f}, "
        f"Q3={Q3:.4f}, IQR={IQR:.4f}"
    )


# ============================================================
# 7. ROBUST SCALING PADA GAP VECTOR
# ============================================================

print("\n[5] Melakukan Robust Scaling...")

gap_robust = pd.DataFrame()

gap_robust["Provinsi"] = data["Provinsi"]

for feature in feature_columns:

    IQR = iqr_values[feature]["IQR"]

    gap_robust[feature] = (
        gap_raw[feature] / IQR
    )

print("Robust Scaling berhasil dilakukan.")


# ============================================================
# 8. MEMBUAT TABEL INFORMASI IQR
# ============================================================

print("\n[6] Membuat tabel informasi IQR...")

iqr_report = pd.DataFrame([
    {
        "Fitur": feature,
        "Q1": iqr_values[feature]["Q1"],
        "Q3": iqr_values[feature]["Q3"],
        "IQR": iqr_values[feature]["IQR"],
        "Nilai_Nasional": nasional_dict[feature]
    }
    for feature in feature_columns
])


# ============================================================
# 9. VALIDASI HASIL
# ============================================================

print("\n[7] Validasi hasil...")

assert gap_raw.shape == (38, 10)

assert gap_robust.shape == (38, 10)

assert gap_robust[feature_columns].notnull().all().all()

assert np.isfinite(
    gap_robust[feature_columns].to_numpy()
).all()

print("Validasi berhasil.")
print("Jumlah provinsi:", len(gap_robust))
print("Jumlah fitur:", len(feature_columns))


# ============================================================
# 10. MENYIMPAN HASIL
# ============================================================

print("\n[8] Menyimpan hasil preprocessing...")

gap_raw.to_csv(
    OUTPUT_DIR / "05_gap_vector_raw.csv",
    index=False
)

gap_robust.to_csv(
    OUTPUT_DIR / "06_gap_vector_robust.csv",
    index=False
)

iqr_report.to_csv(
    OUTPUT_DIR / "07_laporan_iqr.csv",
    index=False
)


# ============================================================
# 11. MENAMPILKAN RINGKASAN
# ============================================================

print("\n")
print("=" * 65)
print("GAP VECTOR & ROBUST SCALING SELESAI")
print("=" * 65)

print("\nPREVIEW GAP RAW:")
print(gap_raw.head().to_string(index=False))

print("\nPREVIEW GAP ROBUST:")
print(gap_robust.head().to_string(index=False))

print("\nSTATISTIK GAP ROBUST:")
print(
    gap_robust[feature_columns].describe().round(4)
)

print("\nOUTPUT FILE:")

for file in OUTPUT_DIR.iterdir():
    print("-", file.name)

print("\nSeluruh proses berhasil diselesaikan.")
print("=" * 65)
