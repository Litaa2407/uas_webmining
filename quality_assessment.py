
# ============================================================
# WEB MINING
# PREPROCESSING - QUALITY ASSESSMENT
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
FILE_GAP_RAW = OUTPUT_DIR / "05_gap_vector_raw.csv"
FILE_GAP_ROBUST = OUTPUT_DIR / "06_gap_vector_robust.csv"
FILE_IQR = OUTPUT_DIR / "07_laporan_iqr.csv"

print("=" * 65)
print("PREPROCESSING - QUALITY ASSESSMENT")
print("=" * 65)

# ============================================================
# 2. MEMBACA DATA
# ============================================================

data = pd.read_csv(FILE_DATA)
gap_raw = pd.read_csv(FILE_GAP_RAW)
gap_robust = pd.read_csv(FILE_GAP_ROBUST)
iqr_report = pd.read_csv(FILE_IQR)

features = [
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

results = []

# ============================================================
# 3. PEMERIKSAAN STRUKTUR DATA
# ============================================================

print("\n[1] Pemeriksaan struktur data")

def record_result(name, status, detail):
    results.append({
        "Pemeriksaan": name,
        "Status": status,
        "Keterangan": detail
    })

record_result(
    "Jumlah provinsi",
    "PASS" if len(data) == 38 else "FAIL",
    f"{len(data)} provinsi"
)

record_result(
    "Jumlah fitur",
    "PASS" if len(features) == 9 else "FAIL",
    f"{len(features)} fitur"
)

record_result(
    "Struktur gap raw",
    "PASS" if gap_raw.shape == (38, 10) else "FAIL",
    str(gap_raw.shape)
)

record_result(
    "Struktur gap robust",
    "PASS" if gap_robust.shape == (38, 10) else "FAIL",
    str(gap_robust.shape)
)

# ============================================================
# 4. PEMERIKSAAN MISSING VALUES
# ============================================================

print("\n[2] Pemeriksaan missing values")

for name, df in [
    ("Dataset utama", data),
    ("Gap raw", gap_raw),
    ("Gap robust", gap_robust)
]:

    missing = int(df.isnull().sum().sum())

    record_result(
        f"Missing values - {name}",
        "PASS" if missing == 0 else "FAIL",
        f"{missing} missing values"
    )

# ============================================================
# 5. PEMERIKSAAN DUPLIKASI
# ============================================================

print("\n[3] Pemeriksaan duplikasi")

for name, df in [
    ("Dataset utama", data),
    ("Gap raw", gap_raw),
    ("Gap robust", gap_robust)
]:

    duplicate = int(df["Provinsi"].duplicated().sum())

    record_result(
        f"Duplikasi - {name}",
        "PASS" if duplicate == 0 else "FAIL",
        f"{duplicate} duplikasi"
    )

# ============================================================
# 6. KONSISTENSI NAMA PROVINSI
# ============================================================

print("\n[4] Pemeriksaan konsistensi provinsi")

same_provinces = (
    set(data["Provinsi"]) == set(gap_raw["Provinsi"])
    and
    set(data["Provinsi"]) == set(gap_robust["Provinsi"])
)

record_result(
    "Konsistensi nama provinsi",
    "PASS" if same_provinces else "FAIL",
    "Nama provinsi konsisten" if same_provinces
    else "Nama provinsi tidak konsisten"
)

# ============================================================
# 7. VALIDASI PERHITUNGAN GAP
# ============================================================

print("\n[5] Validasi perhitungan gap")

# IQR digunakan untuk memeriksa hasil transformasi
iqr_dict = dict(
    zip(iqr_report["Fitur"], iqr_report["IQR"])
)

gap_check = True

for feature in features:

    expected = (
        gap_raw[feature].to_numpy()
        / iqr_dict[feature]
    )

    actual = gap_robust[feature].to_numpy()

    if not np.allclose(expected, actual):
        gap_check = False
        break

record_result(
    "Konsistensi perhitungan gap robust",
    "PASS" if gap_check else "FAIL",
    "Hasil sesuai rumus gap/IQR"
    if gap_check else "Hasil tidak sesuai rumus"
)

# ============================================================
# 8. STATISTIK DESKRIPTIF
# ============================================================

print("\n[6] Statistik deskriptif")

statistics = gap_robust[features].describe().T

statistics["skewness"] = (
    gap_robust[features].skew()
)

statistics = statistics.round(4)

print(statistics)

# ============================================================
# 9. DISTRIBUSI NILAI GAP
# ============================================================

print("\n[7] Pemeriksaan distribusi gap")

distribution = []

for feature in features:

    values = gap_robust[feature]

    distribution.append({

        "Fitur": feature,

        "Di_Bawah_Nasional": int(
            (values < 0).sum()
        ),

        "Sama_Dengan_Nasional": int(
            np.isclose(values, 0).sum()
        ),

        "Di_Atas_Nasional": int(
            (values > 0).sum()
        )

    })

distribution = pd.DataFrame(distribution)

print(distribution.to_string(index=False))

# ============================================================
# 10. MENYIMPAN LAPORAN
# ============================================================

print("\n[8] Menyimpan laporan")

quality_report = pd.DataFrame(results)

quality_report.to_csv(
    OUTPUT_DIR / "08_quality_assessment.csv",
    index=False
)

statistics.to_csv(
    OUTPUT_DIR / "09_statistik_gap_robust.csv"
)

distribution.to_csv(
    OUTPUT_DIR / "10_distribusi_gap.csv",
    index=False
)

# ============================================================
# 11. KESIMPULAN
# ============================================================

print("\n")
print("=" * 65)
print("HASIL QUALITY ASSESSMENT")
print("=" * 65)

print(quality_report.to_string(index=False))

failed = quality_report[
    quality_report["Status"] == "FAIL"
]

if len(failed) == 0:

    print("\nSTATUS: SELURUH PEMERIKSAAN LULUS")

    print(
        "Dataset telah memenuhi pemeriksaan kualitas "
        "untuk melanjutkan ke tahap berikutnya."
    )

else:

    print("\nSTATUS: DITEMUKAN MASALAH")

    print(failed.to_string(index=False))

    print(
        "\nPerbaiki masalah terlebih dahulu "
        "sebelum melanjutkan."
    )

print("\nOutput laporan berhasil disimpan.")
print("=" * 65)
