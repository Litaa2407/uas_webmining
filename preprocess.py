
# ============================================================
# WEB MINING
# PREPROCESSING DATA
# Dashboard Daerah Kembar Pendidikan Indonesia
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

# ============================================================
# 1. KONFIGURASI FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output_preprocessing"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("WEB MINING - PREPROCESSING")
print("=" * 65)


# ============================================================
# 2. MEMERIKSA FOLDER DATASET
# ============================================================

print("\n[1] Memeriksa dataset...")

if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan: {DATA_DIR}"
    )

if not list(DATA_DIR.glob("*.csv")):
    raise FileNotFoundError(
        f"Tidak ada file CSV di folder dataset: {DATA_DIR}"
    )

print(f"Menggunakan file CSV yang sudah tersedia di: {DATA_DIR}")


# ============================================================
# 3. FUNGSI MEMBACA FILE
# ============================================================

def find_file(keyword):

    files = list(DATA_DIR.glob("*.csv"))

    matches = [
        file for file in files
        if keyword.lower() in file.name.lower()
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"File dengan keyword '{keyword}' tidak ditemukan."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Ditemukan lebih dari satu file untuk '{keyword}': {matches}"
        )

    return matches[0]


# ============================================================
# 4. STANDARDISASI NAMA PROVINSI
# ============================================================

def clean_province(df):

    df = df.copy()

    df["Provinsi"] = (
        df["Provinsi"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapping = {

        "KEP. RIAU": "KEPULAUAN RIAU",
        "KEP RIAU": "KEPULAUAN RIAU",

        "KEP. BANGKA BELITUNG":
            "KEPULAUAN BANGKA BELITUNG",

        "KEP BANGKA BELITUNG":
            "KEPULAUAN BANGKA BELITUNG",

        "KEPULAUAN BANGKA BELITUNG":
            "KEPULAUAN BANGKA BELITUNG",

    }

    df["Provinsi"] = df["Provinsi"].replace(mapping)

    return df


# ============================================================
# 5. MEMBACA DATA RATA-RATA LAMA SEKOLAH
# ============================================================

print("\n[2] Membaca dataset RLS...")

file_rls = find_file("Rata-Rata Lama Sekolah")

rls = pd.read_csv(
    file_rls,
    skiprows=3,
    header=None,
    names=["Provinsi", "RLS"],
    encoding="utf-8-sig"
)

rls = clean_province(rls)

print("Dataset RLS:", rls.shape)


# ============================================================
# 6. MEMBACA DATA APS
# ============================================================

print("\n[3] Membaca dataset APS...")

file_aps = find_file(
    "Angka Partisipasi Sekolah (APS) Menurut"
)

aps = pd.read_csv(
    file_aps,
    skiprows=4,
    header=None,
    names=[
        "Provinsi",
        "APS_7_12",
        "APS_13_15",
        "APS_16_18",
        "APS_19_23"
    ],
    encoding="utf-8-sig"
)

aps = clean_province(aps)

print("Dataset APS:", aps.shape)


# ============================================================
# 7. MEMBACA DATA FASILITAS SEKOLAH
# ============================================================

print("\n[4] Membaca dataset fasilitas sekolah...")

file_fasilitas = find_file(
    "Jumlah Desa yang Memiliki Fasilitas Sekolah"
)

fasilitas = pd.read_csv(
    file_fasilitas,
    skiprows=4,
    header=None,
    names=[
        "Provinsi",
        "SD",
        "SMP",
        "SMU",
        "SMK",
        "PT"
    ],
    encoding="utf-8-sig"
)

fasilitas = clean_province(fasilitas)

print("Dataset fasilitas:", fasilitas.shape)


# ============================================================
# 8. KONVERSI TIPE DATA NUMERIK
# ============================================================

print("\n[5] Konversi tipe data...")

datasets = [rls, aps, fasilitas]

for df in datasets:

    for col in df.columns:

        if col != "Provinsi":

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

print("Konversi tipe data selesai.")


# ============================================================
# 9. MEMISAHKAN DATA NASIONAL
# ============================================================

print("\n[6] Memisahkan data nasional...")

def separate_national(df):

    national = df[
        df["Provinsi"] == "INDONESIA"
    ].copy()

    provincial = df[
        df["Provinsi"] != "INDONESIA"
    ].copy()

    return provincial, national


rls, rls_nasional = separate_national(rls)

aps, aps_nasional = separate_national(aps)

fasilitas, fasilitas_nasional = separate_national(
    fasilitas
)

print("Data nasional berhasil dipisahkan.")


# ============================================================
# 10. PEMERIKSAAN DUPLIKASI
# ============================================================

print("\n[7] Pemeriksaan duplikasi...")

for name, df in [
    ("RLS", rls),
    ("APS", aps),
    ("Fasilitas", fasilitas)
]:

    duplicate_count = df["Provinsi"].duplicated().sum()

    print(f"{name}: {duplicate_count} duplikat")

    if duplicate_count > 0:
        raise ValueError(
            f"Terdapat duplikasi provinsi pada dataset {name}."
        )


# ============================================================
# 11. INTEGRASI DATASET PENDIDIKAN
# ============================================================

print("\n[8] Integrasi dataset pendidikan...")

data_master = pd.merge(
    rls,
    aps,
    on="Provinsi",
    how="outer",
    validate="one_to_one"
)

data_master = pd.merge(
    data_master,
    fasilitas,
    on="Provinsi",
    how="outer",
    validate="one_to_one"
)

print("Ukuran data setelah integrasi:", data_master.shape)


# ============================================================
# 12. FEATURE ENGINEERING
# Membuat empat rasio fasilitas pendidikan
# ============================================================

print("\n[9] Feature engineering...")

data_master["Rasio_SMP_SD"] = (
    data_master["SMP"] / data_master["SD"]
)

data_master["Rasio_SMU_SD"] = (
    data_master["SMU"] / data_master["SD"]
)

data_master["Rasio_SMK_SD"] = (
    data_master["SMK"] / data_master["SD"]
)

data_master["Rasio_PT_SD"] = (
    data_master["PT"] / data_master["SD"]
)

print("Empat fitur rasio berhasil dibuat.")


# ============================================================
# 13. SELEKSI 9 FITUR UTAMA
# ============================================================

print("\n[10] Seleksi fitur utama...")

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

data_final = data_master[
    ["Provinsi"] + feature_columns
].copy()

data_final = data_final.sort_values(
    "Provinsi"
).reset_index(drop=True)

print("Jumlah fitur:", len(feature_columns))


# ============================================================
# 14. MEMBACA DATA IPM
# ============================================================

print("\n[11] Membaca dataset IPM...")

file_ipm = find_file(
    "Indeks Pembangunan Manusia"
)

ipm = pd.read_csv(
    file_ipm,
    encoding="utf-8-sig"
)

ipm.columns = [
    "Provinsi",
    "IPM"
]

ipm = clean_province(ipm)

ipm["IPM"] = pd.to_numeric(
    ipm["IPM"],
    errors="coerce"
)

ipm, ipm_nasional = separate_national(ipm)


# ============================================================
# 15. MEMBACA DATA KEMISKINAN
# ============================================================

print("\n[12] Membaca dataset kemiskinan...")

file_miskin = find_file(
    "Jumlah dan Persentase Penduduk Miskin"
)

miskin = pd.read_csv(
    file_miskin,
    encoding="utf-8-sig"
)

miskin = clean_province(miskin)

# Menggunakan persentase kemiskinan September
miskin = miskin[
    [
        "Provinsi",
        "Persentase Penduduk Miskin - September"
    ]
].copy()

miskin.rename(
    columns={
        "Persentase Penduduk Miskin - September":
        "Kemiskinan"
    },
    inplace=True
)

miskin["Kemiskinan"] = pd.to_numeric(
    miskin["Kemiskinan"],
    errors="coerce"
)

miskin, miskin_nasional = separate_national(
    miskin
)


# ============================================================
# 16. INTEGRASI DATA VALIDASI
# ============================================================

print("\n[13] Integrasi data validasi...")

data_validasi = pd.merge(
    ipm,
    miskin,
    on="Provinsi",
    how="outer",
    validate="one_to_one"
)

data_validasi = data_validasi.sort_values(
    "Provinsi"
).reset_index(drop=True)


# ============================================================
# 17. PEMERIKSAAN MISSING VALUES
# ============================================================

print("\n[14] Pemeriksaan missing values...")

print("\nDataset utama:")
print(data_final.isnull().sum())

print("\nDataset validasi:")
print(data_validasi.isnull().sum())


# ============================================================
# 18. VALIDASI JUMLAH DATA
# ============================================================

print("\n[15] Validasi jumlah data...")

assert len(data_final) == 38, (
    f"Jumlah provinsi tidak sesuai: {len(data_final)}"
)

assert len(data_validasi) == 38, (
    f"Jumlah data validasi tidak sesuai: "
    f"{len(data_validasi)}"
)

assert data_final["Provinsi"].nunique() == 38

assert data_validasi["Provinsi"].nunique() == 38

assert data_final[feature_columns].notnull().all().all(), (
    "Terdapat missing values pada fitur utama."
)

assert data_validasi[
    ["IPM", "Kemiskinan"]
].notnull().all().all(), (
    "Terdapat missing values pada data validasi."
)

assert (data_final["SD"] if "SD" in data_final.columns
        else data_master["SD"]).gt(0).all(), (
    "Terdapat nilai SD nol atau negatif."
)

print("Semua pemeriksaan berhasil!")


# ============================================================
# 19. MEMBUAT REFERENSI NILAI NASIONAL
# ============================================================

print("\n[16] Membuat referensi nasional...")

referensi_nasional = pd.DataFrame({

    "Fitur": [

        "RLS",
        "APS_7_12",
        "APS_13_15",
        "APS_16_18",
        "APS_19_23",
        "Rasio_SMP_SD",
        "Rasio_SMU_SD",
        "Rasio_SMK_SD",
        "Rasio_PT_SD"

    ],

    "Nilai_Nasional": [

        rls_nasional["RLS"].iloc[0],

        aps_nasional["APS_7_12"].iloc[0],
        aps_nasional["APS_13_15"].iloc[0],
        aps_nasional["APS_16_18"].iloc[0],
        aps_nasional["APS_19_23"].iloc[0],

        fasilitas_nasional["SMP"].iloc[0] /
        fasilitas_nasional["SD"].iloc[0],

        fasilitas_nasional["SMU"].iloc[0] /
        fasilitas_nasional["SD"].iloc[0],

        fasilitas_nasional["SMK"].iloc[0] /
        fasilitas_nasional["SD"].iloc[0],

        fasilitas_nasional["PT"].iloc[0] /
        fasilitas_nasional["SD"].iloc[0]

    ]

})

print(referensi_nasional)


# ============================================================
# 20. EXPORT DATASET HASIL PREPROCESSING
# ============================================================

print("\n[17] Menyimpan hasil preprocessing...")

# Dataset utama: 38 provinsi x 9 fitur
data_final.to_csv(
    OUTPUT_DIR / "01_dataset_pendidikan_38_provinsi.csv",
    index=False
)

# Dataset validasi IPM dan kemiskinan
data_validasi.to_csv(
    OUTPUT_DIR / "02_dataset_validasi.csv",
    index=False
)

# Referensi nasional
referensi_nasional.to_csv(
    OUTPUT_DIR / "03_referensi_nasional.csv",
    index=False
)

# Data fasilitas mentah untuk dokumentasi
data_master[
    ["Provinsi", "SD", "SMP", "SMU", "SMK", "PT"]
].to_csv(
    OUTPUT_DIR / "04_fasilitas_mentah.csv",
    index=False
)


# ============================================================
# 21. RINGKASAN HASIL PREPROCESSING
# ============================================================

print("\n")
print("=" * 65)
print("PREPROCESSING SELESAI")
print("=" * 65)

print("\nDATASET UTAMA")
print("Jumlah provinsi :", len(data_final))
print("Jumlah fitur    :", len(feature_columns))
print("Ukuran dataset  :", data_final.shape)

print("\nDATASET VALIDASI")
print("Jumlah provinsi :", len(data_validasi))
print("Jumlah fitur    :", 2)
print("Ukuran dataset  :", data_validasi.shape)

print("\nOUTPUT FILE:")

for file in OUTPUT_DIR.iterdir():
    print("-", file.name)

print("\nPreview dataset utama:")
print(data_final.head(10).to_string(index=False))

print("\nStatistik deskriptif:")
print(data_final[feature_columns].describe())

print("\nSemua dataset berhasil diproses dan disimpan.")
print("=" * 65)
