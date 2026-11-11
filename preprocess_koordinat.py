# ============================================================
# WEB MINING
# PREPROCESSING - KOORDINAT IBU KOTA PROVINSI
# Dashboard Daerah Kembar Pendidikan Indonesia
#
# Laporan 3.4, 4.6, dan Lampiran A:
# - daftar-nama-daerah.csv hanya memuat 33 provinsi.
# - Koordinat baris provinsi adalah titik tengah wilayah, BUKAN ibu kota,
#   sehingga yang dipakai adalah baris kabupaten/kota ibu kota provinsi.
# - Nama ganda dengan koordinat identik dibuang salah satunya.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output_preprocessing"
OUTPUT_DIR.mkdir(exist_ok=True)

FILE_DAERAH = DATA_DIR / "daftar-nama-daerah.csv"
FILE_DATA = OUTPUT_DIR / "01_dataset_pendidikan_38_provinsi.csv"

# Provinsi (nama sama dengan output preprocess.py) ->
# (ibu kota, nama baris pada daftar-nama-daerah.csv)   [Lampiran A]
IBU_KOTA = {
    "ACEH": ("Banda Aceh", "Kota Banda Aceh"),
    "SUMATERA UTARA": ("Medan", "Kota Medan"),
    "SUMATERA BARAT": ("Padang", "Kota Padang"),
    "RIAU": ("Pekanbaru", "Kota Pekanbaru"),
    "JAMBI": ("Jambi", "Kota Jambi"),
    "SUMATERA SELATAN": ("Palembang", "Kota Palembang"),
    "BENGKULU": ("Bengkulu", "Kota Bengkulu"),
    "LAMPUNG": ("Bandar Lampung", "Kota Bandar Lampung"),
    "KEPULAUAN BANGKA BELITUNG": ("Pangkalpinang", "Kota Pangkal Pinang"),
    "KEPULAUAN RIAU": ("Tanjungpinang", "Kota Tanjung Pinang"),
    "DKI JAKARTA": ("Jakarta", "Kota Jakarta Pusat"),
    "JAWA BARAT": ("Bandung", "Kota Bandung"),
    "JAWA TENGAH": ("Semarang", "Kota Semarang"),
    "DI YOGYAKARTA": ("Yogyakarta", "Kota Yogyakarta"),
    "JAWA TIMUR": ("Surabaya", "Kota Surabaya"),
    "BANTEN": ("Serang", "Kota Serang"),
    "BALI": ("Denpasar", "Kota Denpasar"),
    "NUSA TENGGARA BARAT": ("Mataram", "Kota Mataram"),
    "NUSA TENGGARA TIMUR": ("Kupang", "Kota Kupang"),
    "KALIMANTAN BARAT": ("Pontianak", "Kota Pontianak"),
    "KALIMANTAN TENGAH": ("Palangka Raya", "Kota Palangka Raya"),
    "KALIMANTAN SELATAN": ("Banjarmasin", "Kota Banjarmasin"),
    "KALIMANTAN TIMUR": ("Samarinda", "Kota Samarinda"),
    "KALIMANTAN UTARA": ("Tanjung Selor", "Kabupaten Bulungan"),
    "SULAWESI UTARA": ("Manado", "Kota Manado"),
    "SULAWESI TENGAH": ("Palu", "Kota Palu"),
    "SULAWESI SELATAN": ("Makassar", "Kota Makassar"),
    "SULAWESI TENGGARA": ("Kendari", "Kota Kendari"),
    "GORONTALO": ("Gorontalo", "Kota Gorontalo"),
    "SULAWESI BARAT": ("Mamuju", "Kabupaten Mamuju"),
    "MALUKU": ("Ambon", "Kota Ambon"),
    "MALUKU UTARA": ("Sofifi", "Kota Tidore Kepulauan"),
    "PAPUA BARAT": ("Manokwari", "Kabupaten Manokwari"),
    "PAPUA BARAT DAYA": ("Sorong", "Kota Sorong"),
    "PAPUA": ("Jayapura", "Kota Jayapura"),
    "PAPUA SELATAN": ("Merauke", "Kabupaten Merauke"),
    "PAPUA TENGAH": ("Nabire", "Kabupaten Nabire"),
    "PAPUA PEGUNUNGAN": ("Wamena", "Kabupaten Jayawijaya"),
}

print("=" * 65)
print("PREPROCESSING - KOORDINAT IBU KOTA PROVINSI")
print("=" * 65)


# ============================================================
# 2. MEMBACA DATA DAERAH
# ============================================================

print("\n[1] Membaca daftar-nama-daerah.csv ...")

if not FILE_DAERAH.exists():
    raise FileNotFoundError(FILE_DAERAH)

daerah = pd.read_csv(FILE_DAERAH, encoding="utf-8-sig")
daerah["name"] = daerah["name"].astype(str).str.strip()

print("Jumlah baris          :", len(daerah))
print("Baris provinsi (type 1):", int((daerah["type"] == 1).sum()))
print("Baris kab/kota (type 2):", int((daerah["type"] == 2).sum()))
print("Nilai kosong lat/lon  :",
      int(daerah[["latitude", "longitude"]].isnull().sum().sum()))


# ============================================================
# 3. MEMBERSIHKAN DUPLIKAT
# ============================================================

print("\n[2] Pemeriksaan duplikat kabupaten/kota ...")

kabkota = daerah[daerah["type"] == 2].copy()

nama_ganda = kabkota[kabkota["name"].duplicated(keep=False)]
print("Nama ganda:", nama_ganda["name"].nunique(), "nama /",
      len(nama_ganda), "baris")

before = len(kabkota)
kabkota = kabkota.drop_duplicates(subset=["name", "latitude", "longitude"])
print("Duplikat identik (nama + koordinat sama) dibuang:", before - len(kabkota))

masih_ganda = kabkota[kabkota["name"].duplicated(keep=False)]["name"].unique()
if len(masih_ganda) > 0:
    print("Nama ganda dengan koordinat BERBEDA (tidak dibuang):", list(masih_ganda))


# ============================================================
# 4. MENGAMBIL KOORDINAT IBU KOTA
# ============================================================

print("\n[3] Mengambil koordinat ibu kota ...")

records = []

for provinsi, (ibu_kota, nama_baris) in IBU_KOTA.items():

    hit = kabkota[kabkota["name"] == nama_baris]

    if len(hit) != 1:
        raise ValueError(
            f"Baris '{nama_baris}' untuk {provinsi} ditemukan {len(hit)} kali."
        )

    records.append({
        "Provinsi": provinsi,
        "Ibu_Kota": ibu_kota,
        "Baris_Berkas": nama_baris,
        "Latitude": float(hit["latitude"].iloc[0]),
        "Longitude": float(hit["longitude"].iloc[0]),
        "Papua": provinsi.startswith("PAPUA"),
    })

koordinat = pd.DataFrame(records).sort_values("Provinsi").reset_index(drop=True)


# ============================================================
# 5. VALIDASI TERHADAP DATASET PENDIDIKAN
# ============================================================

print("\n[4] Validasi kecocokan nama provinsi ...")

if FILE_DATA.exists():
    prov_data = set(pd.read_csv(FILE_DATA)["Provinsi"])
    prov_koor = set(koordinat["Provinsi"])
    if prov_data != prov_koor:
        raise ValueError(
            f"Nama provinsi tidak cocok.\n"
            f"Hanya di dataset: {prov_data - prov_koor}\n"
            f"Hanya di koordinat: {prov_koor - prov_data}"
        )
    print("38 provinsi cocok dengan dataset pendidikan: PASS")
else:
    print("Jalankan preprocess.py terlebih dahulu untuk validasi nama.")

assert len(koordinat) == 38
assert koordinat["Latitude"].between(-11, 6).all()
assert koordinat["Longitude"].between(95, 141).all()
print("Rentang koordinat berada di wilayah Indonesia: PASS")


# ============================================================
# 6. MENYIMPAN HASIL
# ============================================================

output_file = OUTPUT_DIR / "11_koordinat_ibu_kota.csv"
koordinat.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("KOORDINAT IBU KOTA SELESAI")
print("=" * 65)
print(koordinat.to_string(index=False))
print("\nFile:", output_file)