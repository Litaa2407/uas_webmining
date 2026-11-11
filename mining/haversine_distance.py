# ============================================================
# WEB MINING
# MINING - JARAK GEOGRAFIS (HAVERSINE)
# Laporan 5.2 langkah 7 dan 5.7
#
# d(a,b) = 2R * arcsin( sqrt( sin^2((phi_b - phi_a)/2)
#          + cos(phi_a) cos(phi_b) sin^2((lambda_b - lambda_a)/2) ) )
# R = 6.371,0088 km
# Dipakai sebagai PEMBANDING (validasi), bukan fitur.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FILE_KOORDINAT = PROJECT_DIR / "output_preprocessing" / "11_koordinat_ibu_kota.csv"
FILE_SIMILARITY = BASE_DIR / "output" / "mining" / "02_weighted_cosine" / "similarity_matrix.csv"

DISTANCE_DIR = BASE_DIR / "output" / "mining" / "05_jarak_geografis"
DISTANCE_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6371.0088


# ============================================================
# 2. MEMBACA KOORDINAT
# ============================================================

def load_coordinates():

    print("\n" + "=" * 60)
    print("TAHAP 1: MEMBACA KOORDINAT IBU KOTA")
    print("=" * 60)

    if not FILE_KOORDINAT.exists():
        raise FileNotFoundError(
            f"{FILE_KOORDINAT}\nJalankan preprocess_koordinat.py terlebih dahulu."
        )

    df = pd.read_csv(FILE_KOORDINAT)

    # Urutan provinsi disamakan dengan matriks similarity
    if FILE_SIMILARITY.exists():
        order = pd.read_csv(FILE_SIMILARITY, index_col=0).index.tolist()
        df = df.set_index("Provinsi").loc[order].reset_index()

    if len(df) != 38:
        raise ValueError(f"Jumlah provinsi seharusnya 38, ditemukan {len(df)}.")

    print("Jumlah provinsi:", len(df))
    return df


# ============================================================
# 3. MENGHITUNG MATRIKS JARAK HAVERSINE
# ============================================================

def haversine_matrix(df):

    print("\n" + "=" * 60)
    print("TAHAP 2: MATRIKS JARAK HAVERSINE")
    print("=" * 60)

    phi = np.radians(df["Latitude"].to_numpy())
    lam = np.radians(df["Longitude"].to_numpy())

    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]

    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2) ** 2)

    D = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(D, 0.0)

    assert np.allclose(D, D.T)

    provinces = df["Provinsi"].tolist()
    distance_df = pd.DataFrame(D, index=provinces, columns=provinces)
    distance_df.index.name = "Provinsi"

    print("Ukuran matriks:", distance_df.shape)
    return distance_df


# ============================================================
# 4. DAFTAR PASANGAN DAN STATISTIK (Laporan 4.6)
# ============================================================

def distance_pairs_and_summary(distance_df):

    print("\n" + "=" * 60)
    print("TAHAP 3: STATISTIK JARAK")
    print("=" * 60)

    provinces = distance_df.index.tolist()
    D = distance_df.to_numpy()
    iu = np.triu_indices(len(provinces), k=1)

    pairs = pd.DataFrame({
        "Provinsi_A": [provinces[i] for i in iu[0]],
        "Provinsi_B": [provinces[j] for j in iu[1]],
        "Jarak_km": D[iu],
    }).sort_values("Jarak_km").reset_index(drop=True)

    D_no_diag = D.copy()
    np.fill_diagonal(D_no_diag, np.inf)
    nearest = D_no_diag.min(axis=1)

    v = pairs["Jarak_km"]
    summary = pd.DataFrame([
        ("Jumlah pasangan provinsi", len(pairs)),
        ("Jarak minimum (km)", round(v.min(), 1)),
        ("Pasangan minimum", f"{pairs.iloc[0]['Provinsi_A']} - {pairs.iloc[0]['Provinsi_B']}"),
        ("Jarak maksimum (km)", round(v.max(), 1)),
        ("Pasangan maksimum", f"{pairs.iloc[-1]['Provinsi_A']} - {pairs.iloc[-1]['Provinsi_B']}"),
        ("Rata-rata (km)", round(v.mean(), 1)),
        ("Median (km)", round(v.median(), 1)),
        ("Simpangan baku (km)", round(v.std(), 1)),
        ("Kuartil 1 (km)", round(v.quantile(0.25), 1)),
        ("Kuartil 3 (km)", round(v.quantile(0.75), 1)),
        ("Skewness", round(float(skew(v, bias=False)), 2)),
        ("Rata-rata jarak ke provinsi terdekat (km)", round(nearest.mean(), 1)),
        ("Provinsi terjauh dari tetangganya",
         f"{provinces[int(np.argmax(nearest))]} ({nearest.max():.1f} km)"),
    ], columns=["Statistik", "Nilai"])

    print(summary.to_string(index=False))
    return pairs, summary


# ============================================================
# 5. PROGRAM UTAMA
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("JARAK GEOGRAFIS HAVERSINE ANTAR IBU KOTA")
    print("=" * 60)

    df = load_coordinates()
    distance_df = haversine_matrix(df)
    pairs, summary = distance_pairs_and_summary(distance_df)

    distance_df.round(3).to_csv(DISTANCE_DIR / "distance_matrix_km.csv")
    pairs.round(3).to_csv(DISTANCE_DIR / "distance_pairs.csv", index=False)
    summary.to_csv(DISTANCE_DIR / "distance_summary.csv", index=False)

    print("\nFile hasil tersimpan di:", DISTANCE_DIR)
    print("1. distance_matrix_km.csv\n2. distance_pairs.csv\n3. distance_summary.csv")


if __name__ == "__main__":
    main()