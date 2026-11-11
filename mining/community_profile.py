# ============================================================
# WEB MINING
# MINING - PROFIL & LABEL KOMUNITAS
# Laporan 5.6 langkah 4: "Beri label komunitas dengan ciri dominannya
# (misalnya 'akses dasar baik, fasilitas jenjang atas terbatas')."
#
# Profil = MEDIAN vektor gap (robust) setiap fitur di dalam komunitas.
# Median dipakai karena tahan terhadap pencilan (provinsi Papua, DKI).
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

FILE_GAP = PROJECT_DIR / "output_preprocessing" / "06_gap_vector_robust.csv"
FILE_COMMUNITY = BASE_DIR / "output" / "mining" / "04_louvain" / "louvain_communities.csv"

PROFILE_DIR = BASE_DIR / "output" / "mining" / "06_profil_komunitas"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "RLS", "APS_7_12", "APS_13_15", "APS_16_18", "APS_19_23",
    "Rasio_SMP_SD", "Rasio_SMU_SD", "Rasio_SMK_SD", "Rasio_PT_SD",
]

# Fitur dikelompokkan menjadi dimensi yang mudah dibaca
DIMENSI = {
    "capaian lama sekolah": ["RLS"],
    "akses dasar (APS 7-15)": ["APS_7_12", "APS_13_15"],
    "akses menengah (APS 16-18)": ["APS_16_18"],
    "akses pendidikan tinggi (APS 19-23)": ["APS_19_23"],
    "fasilitas SMP": ["Rasio_SMP_SD"],
    "fasilitas jenjang atas (SMU/SMK/PT)": ["Rasio_SMU_SD", "Rasio_SMK_SD", "Rasio_PT_SD"],
}


def tingkat(nilai):
    """Terjemahkan median gap (satuan IQR) menjadi kalimat."""
    if nilai >= 1.0:
        return "jauh di atas nasional"
    if nilai >= 0.3:
        return "di atas nasional"
    if nilai <= -1.0:
        return "jauh di bawah nasional"
    if nilai <= -0.3:
        return "di bawah nasional"
    return "setara nasional"


# ============================================================
# 2. PROGRAM UTAMA
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("PROFIL & LABEL KOMUNITAS LOUVAIN")
    print("=" * 60)

    for f in [FILE_GAP, FILE_COMMUNITY]:
        if not f.exists():
            raise FileNotFoundError(f)

    gap = pd.read_csv(FILE_GAP).set_index("Provinsi")
    kom = pd.read_csv(FILE_COMMUNITY).set_index("Provinsi")["Community"]

    if set(gap.index) != set(kom.index):
        raise ValueError("Nama provinsi pada gap dan komunitas tidak sama.")

    gap = gap.loc[kom.index, FEATURES]

    # --- Profil fitur (median gap) ---
    profil = gap.groupby(kom).median()
    profil.index.name = "Community"

    # --- Profil per dimensi + label otomatis ---
    rows = []
    for cid, row in profil.iterrows():

        nilai_dimensi = {d: float(row[f].mean()) for d, f in DIMENSI.items()}

        # 3 dimensi dengan |nilai| terbesar menjadi ciri dominan
        urut = sorted(nilai_dimensi.items(), key=lambda t: abs(t[1]), reverse=True)
        ciri = [f"{d.split(' (')[0]} {tingkat(v)}"
                for d, v in urut[:3] if tingkat(v) != "setara nasional"]

        label = "; ".join(ciri) if ciri else "pola mendekati nasional di semua dimensi"
        label = label[0].upper() + label[1:]

        anggota = sorted(kom[kom == cid].index)

        rows.append({
            "Community": cid,
            "Jumlah_Provinsi": len(anggota),
            "Label": label,
            **{f"Dimensi_{d}": round(v, 3) for d, v in nilai_dimensi.items()},
            "Anggota_Provinsi": "; ".join(anggota),
        })

    label_df = pd.DataFrame(rows)

    # --- Simpan ---
    profil.round(4).to_csv(PROFILE_DIR / "profil_median_gap.csv")
    label_df.to_csv(PROFILE_DIR / "label_komunitas.csv", index=False)

    print("\nProfil median gap per komunitas (satuan IQR, + = di atas nasional):")
    print(profil.round(2).to_string())

    print("\nLabel komunitas:")
    for _, r in label_df.iterrows():
        print(f"\nKomunitas {r['Community']} ({r['Jumlah_Provinsi']} provinsi)")
        print("  Ciri   :", r["Label"])
        print("  Anggota:", r["Anggota_Provinsi"])

    print("\nFile hasil tersimpan di:", PROFILE_DIR)
    print("1. profil_median_gap.csv\n2. label_komunitas.csv")


if __name__ == "__main__":
    main()