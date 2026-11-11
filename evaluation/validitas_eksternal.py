# ============================================================
# EVALUATION 1 - VALIDITAS EKSTERNAL (Laporan 2.3 & 5.8)
# Pertanyaan analitik 4: apakah kemiripan pola pendidikan sejalan
# dengan kedekatan IPM dan tingkat kemiskinan?
#
# Harapan laporan: korelasi Spearman antara kemiripan pasangan provinsi
# dan SELISIH MUTLAK IPM / kemiskinan bernilai NEGATIF dan signifikan
# (makin mirip -> makin kecil selisih IPM-nya).
# Signifikansi memakai uji Mantel (permutasi label provinsi).
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_utils as U

OUT = U.out_dir("01_validitas_eksternal")


def main():
    U.header("EVALUATION 1: VALIDITAS EKSTERNAL (IPM & KEMISKINAN)")
    U.check_consistency()

    S = U.load_similarity()
    val = U.load_validation().loc[S.index]

    rows, pairs = [], {"Kemiripan": U.upper(S)}
    for col, nama in [("IPM", "Selisih IPM"), ("Kemiskinan", "Selisih % penduduk miskin (Sep)")]:
        M = U.abs_diff_matrix(val[col])
        res = U.mantel(S, M)
        lolos = res["r_spearman"] < 0 and res["p_dua_arah"] < 0.05
        rows.append({
            "Pembanding": nama,
            "Spearman_r": round(res["r_spearman"], 4),
            "Kekuatan": U.kekuatan(res["r_spearman"]),
            "p_Mantel_dua_arah": round(res["p_dua_arah"], 4),
            "p_Mantel_satu_arah_negatif": round(res["p_satu_arah_negatif"], 4),
            "N_pasangan": res["n_pasangan"],
            "N_permutasi": res["n_permutasi"],
            "Kriteria": "r negatif & p < 0,05",
            "Status": "LOLOS" if lolos else "TIDAK LOLOS",
        })
        pairs[nama] = U.upper(M)
        print(f"\n{nama}: r = {res['r_spearman']:.4f} ({U.kekuatan(res['r_spearman'])}), "
              f"p = {res['p_dua_arah']:.4f} -> {'LOLOS' if lolos else 'TIDAK LOLOS'}")

    hasil = pd.DataFrame(rows)
    hasil.to_csv(OUT / "hasil_validitas_eksternal.csv", index=False)

    # Data pasangan untuk lampiran / grafik
    iu = np.triu_indices(len(S), k=1)
    pasangan = pd.DataFrame({"Provinsi_A": S.index[iu[0]], "Provinsi_B": S.index[iu[1]], **pairs})
    pasangan.round(4).to_csv(OUT / "pasangan_kemiripan_vs_selisih.csv", index=False)

    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (_, r), nama in zip(axes, hasil.iterrows(), ["Selisih IPM", "Selisih % penduduk miskin (Sep)"]):
        ax.scatter(pasangan[nama], pasangan["Kemiripan"], s=10, alpha=0.5, color=U.BLUE, edgecolors="none")
        ax.set_xlabel(f"{nama} (mutlak)")
        ax.set_ylabel("Weighted cosine similarity")
        ax.set_title(f"Spearman {r['Spearman_r']:.3f}, p Mantel {r['p_Mantel_dua_arah']:.4f}", loc="left")
    fig.savefig(OUT / "scatter_validitas_eksternal.png")
    plt.close(fig)

    print("\n" + hasil.to_string(index=False))
    print("\nFile hasil:", OUT)


if __name__ == "__main__":
    main()