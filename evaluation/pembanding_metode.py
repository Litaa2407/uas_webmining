# ============================================================
# EVALUATION 6 - PEMBANDING METODE (Laporan 5.8)
# Menunjukkan nilai tambah rancangan dibanding metode dasar:
#   1. Weighted cosine (gap + entropi)        [metode utama]
#   2. Cosine biasa (gap, tanpa bobot)
#   3. Jarak Euclidean berbobot (gap + entropi)  -> menangkap BESAR selisih
#   4. K-Means pada vektor gap berbobot (k dipilih silhouette, dan k = jumlah komunitas Louvain)
# Ukuran: korelasi dengan selisih IPM (validitas), Jaccard top-5 vs utama,
#         ARI kelompok vs Louvain utama, ukuran kelompok.
# Catatan laporan: pembanding lengkap dibahas di minggu 14; ini versi awalnya.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import eval_utils as U

OUT = U.out_dir("06_pembanding_metode")


def main():
    U.header("EVALUATION 6: PEMBANDING METODE")
    U.check_consistency()

    data, nas, val = U.load_data(), U.load_national(), U.load_validation()
    S_main = U.load_similarity()
    order = S_main.index
    lab_main = U.load_communities(order)
    t_main = U.top_k(S_main)
    ipm_M = U.abs_diff_matrix(val.loc[order, "IPM"])

    G = U.gap_robust(data, nas).loc[order]
    w = U.entropy_weights(data)

    kandidat = {
        "Weighted cosine (gap + entropi) [utama]": (S_main, True),
        "Cosine biasa (gap, tanpa bobot)": (U.weighted_cosine(G, U.equal_weights()), True),
        "Jarak Euclidean berbobot (gap + entropi)": (U.weighted_euclidean(G, w), False),
    }

    rows = []
    for nama, (M, besar_mirip) in kandidat.items():
        sim_like = M if besar_mirip else -M
        mt = U.mantel(sim_like, ipm_M, n_perm=999)
        # Euclidean diubah ke kemiripan 1/(1+d) supaya bisa dibuat graf k-NN
        lab, q = U.louvain_best(U.knn_graph(M if besar_mirip else 1 / (1 + M)))
        rows.append({
            "Metode": nama,
            "Spearman_vs_Selisih_IPM": round(mt["r_spearman"], 4),
            "p_Mantel_999": round(mt["p_dua_arah"], 4),
            "Jaccard_Top5_vs_Utama": round(U.mean_jaccard(t_main, U.top_k(M, larger_is_closer=besar_mirip)), 4),
            "Jumlah_Kelompok": lab.nunique(),
            "Ukuran_Kelompok": str(sorted(lab.value_counts().tolist(), reverse=True)),
            "Modularity_Louvain": round(q, 4),
            "ARI_vs_Louvain_Utama": round(U.ari(lab_main, lab), 4),
        })

    # K-Means
    A = G.to_numpy() * np.sqrt(w.to_numpy())
    best = max(((silhouette_score(A, km.labels_), k, km.labels_)
                for k in range(2, 9)
                for km in [KMeans(n_clusters=k, n_init=20, random_state=U.RANDOM_SEED).fit(A)]),
               key=lambda t: t[0])
    k_lv = lab_main.nunique()
    km_same = KMeans(n_clusters=k_lv, n_init=20, random_state=U.RANDOM_SEED).fit(A).labels_
    for nama, labels in [(f"K-Means k={best[1]} (silhouette terbaik {best[0]:.3f})", best[2]),
                         (f"K-Means k={k_lv} (sama dengan Louvain)", km_same)]:
        lab = pd.Series(labels, index=order)
        rows.append({
            "Metode": nama, "Spearman_vs_Selisih_IPM": np.nan, "p_Mantel_999": np.nan,
            "Jaccard_Top5_vs_Utama": np.nan, "Jumlah_Kelompok": lab.nunique(),
            "Ukuran_Kelompok": str(sorted(lab.value_counts().tolist(), reverse=True)),
            "Modularity_Louvain": np.nan, "ARI_vs_Louvain_Utama": round(U.ari(lab_main, lab), 4),
        })

    hasil = pd.DataFrame(rows)
    hasil.to_csv(OUT / "hasil_pembanding_metode.csv", index=False)
    print("\n" + hasil.to_string(index=False))
    print("\nCara membaca:")
    print("- Cosine hanya menangkap ARAH pola; Euclidean juga menangkap BESAR ketertinggalan.")
    print("- K-Means dengan ukuran kelompok sangat timpang (mis. 37 vs 1) menandakan K-Means")
    print("  didominasi pencilan, mendukung pilihan graf k-NN + Louvain.")
    print("\nFile hasil:", OUT)


if __name__ == "__main__":
    main()