# ============================================================
# EVALUATION 5 - KEPEKAAN BOBOT (Laporan 5.4 & 5.8)
# Bandingkan tiga skema bobot:
#   1. Entropi (sebelum dikoreksi)  -> bobot utama, Laporan 5.4
#   2. Entropi terkoreksi redundansi -> w(j) * (1 - rata-rata |r(j,k)|), dinormalisasi
#   3. Sama rata (1/9)
# Ukuran perubahan terhadap skema utama:
#   - Jaccard top-5 provinsi kembar
#   - Spearman antar 703 nilai kemiripan
#   - ARI komunitas Louvain
# Harapan laporan: hasil tidak berubah drastis.
# ============================================================

import pandas as pd
from networkx.algorithms.community import modularity
from scipy.stats import spearmanr

import eval_utils as U

OUT = U.out_dir("05_kepekaan_bobot")

NAMA = {
    "entropy": "Entropi (sebelum dikoreksi) [utama]",
    "entropy_corrected": "Entropi terkoreksi redundansi",
    "equal": "Sama rata",
}


def main():
    U.header("EVALUATION 5: KEPEKAAN BOBOT")
    U.check_consistency()

    data, nas = U.load_data(), U.load_national()
    S_main = U.load_similarity()
    lab_main = U.load_communities(S_main.index)
    t_main = U.top_k(S_main)

    rows, bobot = [], {}
    for sch in ["entropy", "entropy_corrected", "equal"]:
        _, w, S = U.similarity_pipeline(data, nas, scheme=sch)
        S = S.loc[S_main.index, S_main.index]
        lab, q = U.louvain_best(U.knn_graph(S))
        bobot[NAMA[sch]] = w.rename(index=U.FEATURE_LABELS)
        rows.append({
            "Skema_Bobot": NAMA[sch],
            "Jaccard_Top5_vs_Utama": round(U.mean_jaccard(t_main, U.top_k(S)), 4),
            "Spearman_Kemiripan_vs_Utama": round(spearmanr(U.upper(S), U.upper(S_main))[0], 4),
            "ARI_Komunitas_vs_Utama": round(U.ari(lab_main, lab), 4),
            "Jumlah_Komunitas": lab.nunique(),
            "Modularity": round(q, 4),
            "Bobot_Terbesar": f"{U.FEATURE_LABELS[w.idxmax()]} ({w.max():.3f})",
            "Porsi_3_Rasio_Fasilitas_Atas": round(w[["Rasio_SMU_SD", "Rasio_SMK_SD", "Rasio_PT_SD"]].sum(), 3),
        })

    hasil = pd.DataFrame(rows)
    tabel_bobot = pd.DataFrame(bobot).round(4)
    hasil.to_csv(OUT / "hasil_kepekaan_bobot.csv", index=False)
    tabel_bobot.to_csv(OUT / "perbandingan_bobot.csv")

    print("\nBobot per skema:")
    print(tabel_bobot.to_string())
    print("\n" + hasil.to_string(index=False))
    print("\nCara membaca: Jaccard/ARI mendekati 1 = hasil tidak peka terhadap bobot. "
          "Nilai jauh di bawah 1 berarti kesimpulan 'provinsi kembar' bergantung pada pilihan bobot.")
    print("\nFile hasil:", OUT)


if __name__ == "__main__":
    main()