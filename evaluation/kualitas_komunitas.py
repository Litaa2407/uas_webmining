# ============================================================
# EVALUATION 3 - KUALITAS KOMUNITAS (Laporan 2.3 & 5.8)
# Kriteria: modularity Louvain > 0,3 dan silhouette > 0
#           (silhouette memakai jarak = 1 - kemiripan).
# Tambahan: kestabilan Louvain antar seed, karena Louvain stokastik.
#   - ARI rata-rata antar partisi dari seed berbeda
#   - Matriks ko-asosiasi: berapa persen seed menempatkan dua provinsi
#     di komunitas yang sama. Provinsi dengan ko-asosiasi rendah ke
#     komunitasnya sendiri = provinsi "perbatasan" yang labelnya goyah.
# ============================================================

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

import eval_utils as U

OUT = U.out_dir("03_kualitas_komunitas")


def main():
    U.header("EVALUATION 3: KUALITAS KOMUNITAS LOUVAIN")
    U.check_consistency()

    S = U.load_similarity()
    lab = U.load_communities(S.index)
    Gr = U.knn_graph(S)

    # --- modularity partisi terpilih ---
    part = [set(lab[lab == c].index) for c in sorted(lab.unique())]
    q = modularity(Gr, part, weight="weight")

    # --- silhouette (jarak = 1 - sim) ---
    Dm = (1 - S.to_numpy()).clip(min=0)
    np.fill_diagonal(Dm, 0)
    sil = silhouette_score(Dm, lab.to_numpy(), metric="precomputed")
    sil_prov = pd.Series(silhouette_samples(Dm, lab.to_numpy(), metric="precomputed"), index=S.index)

    print(f"Jumlah komunitas: {lab.nunique()}")
    print(f"Modularity      : {q:.4f}  (kriteria > 0,3 -> {'LOLOS' if q > 0.3 else 'TIDAK LOLOS'})")
    print(f"Silhouette      : {sil:.4f}  (kriteria > 0   -> {'LOLOS' if sil > 0 else 'TIDAK LOLOS'})")

    # --- stabilitas antar seed ---
    nodes = list(S.index)
    idx = {n: i for i, n in enumerate(nodes)}
    runs, co = [], np.zeros((len(nodes), len(nodes)))
    for seed in range(U.RANDOM_SEED, U.RANDOM_SEED + U.N_LOUVAIN_RUNS):
        p = louvain_communities(Gr, weight="weight", resolution=1, seed=seed)
        l = np.empty(len(nodes), dtype=int)
        for cid, comm in enumerate(p):
            for n in comm:
                l[idx[n]] = cid
        co += (l[:, None] == l[None, :])
        runs.append((seed, len(p), modularity(Gr, p, weight="weight"), l))
    co /= len(runs)

    aris = [adjusted_rand_score(a[3], b[3]) for a, b in itertools.combinations(runs, 2)]
    ari_vs_best = [adjusted_rand_score(lab.to_numpy(), r[3]) for r in runs]
    n_kom = pd.Series([r[1] for r in runs]).value_counts().sort_index()
    print(f"\nStabilitas {len(runs)} seed:")
    print("  Distribusi jumlah komunitas:", {int(a): int(b) for a, b in n_kom.items()})
    print(f"  ARI rata-rata antar seed    : {np.mean(aris):.4f}")
    print(f"  ARI rata-rata vs partisi terpilih: {np.mean(ari_vs_best):.4f}")

    co_df = pd.DataFrame(co, index=nodes, columns=nodes)
    # ko-asosiasi tiap provinsi dengan anggota komunitasnya sendiri
    own = pd.Series({p: co_df.loc[p, lab[lab == lab[p]].index.drop(p)].mean()
                     if (lab == lab[p]).sum() > 1 else 1.0 for p in nodes})

    per_prov = pd.DataFrame({
        "Community": lab,
        "Silhouette": sil_prov.round(4),
        "Ko_asosiasi_dengan_komunitas_sendiri": own.round(3),
    }).sort_values(["Community", "Silhouette"])
    per_prov["Catatan"] = np.where(
        (per_prov["Silhouette"] < 0) | (per_prov["Ko_asosiasi_dengan_komunitas_sendiri"] < 0.7),
        "perbatasan / label kurang stabil", "")

    ringkasan = pd.DataFrame([
        {"Metrik": "Jumlah komunitas", "Nilai": lab.nunique(), "Kriteria": "-", "Status": "-"},
        {"Metrik": "Modularity", "Nilai": round(q, 4), "Kriteria": "> 0,3",
         "Status": "LOLOS" if q > 0.3 else "TIDAK LOLOS"},
        {"Metrik": "Silhouette (jarak = 1 - sim)", "Nilai": round(sil, 4), "Kriteria": "> 0",
         "Status": "LOLOS" if sil > 0 else "TIDAK LOLOS"},
        {"Metrik": "Provinsi dengan silhouette negatif", "Nilai": int((sil_prov < 0).sum()),
         "Kriteria": "-", "Status": "-"},
        {"Metrik": f"ARI rata-rata antar {len(runs)} seed", "Nilai": round(float(np.mean(aris)), 4),
         "Kriteria": "mendekati 1 = stabil", "Status": "-"},
        {"Metrik": "ARI rata-rata seed vs partisi terpilih", "Nilai": round(float(np.mean(ari_vs_best)), 4),
         "Kriteria": "-", "Status": "-"},
        {"Metrik": "Distribusi jumlah komunitas antar seed",
         "Nilai": "; ".join(f"{k} komunitas: {v} run" for k, v in n_kom.items()), "Kriteria": "-", "Status": "-"},
    ])

    ringkasan.to_csv(OUT / "ringkasan_kualitas_komunitas.csv", index=False)
    per_prov.to_csv(OUT / "per_provinsi_silhouette_koasosiasi.csv")
    co_df.round(3).to_csv(OUT / "matriks_koasosiasi_seed.csv")

    # --- grafik silhouette per provinsi ---
    sp = per_prov.sort_values("Silhouette")
    colors = [U.RED if v < 0 else U.BLUE for v in sp["Silhouette"]]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(sp.index, sp["Silhouette"], color=colors, height=0.7)
    ax.axvline(0, color=U.INK, lw=1)
    ax.set_xlabel("Silhouette (jarak = 1 - kemiripan)")
    ax.set_title(f"Silhouette per provinsi (rata-rata {sil:.3f})", loc="left")
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(OUT / "silhouette_per_provinsi.png")
    plt.close(fig)

    print("\n" + ringkasan.to_string(index=False))
    tidak_stabil = per_prov[per_prov["Catatan"] != ""]
    if len(tidak_stabil):
        print("\nProvinsi perbatasan:")
        print(tidak_stabil.to_string())
    print("\nFile hasil:", OUT)


if __name__ == "__main__":
    main()