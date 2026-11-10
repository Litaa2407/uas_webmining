# ============================================================
# EVALUATION 2 - VALIDASI GEOGRAFIS (Laporan 5.7)
# Pertanyaan analitik 2: apakah provinsi yang berdekatan secara
# geografis otomatis mirip secara pendidikan?
#
# Tiga pengujian:
#  a. Uji Mantel  : Spearman(kemiripan, jarak) atas 703 pasangan,
#                   signifikansi dengan 9.999 permutasi label provinsi.
#  b. Irisan      : berapa dari 5 kembar yang juga 5 provinsi terdekat.
#                   Harapan jika acak = 5/37 = 13,5%.
#  c. Jarak kembar: median jarak ke 5 kembar vs median semua pasangan.
# Seluruh pengujian diulang TANPA 6 provinsi Papua (IQR & bobot
# dihitung ulang dari 32 provinsi; acuan tetap nilai nasional BPS).
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_utils as U

OUT = U.out_dir("02_validasi_geografis")


def topk_index(M, k, largest):
    X = M.astype(float).copy()
    np.fill_diagonal(X, -np.inf if largest else np.inf)
    return np.argsort(-X if largest else X, axis=1, kind="mergesort")[:, :k]


def neighbor_overlap(S, D, k=U.K, n_perm=U.N_PERMUTATIONS):
    n = len(S)
    sim_nb = topk_index(S.to_numpy(), k, True)
    Dm = D.to_numpy()

    def counts(Dmat):
        geo_nb = topk_index(Dmat, k, False)
        return np.array([len(set(sim_nb[i]) & set(geo_nb[i])) for i in range(n)])

    c = counts(Dm)
    obs = c.mean() / k
    rng = np.random.default_rng(U.RANDOM_SEED)
    perm = np.array([counts(Dm[np.ix_(p, p)]).mean() / k
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    geo_nb = topk_index(Dm, k, False)
    per_prov = pd.DataFrame({
        "Irisan": c,
        "Kembar_Top5": [", ".join(S.index[j] for j in sim_nb[i]) for i in range(n)],
        "Terdekat_Top5": [", ".join(S.index[j] for j in geo_nb[i]) for i in range(n)],
    }, index=S.index)
    return {"irisan": obs, "harapan_acak": k / (n - 1),
            "p": float((np.sum(perm >= obs) + 1) / (n_perm + 1))}, per_prov


def twin_distance(S, D, k=U.K, n_perm=U.N_PERMUTATIONS):
    n = len(S)
    nb = topk_index(S.to_numpy(), k, True)
    rows, cols = np.repeat(np.arange(n), k), nb.ravel()
    Dm = D.to_numpy()
    obs = float(np.median(Dm[rows, cols]))
    rng = np.random.default_rng(U.RANDOM_SEED)
    perm = np.array([np.median(Dm[np.ix_(p, p)][rows, cols])
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    return {"median_kembar": obs, "median_semua": float(np.median(U.upper(Dm))),
            "p": float((np.sum(perm <= obs) + 1) / (n_perm + 1))}


def run_tests(S, D, label):
    print(f"\n--- {label} ({len(S)} provinsi) ---")
    m = U.mantel(S, D)
    ov, per_prov = neighbor_overlap(S, D)
    td = twin_distance(S, D)
    print(f"Mantel Spearman = {m['r_spearman']:.4f} ({U.kekuatan(m['r_spearman'])}), p = {m['p_dua_arah']:.4f}")
    print(f"Irisan kembar & terdekat = {ov['irisan']*100:.1f}% (acak {ov['harapan_acak']*100:.1f}%), p = {ov['p']:.4f}")
    print(f"Median jarak kembar = {td['median_kembar']:.1f} km vs semua = {td['median_semua']:.1f} km, p = {td['p']:.4f}")
    rows = [
        {"Skenario": label, "Uji": "Mantel Spearman(kemiripan, jarak)", "Nilai": m["r_spearman"],
         "Pembanding": "0 (tidak ada hubungan)", "p_value": m["p_dua_arah"]},
        {"Skenario": label, "Uji": "Irisan top-5 kembar & top-5 terdekat", "Nilai": ov["irisan"],
         "Pembanding": f"{ov['harapan_acak']:.4f} (harapan acak)", "p_value": ov["p"]},
        {"Skenario": label, "Uji": "Median jarak ke 5 kembar (km)", "Nilai": td["median_kembar"],
         "Pembanding": f"{td['median_semua']:.1f} km (median semua pasangan)", "p_value": td["p"]},
    ]
    return rows, per_prov


def main():
    U.header("EVALUATION 2: VALIDASI GEOGRAFIS")
    U.check_consistency()

    S = U.load_similarity()
    D = U.load_distance(S.index)

    # --- 38 provinsi ---
    rows, per_prov = run_tests(S, D, "38 provinsi")

    # --- tanpa Papua (hitung ulang IQR & bobot dari 32 provinsi) ---
    data, nas = U.load_data(), U.load_national()
    data_np = data.drop(index=U.PAPUA)
    _, _, S_np = U.similarity_pipeline(data_np, nas)
    rows_np, _ = run_tests(S_np, D.loc[S_np.index, S_np.index], "Tanpa 6 provinsi Papua")

    hasil = pd.DataFrame(rows + rows_np)
    hasil["Nilai"] = hasil["Nilai"].round(4)
    hasil["p_value"] = hasil["p_value"].round(4)
    hasil.to_csv(OUT / "hasil_validasi_geografis.csv", index=False)
    per_prov.to_csv(OUT / "irisan_per_provinsi.csv")

    # --- kasus menarik untuk dashboard ---
    iu = np.triu_indices(len(S), k=1)
    df = pd.DataFrame({"Provinsi_A": S.index[iu[0]], "Provinsi_B": S.index[iu[1]],
                       "Kemiripan": U.upper(S), "Jarak_km": U.upper(D)})
    s90 = df["Kemiripan"].quantile(0.90)
    d25, d75 = df["Jarak_km"].quantile([0.25, 0.75])
    mirip_jauh = df[(df.Kemiripan >= s90) & (df.Jarak_km >= d75)].sort_values("Kemiripan", ascending=False)
    dekat_beda = df[df.Jarak_km <= d25].nsmallest(10, "Kemiripan")
    mirip_jauh.round(3).to_csv(OUT / "kasus_mirip_tapi_jauh.csv", index=False)
    dekat_beda.round(3).to_csv(OUT / "kasus_dekat_tapi_beda.csv", index=False)
    print(f"\nMirip tapi jauh (sim >= P90 {s90:.3f}, jarak >= Q3 {d75:.0f} km): {len(mirip_jauh)} pasangan")
    print(mirip_jauh.head(5).round(3).to_string(index=False))
    print(f"\nDekat tapi berbeda (jarak <= Q1 {d25:.0f} km, kemiripan terendah):")
    print(dekat_beda.head(5).round(3).to_string(index=False))

    # --- grafik ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["Jarak_km"], df["Kemiripan"], s=10, alpha=0.5, color=U.BLUE, edgecolors="none")
    ax.set_xlabel("Jarak antar ibu kota (km, Haversine)")
    ax.set_ylabel("Weighted cosine similarity")
    r0 = hasil.iloc[0]
    ax.set_title(f"Kemiripan vs jarak: Mantel Spearman {r0['Nilai']:.3f}, p {r0['p_value']:.4f}", loc="left")
    fig.savefig(OUT / "scatter_kemiripan_vs_jarak.png")
    plt.close(fig)

    print("\n" + hasil.to_string(index=False))
    print("\nFile hasil:", OUT)


if __name__ == "__main__":
    main()