# ============================================================
# EVALUATION 8 - RINGKASAN EVALUASI
# - Tabel kriteria keberhasilan (Laporan 2.3) dengan status LOLOS / TIDAK
# - Pertanyaan analitik 3: dimensi apa yang paling membedakan antarprovinsi
# - Jawaban ringkas 4 pertanyaan analitik (Laporan 2.2) -> markdown
# Jalankan SETELAH evaluation 1-7.
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import kruskal

import eval_utils as U

OUT = U.out_dir("08_ringkasan")


def read(folder, file):
    f = U.EVAL_OUT / folder / file
    U.require(f)
    return pd.read_csv(f)


def dimensi_pembeda():
    """Pertanyaan 3: porsi variansi vektor gap berbobot + uji Kruskal-Wallis antar komunitas."""
    G = U.load_gap()[U.FEATURES]
    lab = U.load_communities(G.index)
    w = pd.read_csv(U.FILE_WEIGHT).set_index("Fitur")["Bobot"].loc[U.FEATURES]
    A = G * np.sqrt(w)
    rows = []
    for f in U.FEATURES:
        groups = [G.loc[lab == c, f].to_numpy() for c in sorted(lab.unique())]
        H, p = kruskal(*groups)
        rows.append({"Fitur": U.FEATURE_LABELS[f], "Bobot_Entropi": w[f],
                     "Variansi_Gap_Berbobot": A[f].var(),
                     "Kruskal_H_Antar_Komunitas": H, "p_Kruskal": p})
    df = pd.DataFrame(rows)
    df["Porsi_Variansi"] = df["Variansi_Gap_Berbobot"] / df["Variansi_Gap_Berbobot"].sum()
    return df.sort_values("Porsi_Variansi", ascending=False).round(4).reset_index(drop=True)


def main():
    U.header("EVALUATION 8: RINGKASAN EVALUASI")

    v = read("01_validitas_eksternal", "hasil_validitas_eksternal.csv")
    g = read("02_validasi_geografis", "hasil_validasi_geografis.csv")
    k = read("03_kualitas_komunitas", "ringkasan_kualitas_komunitas.csv").set_index("Metrik")
    s = read("04_stabilitas_fitur", "ringkasan_lofo.csv").iloc[0]
    b = read("05_kepekaan_bobot", "hasil_kepekaan_bobot.csv")
    m = read("06_pembanding_metode", "hasil_pembanding_metode.csv")
    p = read("07_kemasukakalan", "uji_jawa_vs_papua_pegunungan.csv").iloc[0]
    lab_info = pd.read_csv(U.FILE_LABEL) if U.FILE_LABEL.exists() else None

    mod = float(k.loc["Modularity", "Nilai"])
    sil = float(k.loc["Silhouette (jarak = 1 - sim)", "Nilai"])

    def g_row(skenario, uji):
        return g[(g["Skenario"] == skenario) & (g["Uji"].str.startswith(uji))].iloc[0]

    gm, go = g_row("38 provinsi", "Mantel"), g_row("38 provinsi", "Irisan")
    gt = g_row("38 provinsi", "Median jarak")
    pm, po = g_row("Tanpa 6 provinsi Papua", "Mantel"), g_row("Tanpa 6 provinsi Papua", "Irisan")

    kriteria = pd.DataFrame([
        ("Validitas", "Spearman(kemiripan, selisih IPM) negatif & p < 0,05",
         v.iloc[0]["Spearman_r"], v.iloc[0]["p_Mantel_dua_arah"], v.iloc[0]["Status"]),
        ("Validitas", "Spearman(kemiripan, selisih kemiskinan) negatif & p < 0,05",
         v.iloc[1]["Spearman_r"], v.iloc[1]["p_Mantel_dua_arah"], v.iloc[1]["Status"]),
        ("Kualitas kelompok", "Modularity Louvain > 0,3", mod, np.nan, "LOLOS" if mod > 0.3 else "TIDAK LOLOS"),
        ("Kualitas kelompok", "Silhouette > 0", sil, np.nan, "LOLOS" if sil > 0 else "TIDAK LOLOS"),
        ("Stabilitas", "Rata-rata Jaccard top-5 leave-one-feature-out > 0,6",
         s["Rata_rata_Jaccard_seluruh_skenario"], np.nan, s["Status"]),
        ("Stabilitas", f"Skenario terburuk: tanpa {s['Skenario_terburuk']}",
         s["Jaccard_terburuk"], np.nan, "LOLOS" if s["Jaccard_terburuk"] > 0.6 else "DI BAWAH 0,6"),
        ("Kepekaan bobot", "Jaccard top-5 bobot sama rata vs entropi",
         b.iloc[2]["Jaccard_Top5_vs_Utama"], np.nan, "Dilaporkan"),
        ("Kemasukakalan", "Sim rata-rata Jawa/kota besar > sim ke Papua Pegunungan/Tengah",
         p["Rata2_Kemiripan_Antar_Anggota"], np.nan, p["Status"]),
        ("Validasi geografis", "Mantel Spearman(kemiripan, jarak)", gm["Nilai"], gm["p_value"], "Dilaporkan apa adanya"),
        ("Validasi geografis", "Irisan top-5 kembar & top-5 terdekat (acak 13,5%)", go["Nilai"], go["p_value"], "Dilaporkan apa adanya"),
        ("Validasi geografis", "Median jarak ke 5 kembar (km)", gt["Nilai"], gt["p_value"], "Dilaporkan apa adanya"),
        ("Validasi geografis (tanpa Papua)", "Mantel tanpa 6 provinsi Papua", pm["Nilai"], pm["p_value"], "Dilaporkan apa adanya"),
        ("Validasi geografis (tanpa Papua)", "Irisan tanpa Papua", po["Nilai"], po["p_value"], "Dilaporkan apa adanya"),
    ], columns=["Aspek", "Kriteria", "Nilai", "p_value", "Status"])
    kriteria.to_csv(OUT / "kriteria_keberhasilan.csv", index=False)

    dp = dimensi_pembeda()
    dp.to_csv(OUT / "dimensi_pembeda.csv", index=False)

    f = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")
    L = [
        "# Ringkasan Evaluasi", "",
        "## Kriteria keberhasilan (Laporan 2.3)", "",
        kriteria.to_markdown(index=False), "",
        "## Jawaban pertanyaan analitik (Laporan 2.2)", "",
        "**1. Provinsi mana yang paling mirip dengan provinsi tertentu?**  ",
        "Lihat `07_kemasukakalan/top5_kembar_semua_provinsi.csv`.", "",
        "**2. Apakah provinsi yang berdekatan otomatis mirip?**  ",
        f"Mantel Spearman = {f(gm['Nilai'])} (p = {f(gm['p_value'], 4)}); tanpa Papua {f(pm['Nilai'])} "
        f"(p = {f(pm['p_value'], 4)}). Irisan kembar dengan tetangga terdekat {f(go['Nilai']*100, 1)}% "
        f"vs harapan acak 13,5% (p = {f(go['p_value'], 4)}). Median jarak ke kembar {f(gt['Nilai'], 1)} km "
        f"(p = {f(gt['p_value'], 4)}). Bacaan: hubungan jarak secara global "
        f"{U.kekuatan(gm['Nilai'])}; efek tetangga lokal "
        f"{'ada' if go['p_value'] < 0.05 else 'tidak terbukti'}.", "",
        "**3. Dimensi apa yang paling membedakan antarprovinsi?**", "",
        dp.to_markdown(index=False), "",
        f"Tiga fitur teratas menyumbang {f(dp['Porsi_Variansi'].head(3).sum()*100, 1)}% variansi vektor gap berbobot.", "",
        "**4. Apakah kemiripan sejalan dengan kedekatan IPM dan kemiskinan?**  ",
        f"IPM: r = {f(v.iloc[0]['Spearman_r'])}, p = {f(v.iloc[0]['p_Mantel_dua_arah'], 4)} ({v.iloc[0]['Status']}). "
        f"Kemiskinan: r = {f(v.iloc[1]['Spearman_r'])}, p = {f(v.iloc[1]['p_Mantel_dua_arah'], 4)} ({v.iloc[1]['Status']}).", "",
        "## Kepekaan bobot", "", b.to_markdown(index=False), "",
        "## Pembanding metode", "", m.to_markdown(index=False), "",
    ]
    if lab_info is not None:
        L += ["## Komunitas Louvain", ""]
        for _, r in lab_info.iterrows():
            L.append(f"- **Komunitas {r['Community']}** ({r['Jumlah_Provinsi']} provinsi): {r['Label']}. "
                     f"Anggota: {r['Anggota_Provinsi']}")
    (OUT / "ringkasan_evaluasi.md").write_text("\n".join(L), encoding="utf-8")

    print("\nKRITERIA KEBERHASILAN")
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(kriteria.to_string(index=False))
        print("\nDIMENSI PEMBEDA (pertanyaan 3)")
        print(dp.to_string(index=False))
    print("\nFile hasil:", OUT)
    print("1. kriteria_keberhasilan.csv\n2. dimensi_pembeda.csv\n3. ringkasan_evaluasi.md")


if __name__ == "__main__":
    main()