# ============================================================
# EVALUATION 4 - STABILITAS LEAVE-ONE-FEATURE-OUT (Laporan 2.3 & 5.8)
# Satu fitur dihilangkan bergantian (9 skenario). Untuk tiap skenario:
#   - IQR, gap, bobot entropi, dan kemiripan dihitung ulang dengan 8 fitur
#   - top-5 provinsi kembar dibandingkan dengan model lengkap (Jaccard)
# Kriteria: rata-rata Jaccard > 0,6
# Catatan: laporkan juga skenario TERBURUK, bukan hanya rata-rata.
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eval_utils as U

OUT = U.out_dir("04_stabilitas_fitur")


def main():
    U.header("EVALUATION 4: STABILITAS LEAVE-ONE-FEATURE-OUT")
    U.check_consistency()

    data, nas = U.load_data(), U.load_national()
    S_full = U.load_similarity()
    t_full = U.top_k(S_full)

    rows, per_prov = [], {}
    for f in U.FEATURES:
        feats = [x for x in U.FEATURES if x != f]
        _, w, S = U.similarity_pipeline(data, nas, feats)
        t = U.top_k(S.loc[S_full.index, S_full.index])
        jac = pd.Series({p: U.jaccard(t_full[p], t[p]) for p in t_full})
        per_prov[U.FEATURE_LABELS[f]] = jac
        rows.append({
            "Fitur_Dihapus": U.FEATURE_LABELS[f],
            "Jaccard_Top5_Rata2": round(jac.mean(), 4),
            "Jaccard_Top5_Min": round(jac.min(), 4),
            "Provinsi_Paling_Berubah": jac.idxmin(),
            "Bobot_Terbesar_Sisa": f"{U.FEATURE_LABELS[w.idxmax()]} ({w.max():.3f})",
            "Status": "LOLOS" if jac.mean() > 0.6 else "DI BAWAH 0,6",
        })
        print(f"Tanpa {U.FEATURE_LABELS[f]:<24}: Jaccard rata-rata {jac.mean():.3f}")

    hasil = pd.DataFrame(rows)
    rata = hasil["Jaccard_Top5_Rata2"].mean()
    terburuk = hasil.loc[hasil["Jaccard_Top5_Rata2"].idxmin()]
    print(f"\nRata-rata keseluruhan = {rata:.4f} (kriteria > 0,6 -> {'LOLOS' if rata > 0.6 else 'TIDAK LOLOS'})")
    print(f"Skenario terburuk: tanpa {terburuk['Fitur_Dihapus']} ({terburuk['Jaccard_Top5_Rata2']:.3f})")

    hasil.to_csv(OUT / "hasil_lofo.csv", index=False)
    pd.DataFrame(per_prov).round(3).to_csv(OUT / "jaccard_per_provinsi.csv")
    pd.DataFrame([{"Rata_rata_Jaccard_seluruh_skenario": round(rata, 4),
                   "Skenario_terburuk": terburuk["Fitur_Dihapus"],
                   "Jaccard_terburuk": terburuk["Jaccard_Top5_Rata2"],
                   "Status": "LOLOS" if rata > 0.6 else "TIDAK LOLOS"}]).to_csv(
        OUT / "ringkasan_lofo.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [U.RED if v < 0.6 else U.BLUE for v in hasil["Jaccard_Top5_Rata2"]]
    ax.barh(hasil["Fitur_Dihapus"], hasil["Jaccard_Top5_Rata2"], color=colors, height=0.6)
    ax.axvline(0.6, color=U.INK, ls="--", lw=1)
    for i, v in enumerate(hasil["Jaccard_Top5_Rata2"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Rata-rata Jaccard top-5 vs model lengkap")
    ax.set_title(f"Leave-one-feature-out (rata-rata {rata:.3f}, ambang 0,6)", loc="left")
    fig.savefig(OUT / "lofo_jaccard.png")
    plt.close(fig)
    print("\nFile hasil:", OUT)


if __name__ == "__main__":
    main()