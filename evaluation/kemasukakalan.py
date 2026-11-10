# ============================================================
# EVALUATION 7 - KEMASUKAKALAN (Laporan 2.3 & 5.8)
# Pemeriksaan manual pasangan terdekat & terjauh, ditambah uji
# sederhana yang bisa diulang:
#   "provinsi di Pulau Jawa dan kota besar lebih dekat satu sama lain
#    dibanding ke provinsi Papua bagian pegunungan"
# Juga menghasilkan tabel top-5 kembar setiap provinsi untuk dicek manual.
# ============================================================

import numpy as np
import pandas as pd

import eval_utils as U

OUT = U.out_dir("07_kemasukakalan")

JAWA_KOTA_BESAR = ["DKI JAKARTA", "JAWA BARAT", "JAWA TENGAH", "DI YOGYAKARTA",
                   "JAWA TIMUR", "BANTEN", "KEPULAUAN RIAU"]
PAPUA_PEGUNUNGAN = ["PAPUA PEGUNUNGAN", "PAPUA TENGAH"]


def main():
    U.header("EVALUATION 7: KEMASUKAKALAN")
    U.check_consistency()

    S = U.load_similarity()
    D = U.load_distance(S.index)
    lab = U.load_communities(S.index)

    # --- pasangan paling mirip & paling berbeda ---
    iu = np.triu_indices(len(S), k=1)
    pairs = pd.DataFrame({"Provinsi_A": S.index[iu[0]], "Provinsi_B": S.index[iu[1]],
                          "Kemiripan": U.upper(S), "Jarak_km": U.upper(D)})
    top10 = pairs.nlargest(10, "Kemiripan").round(3)
    bot10 = pairs.nsmallest(10, "Kemiripan").round(3)

    # --- uji Jawa/kota besar vs Papua pegunungan ---
    within = U.upper(S.loc[JAWA_KOTA_BESAR, JAWA_KOTA_BESAR]).mean()
    to_pap = S.loc[JAWA_KOTA_BESAR, PAPUA_PEGUNUNGAN].to_numpy().mean()
    lolos = within > to_pap
    uji = pd.DataFrame([{
        "Kelompok": "Jawa + kota besar (" + ", ".join(JAWA_KOTA_BESAR) + ")",
        "Rata2_Kemiripan_Antar_Anggota": round(within, 4),
        "Rata2_Kemiripan_ke_Papua_Pegunungan_Tengah": round(to_pap, 4),
        "Status": "LOLOS" if lolos else "TIDAK LOLOS",
    }])

    # --- top-5 kembar tiap provinsi untuk dicek manual ---
    t = U.top_k(S)
    rows = []
    for p, twins in t.items():
        for r, q in enumerate(twins, 1):
            rows.append({"Provinsi": p, "Peringkat": r, "Kembar": q,
                         "Kemiripan": round(S.loc[p, q], 4), "Jarak_km": round(D.loc[p, q], 1),
                         "Komunitas_Sama": bool(lab[p] == lab[q])})
    top5 = pd.DataFrame(rows)

    top10.to_csv(OUT / "10_pasangan_paling_mirip.csv", index=False)
    bot10.to_csv(OUT / "10_pasangan_paling_berbeda.csv", index=False)
    uji.to_csv(OUT / "uji_jawa_vs_papua_pegunungan.csv", index=False)
    top5.to_csv(OUT / "top5_kembar_semua_provinsi.csv", index=False)

    print("\n10 pasangan paling mirip:\n" + top10.to_string(index=False))
    print("\n10 pasangan paling berbeda:\n" + bot10.to_string(index=False))
    print("\n" + uji.to_string(index=False))
    print(f"\nKembar yang berada di komunitas sama: {top5['Komunitas_Sama'].mean()*100:.1f}%")
    print("\nFile hasil:", OUT)


if __name__ == "__main__":
    main()