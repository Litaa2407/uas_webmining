# ============================================================
# MENJALANKAN SELURUH TAHAP: PREPROCESSING -> MINING -> EVALUATION
#
#   python run_all.py                 # semua tahap
#   python run_all.py preprocessing   # hanya satu tahap
#   python run_all.py mining
#   python run_all.py evaluation
# ============================================================

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TAHAP = {
    "preprocessing": [
        "preprocess.py",
        "gap_vector.py",
        "quality_assessment.py",
        "preprocess_koordinat.py",
    ],
    "mining": [
        "mining/entropy_weight.py",
        "mining/weighted_cosine.py",
        "mining/knn_graph.py",
        "mining/louvain_community.py",
        "mining/community_profile.py",
        "mining/haversine_distance.py",
    ],
    "evaluation": [
        "evaluation/validitas_eksternal.py",
        "evaluation/validasi_geografis.py",
        "evaluation/kualitas_komunitas.py",
        "evaluation/stabilitas_fitur.py",
        "evaluation/kepekaan_bobot.py",
        "evaluation/pembanding_metode.py",
        "evaluation/kemasukakalan.py",
        "evaluation/ringkasan_evaluasi.py",
    ],
}


def main():
    pilih = sys.argv[1:] or list(TAHAP)
    for t in pilih:
        if t not in TAHAP:
            sys.exit(f"Tahap tidak dikenal: {t}. Pilihan: {', '.join(TAHAP)}")

    logdir = ROOT / "logs"
    logdir.mkdir(exist_ok=True)

    for t in pilih:
        print(f"\n######## TAHAP {t.upper()} ########")
        for script in TAHAP[t]:
            start = time.time()
            log = logdir / (Path(script).stem + ".log")
            with open(log, "w", encoding="utf-8") as fh:
                r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT,
                                   stdout=fh, stderr=subprocess.STDOUT)
            status = "OK" if r.returncode == 0 else "GAGAL"
            print(f"  [{status}] {script:<40} {time.time() - start:5.1f} detik  (log: logs/{log.name})")
            if r.returncode != 0:
                print(f"\nBerhenti. Lihat pesan error di logs/{log.name}:\n")
                print(log.read_text(encoding="utf-8")[-2000:])
                sys.exit(1)

    if "evaluation" in pilih:
        print("\nRingkasan akhir: evaluation/output/08_ringkasan/ringkasan_evaluasi.md")
    print("Selesai.")


if __name__ == "__main__":
    main()