# ============================================================
# WEB MINING
# EVALUATION - FUNGSI BERSAMA
# Dipakai oleh semua skrip di folder evaluation/.
# Rumus gap, bobot entropi, dan weighted cosine di sini SAMA PERSIS
# dengan gap_vector.py, mining/entropy_weight.py, mining/weighted_cosine.py
# (dicek otomatis oleh check_consistency()). Rumus perlu ditulis ulang
# di sini karena evaluasi harus menghitung ulang kemiripan untuk
# skenario lain: tanpa Papua, satu fitur dihapus, bobot berbeda.
# ============================================================

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score


# ============================================================
# 1. LOKASI FILE
# ============================================================

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
PRE_DIR = PROJECT_DIR / "output_preprocessing"
MINING_OUT = PROJECT_DIR / "mining" / "output" / "mining"
EVAL_OUT = EVAL_DIR / "output"

FILE_DATA = PRE_DIR / "01_dataset_pendidikan_38_provinsi.csv"
FILE_VALIDASI = PRE_DIR / "02_dataset_validasi.csv"
FILE_NASIONAL = PRE_DIR / "03_referensi_nasional.csv"
FILE_GAP = PRE_DIR / "06_gap_vector_robust.csv"
FILE_KOORDINAT = PRE_DIR / "11_koordinat_ibu_kota.csv"
FILE_WEIGHT = MINING_OUT / "01_entropy_weight" / "entropy_weights.csv"
FILE_SIMILARITY = MINING_OUT / "02_weighted_cosine" / "similarity_matrix.csv"
FILE_KNN = MINING_OUT / "03_knn" / "knn_neighbors.csv"
FILE_COMMUNITY = MINING_OUT / "04_louvain" / "louvain_communities.csv"
FILE_SEED_RUNS = MINING_OUT / "04_louvain" / "louvain_seed_runs.csv"
FILE_DISTANCE = MINING_OUT / "05_jarak_geografis" / "distance_matrix_km.csv"
FILE_LABEL = MINING_OUT / "06_profil_komunitas" / "label_komunitas.csv"


# ============================================================
# 2. PARAMETER (samakan dengan mining)
# ============================================================

FEATURES = [
    "RLS", "APS_7_12", "APS_13_15", "APS_16_18", "APS_19_23",
    "Rasio_SMP_SD", "Rasio_SMU_SD", "Rasio_SMK_SD", "Rasio_PT_SD",
]
FEATURE_LABELS = {
    "RLS": "Rata-rata lama sekolah", "APS_7_12": "APS 7-12", "APS_13_15": "APS 13-15",
    "APS_16_18": "APS 16-18", "APS_19_23": "APS 19-23",
    "Rasio_SMP_SD": "Rasio fasilitas SMP/SD", "Rasio_SMU_SD": "Rasio fasilitas SMU/SD",
    "Rasio_SMK_SD": "Rasio fasilitas SMK/SD", "Rasio_PT_SD": "Rasio fasilitas PT/SD",
}
K = 5                    # k pada graf k-NN dan top-k provinsi kembar
RANDOM_SEED = 42
N_LOUVAIN_RUNS = 100     # sama dengan mining/louvain_community.py
N_PERMUTATIONS = 9999    # uji permutasi / Mantel (Laporan 5.7)

PAPUA = ["PAPUA", "PAPUA BARAT", "PAPUA BARAT DAYA",
         "PAPUA SELATAN", "PAPUA TENGAH", "PAPUA PEGUNUNGAN"]


def out_dir(name):
    d = EVAL_OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def require(*files):
    for f in files:
        if not Path(f).exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {f}\n"
                "Jalankan tahap preprocessing dan mining terlebih dahulu (lihat run_all.py)."
            )


# ============================================================
# 3. MEMBACA HASIL PREPROCESSING & MINING
# ============================================================

def load_data():
    require(FILE_DATA)
    return pd.read_csv(FILE_DATA).set_index("Provinsi")


def load_validation():
    require(FILE_VALIDASI)
    return pd.read_csv(FILE_VALIDASI).set_index("Provinsi")


def load_national():
    require(FILE_NASIONAL)
    n = pd.read_csv(FILE_NASIONAL)
    return pd.Series(n["Nilai_Nasional"].values, index=n["Fitur"])


def load_similarity():
    require(FILE_SIMILARITY)
    return pd.read_csv(FILE_SIMILARITY, index_col=0)


def load_distance(order):
    require(FILE_DISTANCE)
    D = pd.read_csv(FILE_DISTANCE, index_col=0)
    return D.loc[order, order]


def load_communities(order):
    require(FILE_COMMUNITY)
    return pd.read_csv(FILE_COMMUNITY).set_index("Provinsi")["Community"].loc[order]


def load_gap():
    require(FILE_GAP)
    return pd.read_csv(FILE_GAP).set_index("Provinsi")


# ============================================================
# 4. RUMUS INTI (sama dengan preprocessing & mining)
# ============================================================

def gap_robust(data, nasional, features=FEATURES):
    """g(i,j) = (x(i,j) - nasional(j)) / IQR(j), IQR dari provinsi yang dipakai."""
    X = data[features]
    iqr = X.quantile(0.75) - X.quantile(0.25)
    return (X - nasional[features]) / iqr


def entropy_weights(data, features=FEATURES):
    """Entropy Weight Method (Laporan 5.4) pada data mentah min-max."""
    X = data[features]
    Xn = (X - X.min()) / (X.max() - X.min())
    P = Xn / Xn.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    e = -plogp.sum(axis=0) / np.log(len(X))
    d = 1 - e
    return pd.Series(d / d.sum(), index=features)


def redundancy_corrected_weights(data, features=FEATURES):
    """
    Bobot entropi terkoreksi redundansi:
      w'(j) = w(j) * (1 - rata-rata |r(j,k)|, k != j), lalu dinormalisasi.
    Fitur yang informasinya tumpang tindih (APS 7-12 vs 13-15, r = 0,96)
    mendapat bobot lebih kecil.
    """
    w = entropy_weights(data, features)
    R = data[features].corr().abs().to_numpy()
    uniq = 1 - (R.sum(axis=1) - 1) / (len(features) - 1)
    wc = w * uniq
    return wc / wc.sum()


def equal_weights(features=FEATURES):
    return pd.Series(1 / len(features), index=features)


def weighted_cosine(G, w):
    """sim(a,b) = sum w g_a g_b / (||g_a||_w ||g_b||_w)."""
    A = G.to_numpy(dtype=float) * np.sqrt(w.loc[G.columns].to_numpy(dtype=float))
    U = A / np.linalg.norm(A, axis=1, keepdims=True)
    S = np.clip(U @ U.T, -1, 1)
    np.fill_diagonal(S, 1.0)
    return pd.DataFrame(S, index=G.index, columns=G.index)


def weighted_euclidean(G, w):
    A = G.to_numpy(dtype=float) * np.sqrt(w.loc[G.columns].to_numpy(dtype=float))
    D = np.sqrt(((A[:, None, :] - A[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(D, index=G.index, columns=G.index)


def similarity_pipeline(data, nasional, features=FEATURES, scheme="entropy"):
    G = gap_robust(data, nasional, features)
    if scheme == "entropy":
        w = entropy_weights(data, features)
    elif scheme == "entropy_corrected":
        w = redundancy_corrected_weights(data, features)
    elif scheme == "equal":
        w = equal_weights(features)
    else:
        raise ValueError(scheme)
    return G, w, weighted_cosine(G, w)


def top_k(S, k=K, larger_is_closer=True):
    """k provinsi paling mirip (tanpa diri sendiri)."""
    out = {}
    for p in S.index:
        row = S.loc[p].drop(p).sort_values(ascending=not larger_is_closer, kind="mergesort")
        out[p] = list(row.index[:k])
    return out


def knn_graph(S, k=K):
    """Sama dengan mining/knn_graph.py: union k-NN, hanya similarity positif."""
    Gr = nx.Graph()
    Gr.add_nodes_from(S.index)
    for p in S.index:
        row = S.loc[p].drop(p).sort_values(ascending=False, kind="mergesort")
        for q, s in row.iloc[:k].items():
            if s > 0 and not Gr.has_edge(p, q):
                Gr.add_edge(p, q, weight=float(s))
    return Gr


def louvain_best(Gr, n_runs=N_LOUVAIN_RUNS, seed=RANDOM_SEED):
    """Sama dengan mining/louvain_community.py: ambil modularity tertinggi dari n_runs seed."""
    best, best_q = None, -1
    for s in range(seed, seed + n_runs):
        part = louvain_communities(Gr, weight="weight", resolution=1, seed=s)
        q = modularity(Gr, part, weight="weight")
        if q > best_q + 1e-12:
            best, best_q = part, q
    labels = pd.Series({n: i for i, c in enumerate(best, 1) for n in c}).loc[list(Gr.nodes)]
    return labels, best_q


# ============================================================
# 5. STATISTIK
# ============================================================

def upper(M):
    M = np.asarray(M)
    return M[np.triu_indices(M.shape[0], k=1)]


def _rank_matrix(M):
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)
    R = np.zeros((n, n))
    R[iu] = rankdata(M[iu])
    return R + R.T


def mantel(A, B, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Uji Mantel dengan korelasi Spearman.
    Label provinsi pada matriks B diacak n_perm kali, karena 703 pasangan
    tidak saling bebas (satu provinsi muncul di 37 pasangan).
    p dua arah = (#|r_perm| >= |r_obs| + 1) / (n_perm + 1)
    """
    B = B.loc[A.index, A.index]
    n = len(A)
    iu = np.triu_indices(n, k=1)
    a = _rank_matrix(A.to_numpy(dtype=float))[iu]
    a = (a - a.mean()) / a.std()
    RB = _rank_matrix(B.to_numpy(dtype=float))
    b = RB[iu]
    r_obs = float(np.mean(a * (b - b.mean()) / b.std()))
    rng = np.random.default_rng(seed)
    r_perm = np.empty(n_perm)
    for t in range(n_perm):
        p = rng.permutation(n)
        bp = RB[np.ix_(p, p)][iu]
        r_perm[t] = np.mean(a * (bp - bp.mean()) / bp.std())
    return {
        "r_spearman": r_obs,
        "p_dua_arah": float((np.sum(np.abs(r_perm) >= abs(r_obs)) + 1) / (n_perm + 1)),
        "p_satu_arah_negatif": float((np.sum(r_perm <= r_obs) + 1) / (n_perm + 1)),
        "n_pasangan": int(len(a)),
        "n_permutasi": int(n_perm),
    }


def abs_diff_matrix(s):
    v = s.to_numpy(dtype=float)
    return pd.DataFrame(np.abs(v[:, None] - v[None, :]), index=s.index, columns=s.index)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def mean_jaccard(t1, t2):
    return float(np.mean([jaccard(t1[p], t2[p]) for p in t1]))


def ari(l1, l2):
    return float(adjusted_rand_score(l1.loc[l2.index], l2))


def kekuatan(r):
    a = abs(r)
    return ("sangat lemah" if a < 0.1 else "lemah" if a < 0.3 else "sedang" if a < 0.5 else "kuat")


# ============================================================
# 6. CEK KONSISTENSI DENGAN HASIL MINING
# ============================================================

def check_consistency():
    """Hitung ulang kemiripan dari dataset 01 dan bandingkan dengan similarity_matrix.csv."""
    data, nas = load_data(), load_national()
    S_file = load_similarity()
    _, w, S = similarity_pipeline(data, nas)
    diff = float((S.loc[S_file.index, S_file.index] - S_file).abs().to_numpy().max())
    w_file = pd.read_csv(FILE_WEIGHT).set_index("Fitur")["Bobot"].loc[FEATURES]
    wdiff = float((w - w_file).abs().max())
    ok = diff < 1e-9 and wdiff < 1e-9
    print(f"Cek konsistensi dengan hasil mining: selisih similarity maks = {diff:.2e}, "
          f"selisih bobot maks = {wdiff:.2e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise ValueError("Rumus evaluasi tidak sama dengan hasil mining. Jalankan ulang mining.")
    return ok


# ============================================================
# 7. GAYA GRAFIK
# ============================================================

BLUE, ORANGE, GREEN, RED, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#0b0b0b", "#e1e0d9"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})