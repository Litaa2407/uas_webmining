import pandas as pd
import networkx as nx

from pathlib import Path
from networkx.algorithms.community import (
    louvain_communities,
    modularity
)


# ============================================================
# 1. KONFIGURASI FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# File graf hasil k-NN
KNN_DIR = (
    BASE_DIR / "output" / "mining" / "03_knn"
)

# Folder output Louvain
LOUVAIN_DIR = (
    BASE_DIR / "output" / "mining" / "04_louvain"
)

# Membuat folder output otomatis
LOUVAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. KONFIGURASI INPUT DAN PARAMETER
# ============================================================

FILE_GRAPH = (
    KNN_DIR / "knn_graph.graphml"
)

# Seed agar hasil dapat direproduksi
RANDOM_SEED = 42

# Louvain bersifat stokastik (urutan node diacak), sehingga satu seed
# bisa berhenti di optimum lokal. Algoritma dijalankan N_RUNS kali
# (seed RANDOM_SEED ... RANDOM_SEED + N_RUNS - 1) dan partisi dengan
# modularity TERTINGGI yang dipakai. Set N_RUNS = 1 untuk perilaku lama.
N_RUNS = 100

# Nama atribut bobot pada edge graf
WEIGHT_ATTRIBUTE = "weight"


# ============================================================
# 3. MEMBACA GRAF K-NN
# ============================================================

def load_knn_graph():

    print("\n" + "=" * 60)
    print("TAHAP 1: MEMBACA GRAF K-NN")
    print("=" * 60)

    print("\nLokasi file:")
    print(FILE_GRAPH)

    if not FILE_GRAPH.exists():

        raise FileNotFoundError(
            f"File graf tidak ditemukan:\n{FILE_GRAPH}"
        )

    # Membaca graf GraphML
    G = nx.read_graphml(FILE_GRAPH)

    # Memastikan graf tidak berarah
    if G.is_directed():

        raise ValueError(
            "Graf harus tidak berarah untuk Louvain."
        )

    # Memastikan jumlah node
    if G.number_of_nodes() != 38:

        raise ValueError(
            f"Jumlah node seharusnya 38, "
            f"tetapi ditemukan {G.number_of_nodes()}."
        )

    # Memastikan graf memiliki edge
    if G.number_of_edges() == 0:

        raise ValueError(
            "Graf tidak memiliki edge."
        )

    # Memeriksa bobot setiap edge
    for u, v, attributes in G.edges(data=True):

        if WEIGHT_ATTRIBUTE not in attributes:

            raise ValueError(
                f"Edge {u}-{v} tidak memiliki atribut weight."
            )

        weight = float(
            attributes[WEIGHT_ATTRIBUTE]
        )

        if weight <= 0:

            raise ValueError(
                f"Bobot edge {u}-{v} harus positif."
            )

        attributes[WEIGHT_ATTRIBUTE] = weight

    print("\nGraf berhasil dibaca.")

    print("Jumlah node:", G.number_of_nodes())

    print("Jumlah edge:", G.number_of_edges())

    print("Graf tidak berarah: PASS")

    print("Bobot edge positif: PASS")

    return G


# ============================================================
# 4. MENJALANKAN LOUVAIN COMMUNITY DETECTION
# ============================================================

def detect_communities(G):

    print("\n" + "=" * 60)
    print("TAHAP 2: LOUVAIN COMMUNITY DETECTION")
    print("=" * 60)

    print("\nParameter:")
    print("Random seed awal:", RANDOM_SEED)
    print("Jumlah run (seed berbeda):", N_RUNS)
    print("Bobot edge:", WEIGHT_ATTRIBUTE)

    # Menjalankan algoritma Louvain dengan beberapa seed
    runs = []
    best_communities, best_q, best_seed = None, -1.0, None

    for seed in range(RANDOM_SEED, RANDOM_SEED + N_RUNS):

        candidate = louvain_communities(
            G,
            weight=WEIGHT_ATTRIBUTE,
            resolution=1,
            seed=seed
        )

        q = modularity(G, candidate, weight=WEIGHT_ATTRIBUTE)

        runs.append({
            "Seed": seed,
            "Jumlah_Komunitas": len(candidate),
            "Modularity": q
        })

        # Lebih besar secara nyata (bukan selisih pembulatan) -> ganti
        if q > best_q + 1e-12:
            best_communities, best_q, best_seed = candidate, q, seed

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(LOUVAIN_DIR / "louvain_seed_runs.csv", index=False)

    print("\nRingkasan hasil seluruh run:")
    print(
        runs_df.round({"Modularity": 4})
        .groupby(["Jumlah_Komunitas", "Modularity"])
        .size()
        .rename("Jumlah_Run")
        .reset_index()
        .sort_values("Modularity", ascending=False)
        .to_string(index=False)
    )
    print(f"\nDipakai seed {best_seed} (modularity tertinggi = {best_q:.4f})")

    communities = best_communities

    # Mengurutkan komunitas berdasarkan
    # nama node terkecil agar label konsisten
    communities = sorted(
        communities,
        key=lambda community: min(
            str(node) for node in community
        )
    )

    print("\nDeteksi komunitas selesai.")

    print(
        "Jumlah komunitas:",
        len(communities)
    )

    return communities


# ============================================================
# 5. MEMBUAT LABEL KOMUNITAS SETIAP PROVINSI
# ============================================================

def create_community_dataframe(
    G,
    communities
):

    print("\n" + "=" * 60)
    print("TAHAP 3: MEMBUAT LABEL KOMUNITAS")
    print("=" * 60)

    records = []

    # Membuat pemetaan node ke komunitas
    community_mapping = {}

    for community_id, members in enumerate(
        communities,
        start=1
    ):

        for province in members:

            community_mapping[province] = community_id

            records.append({

                "Provinsi": province,

                "Community": community_id,

                "Degree": G.degree(province),

                "Weighted_Degree": G.degree(
                    province,
                    weight=WEIGHT_ATTRIBUTE
                )

            })

    # Memastikan seluruh node memiliki komunitas
    if len(community_mapping) != G.number_of_nodes():

        raise ValueError(
            "Tidak semua provinsi mendapatkan label komunitas."
        )

    community_df = pd.DataFrame(records)

    community_df = community_df.sort_values(
        by=["Community", "Provinsi"]
    ).reset_index(drop=True)

    print("Label komunitas berhasil dibuat.")

    print("\nHasil pengelompokan provinsi:")

    print(
        community_df.to_string(index=False)
    )

    return community_df, community_mapping


# ============================================================
# 6. MEMBUAT RINGKASAN KOMUNITAS
# ============================================================

def create_community_summary(
    G,
    communities,
    community_mapping
):

    print("\n" + "=" * 60)
    print("TAHAP 4: RINGKASAN KOMUNITAS")
    print("=" * 60)

    records = []

    for community_id, members in enumerate(
        communities,
        start=1
    ):

        # Subgraf untuk anggota komunitas
        subgraph = G.subgraph(members)

        # Menghitung bobot edge internal
        internal_weights = [
            attributes[WEIGHT_ATTRIBUTE]
            for _, _, attributes in subgraph.edges(
                data=True
            )
        ]

        if internal_weights:

            average_similarity = sum(
                internal_weights
            ) / len(internal_weights)

        else:

            average_similarity = 0.0

        records.append({

            "Community": community_id,

            "Jumlah_Provinsi": len(members),

            "Jumlah_Edge_Internal": subgraph.number_of_edges(),

            "Rata_Rata_Similarity_Internal":
                average_similarity,

            "Anggota_Provinsi": "; ".join(
                sorted(members)
            )

        })

    summary_df = pd.DataFrame(records)

    print("\nRingkasan komunitas:")

    print(
        summary_df.to_string(index=False)
    )

    return summary_df


# ============================================================
# 7. MENGHITUNG MODULARITY
# ============================================================

def calculate_modularity(
    G,
    communities
):

    print("\n" + "=" * 60)
    print("TAHAP 5: EVALUASI MODULARITY AWAL")
    print("=" * 60)

    modularity_score = modularity(
        G,
        communities,
        weight=WEIGHT_ATTRIBUTE
    )

    print("\nNilai modularity:")
    print(modularity_score)

    print("\nJumlah komunitas:")
    print(len(communities))

    return modularity_score


# ============================================================
# 8. MENYIMPAN GRAF DENGAN LABEL KOMUNITAS
# ============================================================

def save_community_graph(
    G,
    community_mapping
):

    print("\n" + "=" * 60)
    print("TAHAP 6: MENYIMPAN GRAF KOMUNITAS")
    print("=" * 60)

    # Menambahkan atribut komunitas ke setiap node
    for node in G.nodes():

        G.nodes[node]["community"] = int(
            community_mapping[node]
        )

    output_file = (
        LOUVAIN_DIR /
        "louvain_graph.graphml"
    )

    nx.write_graphml(
        G,
        output_file
    )

    print("\nGraf dengan label komunitas disimpan:")
    print(output_file)


# ============================================================
# 9. MENYIMPAN SELURUH HASIL LOUVAIN
# ============================================================

def save_results(
    community_df,
    summary_df,
    modularity_score,
    G,
    communities
):

    print("\n" + "=" * 60)
    print("TAHAP 7: MENYIMPAN HASIL LOUVAIN")
    print("=" * 60)

    # File 1: Label komunitas setiap provinsi
    community_file = (
        LOUVAIN_DIR /
        "louvain_communities.csv"
    )

    community_df.to_csv(
        community_file,
        index=False
    )

    print("\nFile komunitas disimpan:")
    print(community_file)

    # File 2: Ringkasan setiap komunitas
    summary_file = (
        LOUVAIN_DIR /
        "louvain_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print("\nFile ringkasan disimpan:")
    print(summary_file)

    # File 3: Metrik Louvain
    metrics = {

        "Jumlah_Provinsi": G.number_of_nodes(),

        "Jumlah_Edge": G.number_of_edges(),

        "Jumlah_Komunitas": len(communities),

        "Modularity": modularity_score,

        "Rata_Rata_Ukuran_Komunitas":
            community_df.groupby("Community").size().mean(),

        "Ukuran_Komunitas_Terkecil":
            community_df.groupby("Community").size().min(),

        "Ukuran_Komunitas_Terbesar":
            community_df.groupby("Community").size().max()

    }

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_file = (
        LOUVAIN_DIR /
        "louvain_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    print("\nFile metrik disimpan:")
    print(metrics_file)

    return metrics_df


# ============================================================
# 10. PROGRAM UTAMA
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("LOUVAIN COMMUNITY DETECTION")
    print("WEB MINING - PENDIDIKAN ANTARPROVINSI")
    print("=" * 60)

    print("\nFolder input:")
    print(KNN_DIR)

    print("\nFolder output:")
    print(LOUVAIN_DIR)

    # Tahap 1: Membaca graf
    G = load_knn_graph()

    # Tahap 2: Deteksi komunitas
    communities = detect_communities(G)

    # Tahap 3: Label komunitas
    community_df, community_mapping = (
        create_community_dataframe(
            G,
            communities
        )
    )

    # Tahap 4: Ringkasan komunitas
    summary_df = create_community_summary(
        G,
        communities,
        community_mapping
    )

    # Tahap 5: Menghitung modularity
    modularity_score = calculate_modularity(
        G,
        communities
    )

    # Tahap 6: Menyimpan graf berlabel
    save_community_graph(
        G,
        community_mapping
    )

    # Tahap 7: Menyimpan hasil
    metrics_df = save_results(
        community_df,
        summary_df,
        modularity_score,
        G,
        communities
    )

    print("\n")

    print("=" * 60)
    print("PROSES LOUVAIN SELESAI")
    print("=" * 60)

    print("\nJumlah provinsi:", G.number_of_nodes())

    print("Jumlah edge:", G.number_of_edges())

    print("Jumlah komunitas:", len(communities))

    print("Modularity:", modularity_score)

    print("\nFile hasil tersimpan di:")
    print(LOUVAIN_DIR)

    print("\nFile yang dihasilkan:")
    print("1. louvain_communities.csv")
    print("2. louvain_summary.csv")
    print("3. louvain_metrics.csv")
    print("4. louvain_graph.graphml")


# ============================================================
# 11. EKSEKUSI PROGRAM
# ============================================================

if __name__ == "__main__":
    main()