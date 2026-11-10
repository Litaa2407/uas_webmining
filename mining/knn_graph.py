
import pandas as pd
import numpy as np
import networkx as nx

from pathlib import Path


# ============================================================
# 1. KONFIGURASI FOLDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

COSINE_DIR = (
    BASE_DIR / "output" / "mining" /
    "02_weighted_cosine"
)

KNN_DIR = (
    BASE_DIR / "output" / "mining" /
    "03_knn"
)

KNN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. KONFIGURASI INPUT DAN PARAMETER
# ============================================================

FILE_SIMILARITY = (
    COSINE_DIR / "similarity_matrix.csv"
)

K = 5

# Hanya menggunakan sisi dengan similarity positif
POSITIVE_ONLY = True


# ============================================================
# 3. MEMBACA MATRIKS SIMILARITY
# ============================================================

def load_similarity_matrix():

    print("\n" + "=" * 60)
    print("TAHAP 1: MEMBACA MATRIKS SIMILARITY")
    print("=" * 60)

    print("\nLokasi file:")
    print(FILE_SIMILARITY)

    if not FILE_SIMILARITY.exists():

        raise FileNotFoundError(
            f"Matriks similarity tidak ditemukan:\n"
            f"{FILE_SIMILARITY}"
        )

    df = pd.read_csv(
        FILE_SIMILARITY,
        index_col=0
    )

    # Memastikan matriks persegi
    if df.shape[0] != df.shape[1]:

        raise ValueError(
            "Matriks similarity harus berbentuk persegi."
        )

    # Memastikan nama baris dan kolom sama
    if list(df.index) != list(df.columns):

        raise ValueError(
            "Nama provinsi pada baris dan kolom "
            "tidak sama atau urutannya berbeda."
        )

    # Memastikan tidak ada provinsi duplikat
    if df.index.duplicated().any():

        raise ValueError(
            "Terdapat nama provinsi duplikat."
        )

    # Memastikan jumlah provinsi
    if len(df) != 38:

        raise ValueError(
            f"Seharusnya terdapat 38 provinsi, "
            f"tetapi ditemukan {len(df)}."
        )

    # Mengubah data menjadi numerik
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if df.isnull().any().any():

        raise ValueError(
            "Terdapat nilai kosong atau non-numerik."
        )

    matrix = df.to_numpy(dtype=float)

    if not np.isfinite(matrix).all():

        raise ValueError(
            "Matriks mengandung nilai infinity."
        )

    # Memastikan matriks simetris
    if not np.allclose(
        matrix,
        matrix.T,
        atol=1e-9
    ):

        raise ValueError(
            "Matriks similarity tidak simetris."
        )

    # Memastikan diagonal bernilai 1
    if not np.allclose(
        np.diag(matrix),
        1.0,
        atol=1e-9
    ):

        raise ValueError(
            "Diagonal matriks similarity "
            "seharusnya bernilai 1."
        )

    print("\nMatriks berhasil dibaca.")
    print("Jumlah provinsi:", len(df))
    print("Ukuran matriks:", df.shape)
    print("Pemeriksaan matriks: PASS")

    return df


# ============================================================
# 4. MENENTUKAN 5 TETANGGA TERDEKAT
# ============================================================

def calculate_knn(similarity_df):

    print("\n" + "=" * 60)
    print("TAHAP 2: MENENTUKAN TETANGGA TERDEKAT")
    print("=" * 60)

    provinces = similarity_df.index.tolist()

    matrix = similarity_df.to_numpy()

    n = len(provinces)

    knn_records = []

    neighbor_dict = {}

    for i in range(n):

        province = provinces[i]

        # Mengambil similarity provinsi ke provinsi lainnya
        similarities = matrix[i].copy()

        # Menghilangkan similarity dengan dirinya sendiri
        similarities[i] = -np.inf

        # Mengurutkan similarity tertinggi
        sorted_indices = np.argsort(
            similarities
        )[::-1]

        selected_neighbors = []

        # Memilih lima tetangga
        for j in sorted_indices:

            similarity = similarities[j]

            if POSITIVE_ONLY and similarity <= 0:
                continue

            if not np.isfinite(similarity):
                continue

            selected_neighbors.append(j)

            if len(selected_neighbors) == K:
                break

        neighbor_dict[province] = []

        for rank, j in enumerate(
            selected_neighbors,
            start=1
        ):

            neighbor = provinces[j]

            similarity = matrix[i, j]

            neighbor_dict[province].append(
                neighbor
            )

            knn_records.append({

                "Provinsi": province,

                "Tetangga": neighbor,

                "Rank": rank,

                "Similarity": similarity

            })

    knn_df = pd.DataFrame(knn_records)

    print("Penentuan tetangga terdekat selesai.")

    print("\nJumlah hubungan terarah:", len(knn_df))

    print("\nContoh lima tetangga pertama:")

    print(
        knn_df.head(10).to_string(index=False)
    )

    return knn_df, neighbor_dict


# ============================================================
# 5. MEMBENTUK GRAF K-NN
# ============================================================

def build_knn_graph(
    similarity_df,
    neighbor_dict
):

    print("\n" + "=" * 60)
    print("TAHAP 3: MEMBENTUK GRAF K-NN")
    print("=" * 60)

    G = nx.Graph()

    provinces = similarity_df.index.tolist()

    # Menambahkan seluruh provinsi sebagai node
    G.add_nodes_from(provinces)

    # Membentuk graf Union k-NN
    # Sisi dibuat apabila salah satu node memilih node lainnya.
    for province, neighbors in neighbor_dict.items():

        for neighbor in neighbors:

            similarity = float(
                similarity_df.loc[
                    province,
                    neighbor
                ]
            )

            if similarity <= 0:
                continue

            # Graf tidak berarah otomatis menghindari
            # sisi duplikat.
            if not G.has_edge(
                province,
                neighbor
            ):

                G.add_edge(
                    province,
                    neighbor,
                    weight=similarity
                )

    print("Graf k-NN berhasil dibentuk.")

    print("\nJumlah node:", G.number_of_nodes())

    print("Jumlah edge:", G.number_of_edges())

    print("Graf tidak berarah: PASS")

    return G


# ============================================================
# 6. MEMBUAT DAFTAR EDGE GRAF
# ============================================================

def create_edge_dataframe(G):

    print("\n" + "=" * 60)
    print("TAHAP 4: MEMBUAT DAFTAR EDGE")
    print("=" * 60)

    edge_records = []

    for source, target, attributes in G.edges(
        data=True
    ):

        edge_records.append({

            "Provinsi_A": source,

            "Provinsi_B": target,

            "Similarity": attributes["weight"]

        })

    edge_df = pd.DataFrame(edge_records)

    if not edge_df.empty:

        edge_df = edge_df.sort_values(
            by="Similarity",
            ascending=False
        ).reset_index(drop=True)

    print("Daftar edge selesai dibuat.")

    print("\nJumlah edge:", len(edge_df))

    print("\nLima edge teratas:")

    print(
        edge_df.head(5).to_string(index=False)
    )

    return edge_df


# ============================================================
# 7. MEMBUAT INFORMASI DEGREE NODE
# ============================================================

def create_node_dataframe(G):

    print("\n" + "=" * 60)
    print("TAHAP 5: INFORMASI DEGREE NODE")
    print("=" * 60)

    node_records = []

    for node in G.nodes():

        node_records.append({

            "Provinsi": node,

            "Degree": G.degree(node),

            "Weighted_Degree": G.degree(
                node,
                weight="weight"
            )

        })

    node_df = pd.DataFrame(node_records)

    node_df = node_df.sort_values(
        by="Degree",
        ascending=False
    ).reset_index(drop=True)

    print("Informasi degree selesai dibuat.")

    print("\nLima node dengan degree tertinggi:")

    print(
        node_df.head(5).to_string(index=False)
    )

    return node_df


# ============================================================
# 8. MENYIMPAN GRAF K-NN
# ============================================================

def save_knn_results(
    knn_df,
    edge_df,
    node_df,
    G
):

    print("\n" + "=" * 60)
    print("TAHAP 6: MENYIMPAN HASIL K-NN")
    print("=" * 60)

    # File 1: Daftar tetangga setiap provinsi
    knn_file = (
        KNN_DIR / "knn_neighbors.csv"
    )

    knn_df.to_csv(
        knn_file,
        index=False
    )

    print("\nFile tetangga disimpan:")
    print(knn_file)

    # File 2: Daftar edge graf
    edge_file = (
        KNN_DIR / "knn_edges.csv"
    )

    edge_df.to_csv(
        edge_file,
        index=False
    )

    print("\nFile edge disimpan:")
    print(edge_file)

    # File 3: Informasi degree node
    node_file = (
        KNN_DIR / "knn_nodes.csv"
    )

    node_df.to_csv(
        node_file,
        index=False
    )

    print("\nFile node disimpan:")
    print(node_file)

    # File 4: GraphML untuk penyimpanan graf
    graph_file = (
        KNN_DIR / "knn_graph.graphml"
    )

    nx.write_graphml(
        G,
        graph_file
    )

    print("\nFile graf disimpan:")
    print(graph_file)


# ============================================================
# 9. PROGRAM UTAMA
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("K-NEAREST NEIGHBORS GRAPH")
    print("WEB MINING - PENDIDIKAN ANTARPROVINSI")
    print("=" * 60)

    print("\nParameter k:", K)

    print("Metode graf: Union k-NN")

    print("Hanya similarity positif:", POSITIVE_ONLY)

    print("\nFolder output:")
    print(KNN_DIR)

    # Tahap 1: Membaca matriks similarity
    similarity_df = load_similarity_matrix()

    # Tahap 2: Menentukan tetangga terdekat
    knn_df, neighbor_dict = calculate_knn(
        similarity_df
    )

    # Tahap 3: Membentuk graf
    G = build_knn_graph(
        similarity_df,
        neighbor_dict
    )

    # Tahap 4: Membuat daftar edge
    edge_df = create_edge_dataframe(G)

    # Tahap 5: Membuat informasi node
    node_df = create_node_dataframe(G)

    # Tahap 6: Menyimpan seluruh hasil
    save_knn_results(
        knn_df,
        edge_df,
        node_df,
        G
    )

    print("\n")

    print("=" * 60)
    print("PROSES K-NN SELESAI")
    print("=" * 60)

    print("\nJumlah provinsi:", G.number_of_nodes())

    print("Jumlah edge:", G.number_of_edges())

    print("\nFile hasil tersimpan di:")
    print(KNN_DIR)

    print("\nFile yang dihasilkan:")
    print("1. knn_neighbors.csv")
    print("2. knn_edges.csv")
    print("3. knn_nodes.csv")
    print("4. knn_graph.graphml")


# ============================================================
# 10. EKSEKUSI PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
