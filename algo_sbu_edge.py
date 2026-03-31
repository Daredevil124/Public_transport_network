import time
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from data_loader import load_network
from metrics import get_all_metrics


def run_sbu_algorithm(choice):
    G, name = load_network(choice)
    results = []

    # --- PHASE 0: BASELINE ---
    print(f"\n[PHASE 0] Calculating Baseline for {name}...")
    met = get_all_metrics(G)
    results.append(["Baseline", *met])

    node_bc = nx.betweenness_centrality(G, normalized=True)
    kingpin = max(node_bc, key=node_bc.get)
    print(f"-> KINGPIN: {kingpin} (Load: {node_bc[kingpin] * 100:.2f}%)")

    # --- PHASE 1: ATTACK ---
    print(f"\n[PHASE 1] Destroying '{kingpin}'...")
    G_broken = G.copy()
    G_broken.remove_node(kingpin)
    main_comp = G_broken.subgraph(
        max(nx.connected_components(G_broken), key=len)
    ).copy()
    met = get_all_metrics(main_comp)
    results.append(["Post-Attack", *met])

    # =========================================================================
    # --- PHASE 2: THEIR ALGORITHM (SBU-CSE-14-1: Natural Connectivity) ---
    # =========================================================================
    print(
        f"\n[PHASE 2] Executing 'Their Algorithm' (Maximizing Natural Connectivity)..."
    )
    start_time = time.time()

    # 1. Helper function to calculate Natural Connectivity (Eigenvalue Math)
    def calc_natural_connectivity(graph):
        A = nx.to_numpy_array(graph)
        eigenvalues = np.linalg.eigvalsh(A)
        # Numerically stable calculation of log(mean(exp(eigenvalues)))
        max_eig = np.max(eigenvalues)
        return max_eig + np.log(np.mean(np.exp(eigenvalues - max_eig)))

    # 2. Pruning: To maximize Natural Connectivity, the paper's heuristic connects
    # nodes with the highest Eigenvector Centrality.
    try:
        ec = nx.eigenvector_centrality_numpy(main_comp)
    except:
        ec = nx.degree_centrality(main_comp)  # Fallback if matrix fails to converge

    candidates = sorted(ec, key=ec.get, reverse=True)[:40]  # Top 40 structural hubs

    best_nc = -float("inf")
    best_edge = None

    possible_edges = [
        (u, v) for u, v in combinations(candidates, 2) if not main_comp.has_edge(u, v)
    ]
    print(
        f"-> Calculating Matrix Eigenvalues for {len(possible_edges)} possible edges..."
    )

    for u, v in possible_edges:
        G_test = main_comp.copy()
        G_test.add_edge(u, v)
        nc = calc_natural_connectivity(G_test)

        if nc > best_nc:
            best_nc = nc
            best_edge = (u, v)

    print(f"-> Optimization Complete in {time.time() - start_time:.2f} seconds.")
    print(f"-> Best Edge for Natural Connectivity: {best_edge[0]} <---> {best_edge[1]}")

    G_edge = main_comp.copy()
    G_edge.add_edge(
        best_edge[0], best_edge[1], distance=2.0
    )  # Assume 2km bypass for our E_weight calculation
    results.append(["With Bypass Edge (Their Algo)", *get_all_metrics(G_edge)])

    # =========================================================================
    # --- PHASE 3: OUR ULTIMATE ALGORITHM (Matrix-Optimized Global Search) ---
    # =========================================================================
    print(
        f"\n[PHASE 3] Executing 'Our Ultimate Node Addition' (O(V^2) Matrix Global Search)..."
    )
    start_time = time.time()

    nodes = list(main_comp.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)
    dist_matrix = np.full((N, N), np.inf)
    lengths = dict(nx.all_pairs_dijkstra_path_length(main_comp, weight="distance"))
    for u in nodes:
        for v, d in lengths[u].items():
            dist_matrix[node_to_idx[u], node_to_idx[v]] = d

    edge_bc = nx.edge_betweenness_centrality(main_comp, weight="distance")
    sub_bc = nx.betweenness_centrality(main_comp)
    candidate_tracks = sorted(edge_bc, key=edge_bc.get, reverse=True)[:20]
    candidate_hubs = sorted(sub_bc, key=sub_bc.get, reverse=True)[:20]

    print(
        f"-> Cross-evaluating {len(candidate_tracks) * 9 * len(candidate_hubs)} combinations for minimal travel time..."
    )

    best_geo_eff = -1
    best_setup = None

    for track in candidate_tracks:
        u_idx, v_idx = node_to_idx[track[0]], node_to_idx[track[1]]
        track_dist = main_comp[track[0]][track[1]]["distance"]

        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            du = track_dist * p
            dv = track_dist * (1.0 - p)

            for hub in candidate_hubs:
                if hub in track:
                    continue
                h_idx = node_to_idx[hub]
                dh = 1.5

                dist_to_X = np.minimum(
                    dist_matrix[:, u_idx] + du,
                    np.minimum(dist_matrix[:, v_idx] + dv, dist_matrix[:, h_idx] + dh),
                ).reshape(N, 1)
                new_dist_matrix = np.minimum(dist_matrix, dist_to_X + dist_to_X.T)

                with np.errstate(divide="ignore"):
                    inv_dist = 1.0 / new_dist_matrix
                np.fill_diagonal(inv_dist, 0)
                sum_existing = np.sum(inv_dist)

                with np.errstate(divide="ignore"):
                    inv_to_X = 1.0 / dist_to_X
                inv_to_X[inv_to_X == np.inf] = 0
                total_eff = (sum_existing + (np.sum(inv_to_X) * 2)) / ((N + 1) * N)

                if total_eff > best_geo_eff:
                    best_geo_eff = total_eff
                    best_setup = (track, p, hub)

    best_track, best_p, best_hub = best_setup
    G_ultimate = main_comp.copy()
    G_ultimate.remove_edge(*best_track)

    new_node = "NEW_RELIEF_HUB"
    G_ultimate.add_node(new_node)
    G_ultimate.add_edge(
        best_track[0],
        new_node,
        distance=main_comp[best_track[0]][best_track[1]]["distance"] * best_p,
    )
    G_ultimate.add_edge(
        new_node,
        best_track[1],
        distance=main_comp[best_track[0]][best_track[1]]["distance"] * (1.0 - best_p),
    )
    G_ultimate.add_edge(new_node, best_hub, distance=1.5)

    print(f"-> Optimization Complete in {time.time() - start_time:.2f} seconds!")
    results.append(["With Relief Node (Our Algo)", *get_all_metrics(G_ultimate)])

    # --- FINAL SUMMARY TABLE ---
    df_res = pd.DataFrame(
        results,
        columns=[
            "Phase",
            "E_top (Directness)",
            "E_weight (Geo-Eff)",
            "APL (Stops)",
            "Diameter (Max Stops)",
        ],
    )
    print("\n" + "=" * 85)
    print(f" FINAL RESEARCH DATASET: {name.upper()} ")
    print("=" * 85)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_sbu_algorithm(1)  # Change to 2 to run London
