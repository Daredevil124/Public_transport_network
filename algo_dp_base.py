import time

import networkx as nx
import numpy as np
import pandas as pd

from data_loader import load_network
from metrics import get_all_metrics


def run_our_algorithm(choice):
    G, name = load_network(choice)
    results = []

    # Baseline & Attack
    results.append(["Baseline", *get_all_metrics(G)])
    node_bc = nx.betweenness_centrality(G, normalized=True)
    kingpin = max(node_bc, key=node_bc.get)

    G_broken = G.copy()
    G_broken.remove_node(kingpin)
    main_comp = G_broken.subgraph(
        max(nx.connected_components(G_broken), key=len)
    ).copy()
    results.append(["Post-Attack", *get_all_metrics(main_comp)])

    # OUR ALGORITHM: O(V^2) Matrix Node Addition
    print(f"\nRunning Our Ultimate Matrix Algorithm on {name}...")
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

    best_eff = -1
    best_setup = None

    for track in candidate_tracks:
        u_idx, v_idx = node_to_idx[track[0]], node_to_idx[track[1]]
        track_dist = main_comp[track[0]][track[1]]["distance"]

        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            du, dv = track_dist * p, track_dist * (1.0 - p)

            for hub in candidate_hubs:
                if hub in track:
                    continue
                h_idx = node_to_idx[hub]

                dist_to_X = np.minimum(
                    dist_matrix[:, u_idx] + du,
                    np.minimum(dist_matrix[:, v_idx] + dv, dist_matrix[:, h_idx] + 1.5),
                ).reshape(N, 1)
                new_matrix = np.minimum(dist_matrix, dist_to_X + dist_to_X.T)

                with np.errstate(divide="ignore"):
                    inv_dist = 1.0 / new_matrix
                np.fill_diagonal(inv_dist, 0)

                with np.errstate(divide="ignore"):
                    inv_to_X = 1.0 / dist_to_X
                inv_to_X[inv_to_X == np.inf] = 0

                total_eff = (np.sum(inv_dist) + (np.sum(inv_to_X) * 2)) / ((N + 1) * N)

                if total_eff > best_eff:
                    best_eff = total_eff
                    best_setup = (track, p, hub)

    print(f"-> Our Algorithm finished in {time.time() - start_time:.2f} seconds.")

    best_track, best_p, best_hub = best_setup
    G_final = main_comp.copy()
    G_final.remove_edge(*best_track)
    new_node = "OUR_RELIEF_HUB"
    G_final.add_node(new_node)
    G_final.add_edge(
        best_track[0],
        new_node,
        distance=main_comp[best_track[0]][best_track[1]]["distance"] * best_p,
    )
    G_final.add_edge(
        new_node,
        best_track[1],
        distance=main_comp[best_track[0]][best_track[1]]["distance"] * (1.0 - best_p),
    )
    G_final.add_edge(new_node, best_hub, distance=1.5)

    results.append(["Our Node Addition", *get_all_metrics(G_final)])

    df_res = pd.DataFrame(
        results, columns=["Phase", "E_top", "E_weight", "APL", "Diameter"]
    )
    print("\n=== OUR ULTIMATE ALGORITHM RESULTS ===")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_our_algorithm(1)  # Change to 2 to run London
