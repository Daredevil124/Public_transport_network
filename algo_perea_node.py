import time
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from data_loader import load_network
from metrics import get_all_metrics


def run_perea_algorithm(choice):
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

    # --- PHASE 2: OPTIMAL EDGE (OUR OPTIMIZED MATRIX PRUNING) ---
    print(f"\n[PHASE 2] Searching for Optimal Bypass Tunnel...")
    nodes = list(main_comp.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)

    dist_matrix = np.full((N, N), np.inf)
    lengths = dict(nx.all_pairs_dijkstra_path_length(main_comp, weight="distance"))
    for u in nodes:
        for v, d in lengths[u].items():
            dist_matrix[node_to_idx[u], node_to_idx[v]] = d

    sub_bc = nx.betweenness_centrality(main_comp)
    candidates = sorted(sub_bc, key=sub_bc.get, reverse=True)[:80]
    best_edge, best_val = None, -1

    for u, v in combinations(candidates, 2):
        if main_comp.has_edge(u, v):
            continue
        iu, iv = node_to_idx[u], node_to_idx[v]
        new_dist = np.minimum(
            dist_matrix,
            np.minimum(
                dist_matrix[:, [iu]] + 2.0 + dist_matrix[[iv], :],
                dist_matrix[:, [iv]] + 2.0 + dist_matrix[[iu], :],
            ),
        )
        val = np.sum(1.0 / new_dist[new_dist > 0]) / (N * (N - 1))
        if val > best_val:
            best_val, best_edge = val, (u, v)

    print(f"-> Optimal Bypass found: {best_edge[0]} <---> {best_edge[1]}")
    G_edge = main_comp.copy()
    G_edge.add_edge(*best_edge, distance=2.0)
    results.append(["With Bypass Edge", *get_all_metrics(G_edge)])

    # =========================================================================
    # --- PHASE 3: THEIR ALGORITHM (Enumerative Discretization Search) ---
    # Based on Algorithm 2 from Perea et al. (2014)
    # =========================================================================
    print(
        f"\n[PHASE 3] Executing 'Their Algorithm' (Brute-Force Enumerative Search)..."
    )
    start_time = time.time()

    # Identify top 5 busiest tracks to test placing the station
    edge_bc = nx.edge_betweenness_centrality(main_comp, normalized=True)
    candidate_tracks = sorted(edge_bc, key=edge_bc.get, reverse=True)[:5]

    # Identify top 10 busiest hubs to test connecting the road to
    candidate_hubs = sorted(sub_bc, key=sub_bc.get, reverse=True)[:10]

    best_geo_eff = -1
    best_setup = None

    print(
        f"-> Commencing nested loops: Testing multiple discrete locations on {len(candidate_tracks)} tracks, connecting to {len(candidate_hubs)} different hubs..."
    )

    # Pre-calculate track distances to avoid repeat lookups
    track_distances = {track: main_comp[track[0]][track[1]]["distance"] for track in candidate_tracks}

    # NESTED LOOP 1: Iterate over the tracks (The Rail Line)
    for track in candidate_tracks:
        track_dist = track_distances[track]
        u_idx, v_idx = node_to_idx[track[0]], node_to_idx[track[1]]

        # NESTED LOOP 2: Discretize the track (Testing 25%, 50%, and 75% marks)
        for discrete_point in [0.25, 0.50, 0.75]:
            dist_from_u = track_dist * discrete_point
            dist_from_v = track_dist * (1.0 - discrete_point)

            # NESTED LOOP 3: Iterate over the hubs (The connection to the road)
            for hub in candidate_hubs:
                hub_idx = node_to_idx[hub]
                
                # OPTIMIZED: Calculate distance to new node X
                # dist(i, X) = min(dist(i, u) + duX, dist(i, v) + dvX, dist(i, h) + dhX)
                duX, dvX, dhX = dist_from_u, dist_from_v, 1.5
                dist_to_X = np.minimum(
                    dist_matrix[:, [u_idx]] + duX,
                    np.minimum(
                        dist_matrix[:, [v_idx]] + dvX,
                        dist_matrix[:, [hub_idx]] + dhX
                    )
                )

                # Update matrix: min(dist(i,j), dist(i,X) + dist(X,j))
                new_dist_matrix = np.minimum(dist_matrix, dist_to_X + dist_to_X.T)

                with np.errstate(divide="ignore"):
                    inv_dist = 1.0 / new_dist_matrix
                np.fill_diagonal(inv_dist, 0)
                sum_existing = np.sum(inv_dist)

                with np.errstate(divide="ignore"):
                    inv_to_X = 1.0 / dist_to_X
                inv_to_X[inv_to_X == np.inf] = 0
                sum_to_X = np.sum(inv_to_X) * 2

                total_eff = (sum_existing + sum_to_X) / ((N + 1) * N)

                if total_eff > best_geo_eff:
                    best_geo_eff = total_eff
                    best_setup = (track, discrete_point, hub)

    print(f"-> Enumerative Search Complete in {time.time() - start_time:.2f} seconds.")
    
    # Physically build the absolute best solution found
    best_track, best_p, best_hub = best_setup
    G_test = main_comp.copy()
    G_test.remove_edge(*best_track)
    
    new_node = "NEW_RELIEF_HUB"
    G_test.add_node(new_node)
    dist_u = main_comp[best_track[0]][best_track[1]]["distance"] * best_p
    dist_v = main_comp[best_track[0]][best_track[1]]["distance"] * (1.0 - best_p)
    G_test.add_edge(best_track[0], new_node, distance=dist_u)
    G_test.add_edge(new_node, best_track[1], distance=dist_v)
    G_test.add_edge(new_node, best_hub, distance=1.5)

    print(f"-> Best Enumerative Result: Station {best_p*100}% down {best_track}, connected to {best_hub}")
    results.append(["With Relief Node (Their Algo)", *get_all_metrics(G_test)])

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
    return df_res


if __name__ == "__main__":
    run_perea_algorithm(1)  # Change to 2 to run London
