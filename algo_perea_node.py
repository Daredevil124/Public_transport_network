import time

import networkx as nx
import pandas as pd

from data_loader import load_network
from metrics import get_all_metrics


def run_perea_algorithm(choice):
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

    # Perea et al. Enumerative Node Addition
    print(f"\nRunning Perea et al. (2014) Enumerative Search on {name}...")
    start_time = time.time()

    edge_bc = nx.edge_betweenness_centrality(main_comp)
    sub_bc = nx.betweenness_centrality(main_comp)
    candidate_tracks = sorted(edge_bc, key=edge_bc.get, reverse=True)[:5]
    candidate_hubs = sorted(sub_bc, key=sub_bc.get, reverse=True)[:10]

    best_eff = -1
    best_G = None

    for track in candidate_tracks:
        track_dist = main_comp[track[0]][track[1]]["distance"]
        for p in [0.25, 0.50, 0.75]:
            for hub in candidate_hubs:
                if hub in track:
                    continue

                G_test = main_comp.copy()
                G_test.remove_edge(*track)
                new_node = "PEREA_RELIEF_HUB"
                G_test.add_node(new_node)
                G_test.add_edge(track[0], new_node, distance=track_dist * p)
                G_test.add_edge(new_node, track[1], distance=track_dist * (1.0 - p))
                G_test.add_edge(new_node, hub, distance=1.5)

                test_eff = get_all_metrics(G_test)[1]
                if test_eff > best_eff:
                    best_eff = test_eff
                    best_G = G_test.copy()

    print(f"-> Perea Algorithm finished in {time.time() - start_time:.2f} seconds.")
    results.append(["Perea Node Addition", *get_all_metrics(best_G)])

    df_res = pd.DataFrame(
        results, columns=["Phase", "E_top", "E_weight", "APL", "Diameter"]
    )
    print("\n=== PEREA ET AL. (2014) RESULTS ===")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_perea_algorithm(1)  # Change to 2 to run London
