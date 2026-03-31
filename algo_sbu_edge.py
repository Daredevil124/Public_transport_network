import time
from itertools import combinations

import networkx as nx
import pandas as pd

from data_loader import load_network
from metrics import calc_natural_connectivity, get_all_metrics


def run_sbu_algorithm(choice):
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

    # SBU-CSE-14-1 Edge Addition (Natural Connectivity)
    print(f"\nRunning SBU-CSE-14-1 Natural Connectivity Search on {name}...")
    start_time = time.time()

    try:
        ec = nx.eigenvector_centrality_numpy(main_comp)
    except:
        ec = nx.degree_centrality(main_comp)

    candidates = sorted(ec, key=ec.get, reverse=True)[:40]
    best_nc = -float("inf")
    best_edge = None

    possible_edges = [
        (u, v) for u, v in combinations(candidates, 2) if not main_comp.has_edge(u, v)
    ]

    for u, v in possible_edges:
        G_test = main_comp.copy()
        G_test.add_edge(u, v)
        nc = calc_natural_connectivity(G_test)
        if nc > best_nc:
            best_nc = nc
            best_edge = (u, v)

    print(f"-> SBU Edge Algorithm finished in {time.time() - start_time:.2f} seconds.")

    G_final = main_comp.copy()
    G_final.add_edge(best_edge[0], best_edge[1], distance=2.0)
    results.append(["SBU Edge Addition", *get_all_metrics(G_final)])

    df_res = pd.DataFrame(
        results, columns=["Phase", "E_top", "E_weight", "APL", "Diameter"]
    )
    print("\n=== SBU-CSE-14-1 RESULTS ===")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_sbu_algorithm(1)  # Change to 2 to run London
