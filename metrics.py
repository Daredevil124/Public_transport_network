import networkx as nx
import numpy as np


def get_all_metrics(G):
    """Calculates Directness, Geo-Efficiency, APL, and Diameter."""
    nodes = len(G)
    if nodes < 2:
        return 0, 0, 0, 0
    e_top = nx.global_efficiency(G)

    # Calculate shortest physical paths (kilometers)
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
    e_weight = sum(
        1 / lengths[u][v] for u in G for v in G if u != v and v in lengths[u]
    ) / (nodes * (nodes - 1))

    apl = nx.average_shortest_path_length(G)
    diam = nx.diameter(G)

    return e_top, e_weight, apl, diam


def calc_natural_connectivity(graph):
    """Calculates structural robustness using Adjacency Matrix Eigenvalues."""
    A = nx.to_numpy_array(graph)
    eigenvalues = np.linalg.eigvalsh(A)
    # Numerically stable calculation to prevent overflow
    max_eig = np.max(eigenvalues)
    return max_eig + np.log(np.mean(np.exp(eigenvalues - max_eig)))
