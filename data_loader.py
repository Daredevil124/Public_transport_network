import re

import networkx as nx
import pandas as pd


def clean_station_name(name):
    """Links transfer stations by removing line-specific annotations."""
    return re.sub(r"\s*\[.*?\]|\s*\(.*?\)", "", str(name)).strip()


def load_network(choice):
    if choice == 1:
        print("Loading Delhi Metro...")
        df = pd.read_csv("Delhi-Metro-Network.csv")
        df = df.sort_values(by=["Line", "Distance from Start (km)"])
        G = nx.Graph()
        for _, group in df.groupby("Line"):
            stations = group["Station Name"].apply(clean_station_name).tolist()
            dists = group["Distance from Start (km)"].tolist()
            for i in range(len(stations) - 1):
                G.add_edge(
                    stations[i],
                    stations[i + 1],
                    distance=max(0.1, abs(dists[i + 1] - dists[i])),
                )
        name = "Delhi Metro"
    else:
        print("Loading London Railway (edges.csv)...")
        df = pd.read_csv("edges.csv")
        G = nx.from_pandas_edgelist(df, "source", "target", ["distance"])
        name = "London Railway"

    # Isolate the main connected grid to prevent infinite paths
    main_grid = max(nx.connected_components(G), key=len)
    return G.subgraph(main_grid).copy(), name
