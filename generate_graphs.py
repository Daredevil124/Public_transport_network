import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from algo_dp_base import run_our_algorithm
from algo_perea_node import run_perea_algorithm
from algo_sbu_edge import run_sbu_algorithm

def get_data(choice):
    print(f"\n--- Gathering data for choice {choice} ---")
    df_our = run_our_algorithm(choice)
    df_perea = run_perea_algorithm(choice)
    df_sbu = run_sbu_algorithm(choice)
    
    # Identify relevant rows
    our_row = df_our[df_our['Phase'].str.contains('Ultimate', case=False)].iloc[0]
    perea_row = df_perea[df_perea['Phase'].str.contains('Their Algo', case=False)].iloc[0]
    sbu_row = df_sbu[df_sbu['Phase'].str.contains('Their Algo', case=False)].iloc[0]
    
    # Groups of metrics
    group1_cols = ['APL (Stops)', 'Diameter (Max Stops)']
    group1_labels = ['APL', 'Diameter']
    
    group2_cols = ['E_weight (Geo-Eff)', 'E_top (Directness)']
    group2_labels = ['E_Weight', 'E_top']
    
    data = {
        'our': {
            'g1': [our_row[m] for m in group1_cols],
            'g2': [our_row[m] for m in group2_cols]
        },
        'perea': {
            'g1': [perea_row[m] for m in group1_cols],
            'g2': [perea_row[m] for m in group2_cols]
        },
        'sbu': {
            'g1': [sbu_row[m] for m in group1_cols],
            'g2': [sbu_row[m] for m in group2_cols]
        }
    }
    
    return group1_labels, group2_labels, data

def plot_comparison(labels, our, other, other_name, dataset_name, group_name, filename):
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, our, width, label='Our Algorithm', color='skyblue')
    rects2 = ax.bar(x + width/2, other, width, label=other_name, color='salmon')
    
    ax.set_ylabel('Values')
    ax.set_title(f'{dataset_name}: {group_name}\nOur Algorithm vs {other_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.4f' if 'E_' in group_name else '%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.4f' if 'E_' in group_name else '%.2f')
    
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def main():
    # Dataset 1: Delhi Metro
    g1_labels, g2_labels, delhi_data = get_data(1)
    
    # Delhi: Our vs Perea
    plot_comparison(g1_labels, delhi_data['our']['g1'], delhi_data['perea']['g1'], 
                    'Perea et al.', 'Delhi Metro', 'APL and Diameter', 'delhi_perea_apl_diam.png')
    plot_comparison(g2_labels, delhi_data['our']['g2'], delhi_data['perea']['g2'], 
                    'Perea et al.', 'Delhi Metro', 'E_Weight and E_top', 'delhi_perea_eff.png')
    
    # Delhi: Our vs SBU
    plot_comparison(g1_labels, delhi_data['our']['g1'], delhi_data['sbu']['g1'], 
                    'SBU-CSE', 'Delhi Metro', 'APL and Diameter', 'delhi_sbu_apl_diam.png')
    plot_comparison(g2_labels, delhi_data['our']['g2'], delhi_data['sbu']['g2'], 
                    'SBU-CSE', 'Delhi Metro', 'E_Weight and E_top', 'delhi_sbu_eff.png')
    
    # Dataset 2: London Railway
    g1_labels, g2_labels, london_data = get_data(2)
    
    # London: Our vs Perea
    plot_comparison(g1_labels, london_data['our']['g1'], london_data['perea']['g1'], 
                    'Perea et al.', 'London Railway', 'APL and Diameter', 'london_perea_apl_diam.png')
    plot_comparison(g2_labels, london_data['our']['g2'], london_data['perea']['g2'], 
                    'Perea et al.', 'London Railway', 'E_Weight and E_top', 'london_perea_eff.png')
    
    # London: Our vs SBU
    plot_comparison(g1_labels, london_data['our']['g1'], london_data['sbu']['g1'], 
                    'SBU-CSE', 'London Railway', 'APL and Diameter', 'london_sbu_apl_diam.png')
    plot_comparison(g2_labels, london_data['our']['g2'], london_data['sbu']['g2'], 
                    'SBU-CSE', 'London Railway', 'E_Weight and E_top', 'london_sbu_eff.png')

if __name__ == "__main__":
    main()
