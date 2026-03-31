# Transit Network Optimization & Resilience

This repository contains Python implementations for assessing and improving the resilience of large-scale urban transit networks (e.g., Delhi Metro, London Railway) following targeted nodal attacks. 

It acts as a comparative benchmark suite, evaluating an optimized $O(V^2)$ NumPy vectorized approach against two established methodologies from academic literature.

## Included Methodologies
1. **Perea et al. (2014)** (`algo_perea_node.py`): Implements an enumerative discretization search for optimal continuous node addition. 
2. **SBU-CSE-14-1** (`algo_sbu_edge.py`): Implements edge addition focused strictly on maximizing structural robustness (Natural Connectivity / Eigenvalues).
3. **Optimized Vectorized Search** (`algo_ours_ultimate.py`): A custom $O(V^2)$ algorithm that translates topological routing into parallel matrix mathematics, identifying the optimal geographical efficiency ($E_{weight}$) in a fraction of the computational time.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Execute any of the algorithm files: `python algo_ours_ultimate.py`
*(Note: Edit the `choice` variable at the bottom of the execution scripts to toggle between the Delhi dataset and the London dataset).*
