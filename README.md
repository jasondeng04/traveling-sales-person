# Large-Scale TSP via Clustering and Meta-Heuristic Optimization

A two-level solver for the Travelling Salesman Problem on a **115,475-city** instance. The problem is decomposed with k-means clustering, solved per-cluster with meta-heuristics, and stitched back into a single global tour. Guided Local Search (GLS) and Ant Colony Optimization (ACO) are benchmarked against a Simulated Annealing (SA) baseline.

**Headline result:** GLS produces a final tour of **17.6M** vs. ACO's **18.5M** — despite ACO finding ~5% *better* sub-tours at equal time budgets. Decomposition throughput beats per-subproblem quality when there are 1,000 subproblems to solve.

---

## Problem

Given $n$ cities with pairwise distances $d(c_i, c_j)$, find the permutation $\pi$ minimizing total tour length:

$$\min_{\pi \in S_n} \sum_{i=1}^{n} d\big(c_{\pi(i)},\, c_{\pi(i+1)}\big), \qquad \pi(n+1) = \pi(1)$$

TSP is NP-hard, and the search space of $(n-1)!/2$ distinct tours makes exhaustive search hopeless well below $n = 20$. At $n = 115{,}475$, a full distance matrix is infeasible to hold in memory ($\approx 1.3 \times 10^{10}$ entries), so the solver never materializes one — distance matrices are precomputed per cluster only.

## Approach

**1. Spatial decomposition (k-means).** Cities are partitioned into $k = 1000$ geographic clusters by minimizing within-cluster variance:

$$J = \sum_{j=1}^{k} \sum_{x \in C_j} \lVert x - \mu_j \rVert^2$$

**2. Intra-cluster optimization.** Each cluster is solved as an independent TSP by SA, GLS, or ACO.

**3. Inter-cluster ordering and stitching.** Cluster centroids are treated as a small TSP (solved with SA) to fix visiting order, then adjacent sub-tours are joined at their nearest boundary-city pair to form one continuous global tour.

## Algorithms

### Simulated Annealing (baseline)

Accepts worsening moves with probability $P = \exp(-\Delta / T)$ under a cooling schedule. Run without clustering on the full instance, SA barely moves a random tour off its starting length — establishing that decomposition, not the search heuristic, is what makes the problem tractable.

### Guided Local Search

GLS wraps 2-opt local search in a penalty term that discourages repeatedly-used bad edges, optimizing an augmented objective:

$$h(s) = g(s) + \lambda \sum_{i=1}^{M} I_i(s)\, p_i$$

where $g(s)$ is true tour length, $p_i$ is the penalty on edge $i$, and $I_i(s)$ indicates its presence in $s$. At each local optimum, the edge with maximum utility is penalized:

$$u_i = I_i(s) \cdot \frac{c_i}{1 + p_i}$$

GLS reshapes the landscape rather than randomizing out of it, which makes it strong on structured Euclidean instances.

### Ant Colony Optimization

Ants build tours probabilistically, biased by pheromone $\tau_{ij}$ and desirability $\eta_{ij} = 1/d_{ij}$:

$$p_{ij}^{k} = \frac{\tau_{ij}^{\alpha}\, \eta_{ij}^{\beta}}{\sum_{l \in N_i^{k}} \tau_{il}^{\alpha}\, \eta_{il}^{\beta}}$$

Trails evaporate and are reinforced along good tours each iteration:

$$\tau_{ij} \leftarrow (1 - \rho)\,\tau_{ij} + \sum_{k} \Delta \tau_{ij}^{k}$$

## Implementation note: $O(1)$ delta evaluation

Recomputing full tour length after every candidate swap is $O(n)$ and dominates runtime. Since a 2-opt swap on edges $(a,b)$ and $(c,d)$ only changes four edges, the cost difference is evaluated directly:

$$\Delta = d(a,c) + d(b,d) - d(a,b) - d(c,d)$$

With a precomputed per-cluster distance matrix this is constant-time. The result is **7,500 swap evaluations per cluster in 0.08 seconds** — the single change that makes GLS viable across 1,000 clusters.

## Results

| Experiment | Setup | Result |
|---|---|---|
| SA, no clustering | 115,475 cities, full instance | ~507M — essentially no gain over a random tour |
| GLS, clustered | $k = 1000$ | 91%+ improvement in sub-tour length |
| ACO, clustered | $k = 1000$ | 91%+ improvement; ~5% better sub-tours than GLS at equal time |
| Per-cluster speed | GLS with $O(1)$ delta eval | 7,500 swaps in 0.08 s; **7× faster per cluster than ACO** |
| **Full pipeline tour** | 115,475 cities, $k = 1000$ | **GLS 17.6M** vs. ACO 18.5M |

**Takeaway.** Three findings drive the result. First, clustering quality is the foundation — unclustered SA fails outright, while any decent decomposition unlocks 91%+ improvement from either meta-heuristic. Second, GLS wins end-to-end on throughput, not solution quality: it is 7× faster per cluster, so within a fixed wall-clock budget it completes more optimization rounds across all 1,000 clusters than ACO can, and that compounds into a ~5% shorter global tour. Third, the greedy boundary-city stitch is now the binding constraint — many inter-cluster edges are longer than necessary, and running SA on centroids for cluster ordering only partially fixes this.

## Repository structure

```
TSP/
├── TSP_Project.ipynb      # Full pipeline: clustering, SA/GLS/ACO, stitching, plots
├── TSP_Presentation.pptx  # Slide deck summarizing method and results
└── README.md
```

## Running it

1. Clone the repo: `git clone https://github.com/jasondeng04/TSP.git`
2. Install dependencies: `pip install numpy pandas scikit-learn matplotlib`
3. Place the city coordinate file at `[path/filename]` — see [dataset source].
4. Open `TSP_Project.ipynb` and run all cells top to bottom.

Key parameters at the top of the notebook: cluster count `k`, SA cooling rate, GLS penalty weight $\lambda$, and ACO's $\alpha$, $\beta$, $\rho$.

## Limitations and next steps

- **Stitching is the biggest remaining gap.** Cluster boundaries are fixed after k-means and the join is greedy, so many inter-cluster edges are avoidable. Applying GLS or ACO to the stitch itself — or a 2-opt pass restricted to seam edges — is the clearest path to further improvement.
- $k$ was set manually rather than tuned; there is a real trade-off between sub-problem tractability and the number of stitch seams introduced.
- ACO's pheromone matrix limits cluster size. Candidate lists or nearest-neighbor restriction would let it scale to larger sub-problems where its per-cluster quality edge might actually pay off.
- No comparison against an exact or near-exact reference (Concorde, LKH) to quantify the optimality gap of the 17.6M tour.

## Authors

Jason Deng and Geno — May 2026.
