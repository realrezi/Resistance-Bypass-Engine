# Rule: Pathway Graph Mathematics & Resistance Modeling
## 1. Graph Construction Rules
- Graph: Undirected `networkx.Graph`. Nodes: Canonical HGNC Gene Symbols.
- Hub Exclusion List: Drop `{"TP53", "UBC", "UBB", "RPS27A"}`.
- Edges: STRING-DB (`add_nodes=25`).
- Edge Weight: $S_{\text{norm}} = S \text{ if } S \le 1.0 \text{ else } S / 1000$; $w(e) = 1.005 - S_{\text{norm}}$.
- Strip self-loops: `G.remove_edges_from(nx.selfloop_edges(G))`.
## 2. Canonical Overwrite & Branching
- Overwrite `T_primary` and `T_resistance` with `canonical_symbol` from ID Mapper.
- If $T_{\text{primary}} == T_{\text{resistance}}$: On-target bypass. If $\neq$: Execute NetworkX pipeline.
## 3. LCC & Bottleneck Math
- Guard: `if len(G.nodes) < 2: raise ValueError("NoPathwayFound")`.
- Extract LCC: `G_lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()`.
- Distance $d$: `nx.dijkstra_path_length(G_lcc, T_2, P_2, weight="weight")` if path exists else 2.0.
- Centrality $C_B$: `nx.betweenness_centrality(G_lcc, weight="weight")`.
- Adjusted Centrality: $C_{B,\text{adjusted}}(v) = C_B(v) / \log_2(\text{degree}(v) + 2)$.
## 4. Synergy Scoring
- Normalization Guard: If `abs(max_val - min_val) < 1e-9`, default normalized value to 1.0. Clamp strictly `[0.0, 1.0]`.
- Case A (Complete): $\text{Synergy} = 0.40 \cdot C_{B,\text{norm}} + 0.30 \cdot (1.0 - d_{\text{norm}}) + 0.30 \cdot \text{Affinity}_{\text{norm}}$.
- Case B (Missing ChEMBL): $\text{Synergy} = 0.55 \cdot C_{B,\text{norm}} + 0.45 \cdot (1.0 - d_{\text{norm}})$.
