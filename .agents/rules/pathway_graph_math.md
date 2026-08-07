# Rule: Pathway Graph Mathematics & Resistance Modeling
## 1. Graph Construction Rules
- Graph: Undirected `networkx.Graph`. Nodes: Canonical HGNC Gene Symbols.
- Hub Exclusion List: Drop only technical ubiquitin/ribosomal hubs `{"UBC", "UBB", "RPS27A"}`; retain cancer-relevant TP53.
- Edges: STRING-DB (`add_nodes=25`).
- Edge Weight: $S_{\text{norm}} = S \text{ if } S \le 1.0 \text{ else } S / 1000$; $w(e) = 1.005 - S_{\text{norm}}$.
- Strip self-loops: `G.remove_edges_from(nx.selfloop_edges(G))`.
## 2. Canonical Overwrite & Branching
- Overwrite `T_primary` and `T_resistance` with `canonical_symbol` from ID Mapper.
- If $T_{\text{primary}} == T_{\text{resistance}}$: On-target bypass. If $\neq$: Execute NetworkX pipeline.
## 3. LCC & Bottleneck Math
- Guard: `if len(G.nodes) < 2: raise ValueError("NoPathwayFound")`.
- Extract the component containing both requested targets; fail explicitly if none exists.
- Distance $d$: use Dijkstra only when both nodes exist and are connected; reject missing targets rather than assigning a favorable default.
- Centrality $C_B$: `nx.betweenness_centrality(G_lcc, weight="weight")`.
- Adjusted Centrality: $C_{B,\text{adjusted}}(v) = C_B(v) / \log_2(\text{degree}(v) + 2)$.
## 4. Research Priority Scoring
- Never call the heuristic experimental synergy.
- Use fixed bounded transforms so scores do not change with candidate-pool composition.
- Components: topology 0.30, proximity 0.25, pharmacology 0.20, clinical evidence 0.25.
- Renormalize weights when pharmacology is missing and return every component plus limitations.
