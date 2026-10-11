from pathlib import Path
import csv
import igraph as ig

DATASET = Path("facebook_combined.txt")

edges = []

with DATASET.open("r") as file:
    for line_number, line in enumerate(file, start=1):
        parts = line.split()

        if not parts:
            continue

        if len(parts) != 2:
            raise ValueError(
                f"Expected two columns at line {line_number}: {line.strip()}"
            )

        u, v = map(int, parts)
        edges.append((u, v))

if not edges:
    raise ValueError("Dataset contains no edges.")

# Preserve original node IDs, even when IDs have gaps.
node_ids = sorted({node for edge in edges for node in edge})
id_to_index = {node: i for i, node in enumerate(node_ids)}

indexed_edges = [
    (id_to_index[u], id_to_index[v])
    for u, v in edges
]

g = ig.Graph(
    n=len(node_ids),
    edges=indexed_edges,
    directed=False
)

g.vs["original_id"] = node_ids

original_edge_count = g.ecount()
self_loop_count = sum(u == v for u, v in indexed_edges)

# Remove self-loops and duplicate edges.
g.simplify(multiple=True, loops=True)

print("Original edge rows:", original_edge_count)
print("Self-loops:", self_loop_count)
print("Unique vertices:", g.vcount())
print("Unique edges:", g.ecount())
print("Density:", g.density())
print("Average degree:", sum(g.degree()) / g.vcount())
print("Connected components:", len(g.connected_components()))
print("Degree range:", min(g.degree()), max(g.degree()))

# Community detection
communities = g.community_leiden(
    objective_function="modularity",
    resolution=1.0
)

membership = communities.membership

print("Communities detected:", len(communities))
print("Modularity:", communities.modularity)

# Node-level features
degrees = g.degree()
clustering = g.transitivity_local_undirected(mode="zero")
coreness = g.coreness()
betweenness = g.betweenness()
pagerank = g.pagerank()
community_sizes = communities.sizes()

rows = []

for i, vertex in enumerate(g.vs):
    neighbors = g.neighbors(i)

    if neighbors:
        mean_neighbor_degree = (
            sum(degrees[j] for j in neighbors) / len(neighbors)
        )
    else:
        mean_neighbor_degree = 0.0

    # Nodes reachable within two hops, excluding the node itself.
    two_hop = set(neighbors)

    for neighbor in neighbors:
        two_hop.update(g.neighbors(neighbor))

    two_hop.discard(i)

    rows.append({
        "node_id": vertex["original_id"],
        "degree": degrees[i],
        "clustering_coefficient": clustering[i],
        "mean_neighbor_degree": mean_neighbor_degree,
        "two_hop_reachable_count": len(two_hop),
        "coreness": coreness[i],
        "betweenness": betweenness[i],
        "pagerank": pagerank[i],
        "community_id": membership[i],
        "community_size": community_sizes[membership[i]]
    })

with open("node_features.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved node_features.csv")



palette = ig.drawing.colors.ClusterColoringPalette(len(communities))
vertex_colors = [palette[c] for c in membership]

layout = g.layout("fr", niter=500)

ig.plot(
    g,
    target="graph_communities.png",
    layout=layout,
    vertex_color=vertex_colors,
    vertex_size=5,
    vertex_label=None,
    edge_width=0.4,
    bbox=(1800, 1400),
    margin=30
)

print("Saved graph_communities.png")
