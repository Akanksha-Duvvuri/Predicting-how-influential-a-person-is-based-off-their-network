from pathlib import Path #inbuilt tool for working with files and folders - helps you represent a file path as a python object
import csv #inbuilt python library for working with tabular files - using it to save information about each node
import igraph as ig #importing the library -- importing as ig (giving it a shorter name - like a nickname)

DATASET = Path("facebook_combined.txt") #creates a path pointing to your dataset file, doesnt read the file yet

edges = [] #create an empty list - a list is a collection of items in python, store all the connections ex (edge = [(0,1), (0,2) etc etc]) from the dataset file

#each pair is a tuple - a tuple lets you group the two endpoints of an edge together

#so in simple terms - edge is the whole collection, (0,1) is one connection and 0 and 1 are the endpoints of that connection

with DATASET.open("r") as file: #opening the file, r means read mode - you want to read the file not overwrite it by mistake

#as file assigns the opened file to a var named "file", with automatically closes the file when the indented block finishes(even if an error occurs)
    for line_number, line in enumerate(file, start=1):  #read everyline of the file, enumerate is a function used to loop over an iterable (ex a list, type or string) while automatically keeping track of the index counter - eliminates the need to manually create and increment a counter variable inside your loops
        parts = line.split() #seperates a string into pieces using whitespace, including spaces, tabs and new lines

        if not parts: #skips empty lines - not parts as in if the list is empty, continue (so skip the rest of the current loop itr and move to the next line)
            continue

        if len(parts) != 2: #check if every line has two values - if the length of the part isnt two then raise an error
            raise ValueError( #deliberately stops the program and displays an error if the line doesnt contain two values
                f"Expected two columns at line {line_number}: {line.strip()}"
            ) #f"..." - creates an f-string, which lets you insert variables inside text using curly braces, line strip removes whitespaces from the beg and end of the line

        u, v = map(int, parts) #converting strings into integers - applies int() to every item in parts and then unpacks the two values into u and v
        edges.append((u, v)) #adds one item to the end of a list - u = 0 and v = 1, this adds the tuple (0,1) to edges
#after the loop finishes, the list contains all the connections read from the file

if not edges: 
    raise ValueError("Dataset contains no edges.") #if edges is empty or no valid connections were loaded - the program stops rather than attempting to construct an empty graph

# Preserve original node IDs, even when IDs have gaps - converting the node ids into igraphs indices
node_ids = sorted({node for edge in edges for node in edge}) 
#igraph internally identifies vertices using indices from 0 to the number of vertices - 1 ---> create a mapping bw the original ids and the internal indices

#the curly braces create a set (keeps only unique values)
#sorted arranges them in ascending order

#created a list of every distinct node that appears in the dataset
id_to_index = {node: i for i, node in enumerate(node_ids)}  #enumerate again generates pairs of index and values -- the original node is the key and the internal graph index is the value

indexed_edges = [ 
    (id_to_index[u], id_to_index[v]) #
    for u, v in edges
]

g = ig.Graph( #creates a graph object
    n=len(node_ids), #creates as many vertices as the number of unique node ids
    edges=indexed_edges, #connect vertices according to our list of edges
    directed=False #create edges as undirected (connection bw to nodes works both ways)
)

g.vs["original_id"] = node_ids #g.vs refers to the graphs collection of vertices, original id assigns a vertex attribute named original id, and node ids supplies the original id for each vertex in index order

original_edge_count = g.ecount() #returns the number of edges currently in the graph 
self_loop_count = sum(u == v for u, v in indexed_edges) #an edge connecting a node to itself, checks if both the end points are equal - gives T or F for each edge. 1 is true and 0 is false 

# Remove self-loops and duplicate edges.
g.simplify(multiple=True, loops=True)

print("Original edge rows:", original_edge_count)
print("Self-loops:", self_loop_count)
print("Unique vertices:", g.vcount()) #Number of vertices in the graph
print("Unique edges:", g.ecount()) #Number of edges after simplification
print("Density:", g.density())  #Proportion of possible edges that actually exist
print("Average degree:", sum(g.degree()) / g.vcount())
print("Connected components:", len(g.connected_components())) #Number of connections for each vertex
print("Degree range:", min(g.degree()), max(g.degree())) #Smallest and largest values

#A connected component is a group of vertices where every vertex can reach every other vertex in that group by following edges. Two disconnected groups have no path connecting them.
#the avg degree is calculated by adding all degrees and diving by the number of vertices 

#density - a density close to 0 means few connections exist, a denstiy of 1 means every distinct pair of vertices is connected. Density = 2m / n(n-1), n vertices and m edges
# Community detection


communities = g.community_leiden( #a method approved by igraph
    objective_function="modularity", #tells the algo to optimize modularity (a measure of community struct, the quality of building larger systems by combining separate, independent parts or subsystems)
    resolution=1.0 #influences the size and number of communities, a higher resolution generally favors larger ones
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
