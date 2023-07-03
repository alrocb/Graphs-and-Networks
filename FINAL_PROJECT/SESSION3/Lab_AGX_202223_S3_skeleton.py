import networkx as nx
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Pass to a list the graphs passed as a parameter
    graph_list = list(arg)

    # Create a definitive list
    graphL = list()

    # Get as a set all the nodes in a single graph, save it then on another list
    for graph in graph_list:
      graphL.append(set(graph.nodes))

    # Get the intersection of all the graphs.
    common_nodes = set.intersection(*graphL)

    # Get the number
    num_common_nodes = len(common_nodes)

    return num_common_nodes

    # ----------------- END OF FUNCTION --------------------- #


def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Convert DegreeView object to dictionary
    node_degrees = dict(g.degree())

    # Initialize the dictionary for degree distribution.
    degree_distribution_dict = {}

    # Iterate over the nodes_degrees
    for degree in node_degrees.values():
        if degree in degree_distribution_dict:
            degree_distribution_dict[degree] += 1
        else:
            degree_distribution_dict[degree] = 1

    return degree_distribution_dict

    # ----------------- END OF FUNCTION --------------------- #


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes with the specified centrality.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    # Pass the string of passed as a parameter to lower
    metric = metric.lower()

    # Set the centrality centrality
    if "degree" in metric:
      centrality = nx.degree_centrality(g)
    elif "betweenness" in metric:
      centrality = nx.betweenness_centrality(g)
    elif "closeness" in metric:
      centrality = nx.closeness_centrality(g)
    elif "eigenvector" in metric:
      centrality = nx.eigenvector_centrality(g)
    elif "katz" in metric:
      centrality = nx.katz_centrality(g)
    elif "pagerank" in metric:
      centrality = nx.pagerank(g)
    elif "harmonic" in metric:
      centrality = nx.harmonic_centrality(g)
    elif "load" in metric:
      centrality = nx.load_centrality(g)
    
    # Calculate the K most central nodes with the corresponding accuracy.
    most_central_nodes = sorted(centrality, key = centrality.get, reverse = True)[:num_nodes]

    return most_central_nodes

    # ----------------- END OF FUNCTION --------------------- #


def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Convert the directed graph to undirected, preventing programming
    g = g.to_undirected()

    # Get the cliques of the graph
    cliques = nx.find_cliques(g)

    # Get all the cliques with the minimum size
    filtered_cliques = [l for l in cliques if len(l) >= min_size_clique]

    # Get all the nodes in all the cliques without repeating
    nodes = set()

    # Iterate over all cliques
    for clique in filtered_cliques:

      # Add each node in the clique to the set
      nodes.update(clique)

    return filtered_cliques, list(nodes)

    # ----------------- END OF FUNCTION --------------------- #


def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Convert the directed graph to undirected, preventing programming
    g = g.to_undirected()

    method = method.lower()

    if method == "louvain":
        partition_dict = community_louvain.best_partition(g)
        modularity = community_louvain.modularity(partition_dict, g)

        # Transform partition_dict into a list of communities
        partition = [[] for _ in range(max(partition_dict.values())+1)]
        for node, comm in partition_dict.items():
            partition[comm].append(node)

    elif method == "girvan_newman" or "girvan-newman":
        comp = nx.algorithms.community.girvan_newman(g)
        partition = tuple(sorted(c) for c in next(comp))  # first tuple of communities
        modularity = None  # you can add calculation of modularity here if needed

        # Create a dictionary that maps each node to the index of the community it belongs to
        partition_dict = {node: i for i, comm in enumerate(partition) for node in comm}

    else:
        raise ValueError("Unknown method: " + method)

    # Visualization
    pos = nx.spring_layout(g)
    cmap = cm.get_cmap('viridis', max(partition_dict.values()) + 1)
    nx.draw_networkx_nodes(g, pos, partition_dict.keys(), node_size=40, cmap=cmap, node_color=list(partition_dict.values()))
    nx.draw_networkx_edges(g, pos, alpha=0.5)
    plt.show()

    return partition, modularity

    # ----------------- END OF FUNCTION --------------------- #


if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    pass
    # ------------------- END OF MAIN ------------------------ #
