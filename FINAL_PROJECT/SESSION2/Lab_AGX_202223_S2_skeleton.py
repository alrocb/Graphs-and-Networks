import networkx as nx
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

def cosine_similarity(feature_vec1, feature_vec2):
    """
    This function calculates the cosine similarity between two feature vectors. Cosine similarity is a measure of 
    similarity between two non-zero vectors, which measures the cosine of the angle between them.
    
    Args:
    feature_vec1 (numpy.array): First feature vector.
    feature_vec2 (numpy.array): Second feature vector.

    Returns:
    float: Cosine similarity score between feature_vec1 and feature_vec2.
    """
    # Calculate dot product of two feature vectors
    dot_product = np.dot(feature_vec1, feature_vec2)
    # Calculate norms of both feature vectors
    norm_product = np.linalg.norm(feature_vec1) * np.linalg.norm(feature_vec2)
    # Return cosine similarity (dot product divided by the product of norms)
    return dot_product / norm_product

def euclidean_similarity(feature_vec1, feature_vec2):
    """
    This function calculates the euclidean similarity (which is the inverse of the euclidean distance) between two 
    feature vectors. This measure is used to calculate the similarity between two vectors in the Euclidean space.
    
    Args:
    feature_vec1 (numpy.array): First feature vector.
    feature_vec2 (numpy.array): Second feature vector.

    Returns:
    float: Euclidean similarity score between feature_vec1 and feature_vec2.
    """
    # Calculate Euclidean distance between two feature vectors
    euclidean_dist = np.linalg.norm(feature_vec1 - feature_vec2)
    # Return Euclidean similarity (which is the inverse of Euclidean distance)
    return 1 / (1 + euclidean_dist)

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    # Create an undirected graph
    undirected_graph = nx.Graph()

    # Iterate over each edge in the directed graph
    for edge in g.edges():
        u, v = edge

        # Check if the reverse edge exists in the graph
        if g.has_edge(v, u):
            undirected_graph.add_edge(u, v)

    # Save the undirected graph to a file in graphml format
    nx.write_graphml(undirected_graph, out_filename)

    # Return the undirected graph
    return undirected_graph

    # ----------------- END OF FUNCTION --------------------- #


def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    # Create a copy of the input graph
    pruned_graph = g.copy()
    print('The original graph is a',g)

    # Find nodes with degree less than min_degree
    nodes_to_remove = []
    for node in g.nodes():
        node_degree=int(g.degree[node])

        if node_degree < min_degree:
            nodes_to_remove.append(node)

    # Remove nodes with degree less than min_degree from the pruned graph
    print('The nodes with lower degree than the minimum one are:',nodes_to_remove)
    pruned_graph.remove_nodes_from(nodes_to_remove)
    print('The first prune graph is a',pruned_graph)

    # Remove zero-degree nodes from the pruned graph
    zero_degree_nodes = []
    for node in pruned_graph.nodes():
        if pruned_graph.degree[node] == 0:
            zero_degree_nodes.append(node)
    pruned_graph.remove_nodes_from(zero_degree_nodes)

    print('The prune graph without zero-degree nodes is a',pruned_graph)
    # Save the pruned graph to a file in graphml format
    nx.write_graphml(pruned_graph, out_filename)

    # Return the pruned graph
    return pruned_graph

    # ----------------- END OF FUNCTION --------------------- #


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    if min_weight is None and min_percentile is None:
        raise ValueError("Either 'min_weight' or 'min_percentile' must be specified.")

    if min_weight is not None and min_percentile is not None:
        raise ValueError("Only one of 'min_weight' and 'min_percentile' can be specified.")

    pruned = g.copy()

    if min_weight is not None:
        edges_to_remove = []
        for u, v, w in pruned.edges(data='weight'):
            if w < min_weight:
                edges_to_remove.append((u, v))
    else:  # min_percentile is not None
        weight_values = []
        for u, v, w in pruned.edges(data='weight'):
            weight_values.append(w)
        threshold = np.percentile(weight_values, min_percentile)

        edges_to_remove = []
        for u, v, w in pruned.edges(data='weight'):
            if w < threshold:
                edges_to_remove.append((u, v))

    pruned.remove_edges_from(edges_to_remove)
    pruned.remove_nodes_from(list(nx.isolates(pruned)))

    if out_filename is not None:
        nx.write_graphml(pruned, out_filename)

    return pruned 

    # ----------------- END OF FUNCTION --------------------- #


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    mean_audio_features = {}  # Dictionary to store mean audio features for each artist

    # Convert string representations of dictionaries to actual dictionaries
    tracks_df["audio_feature"] = tracks_df["audio_feature"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    tracks_df["song_data"] = tracks_df["song_data"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    tracks_df["artists"] = tracks_df["artists"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    tracks_df['albums'] = tracks_df['albums'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Get unique artist names from the dataframe
    artist_names = tracks_df['artists'].apply(lambda x: x.get('name')).unique()

    for artist_name in artist_names:
        filtered_df = tracks_df[tracks_df['artists'].apply(lambda x: x.get('name') == artist_name)]
        artist_id = filtered_df["artists"].iat[0]['id']  # Get the artist ID from the first row

        grouped_audio_features = defaultdict(int)  # Defaultdict to store aggregated audio features
        num_songs = len(filtered_df)  # Number of songs for the artist

        for entry in filtered_df["audio_feature"].values:
            for key in entry:
                grouped_audio_features[key] += entry[key]  # Aggregate audio features

        # Calculate mean audio features
        mean_audio = {key: round(grouped_audio_features[key] / num_songs, 2) for key in grouped_audio_features}
        mean_audio_features[artist_id] = {
            "artist_id": artist_id,
            "artist_name": artist_name,
            "danceability": mean_audio['danceability'],
            "energy": mean_audio['energy'],
            "loudness": mean_audio['loudness'],
            "speechiness": mean_audio['speechiness'],
            "acousticness": mean_audio['acousticness'],
            "instrumentalness": mean_audio['instrumentalness'],
            "liveness": mean_audio['liveness'],
            "valence": mean_audio['valence'],
            "tempo": mean_audio['tempo']
        }

    df = pd.DataFrame.from_dict(mean_audio_features, orient="index")
    return df

    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> \
        nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    # Compute similarity matrix based on the selected similarity measure
    if similarity == 'cosine':
        similarity_matrix = cosine_similarity(artist_audio_features_df.iloc[:, 2:])
    elif similarity == 'euclidean':
        similarity_matrix = 1 / (1 + euclidean_distances(artist_audio_features_df.iloc[:, 2:]))

    # Create an undirected graph
    similarity_graph = nx.Graph()

    # Add artists as nodes to the graph
    for _, row in artist_audio_features_df.iterrows():
        artist_id = row['artist_id']
        artist_name = row['artist_name']
        similarity_graph.add_node(artist_id, name=artist_name)

    # Add weighted edges to the graph based on the similarity matrix
    num_artists = len(artist_audio_features_df)
    for i in range(num_artists):
        for j in range(i + 1, num_artists):
            artist_i = artist_audio_features_df.iloc[i]['artist_id']
            artist_j = artist_audio_features_df.iloc[j]['artist_id']
            similarity = similarity_matrix[i, j]
            similarity_graph.add_edge(artist_i, artist_j, weight=similarity)

    # Save the graph in graphml format if out_filename is provided
    if out_filename:
        nx.write_graphml(similarity_graph, out_filename)

    return similarity_graph

    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    
    """
    (a) Two undirected graphs (g′B and g′D) of artists obtained by applying the programmed function in exercise 1, retrieve bidirectional edges, to the graphs obtained by the crawler in session 1, gB and gD.
    """

    # Calling the function 'retrieve_bidirectional_edges' for graph gB and gD. 
    # This function is likely designed to identify and return all bidirectional edges present in the input graph.
    # The result for each graph is stored in variables gB_ and gD_, respectively.
    # Additionally, the names of the output .graphml files ("g'B.graphml" and "g'D.graphml") are passed to the function, 
    # which suggests that the function might also be writing the output graphs to these files.

    gB_ = retrieve_bidirectional_edges(nx.read_graphml("/Users/adrigarc/Downloads/Session1/gB.graphml"), "g'B.graphml")

    gD_ = retrieve_bidirectional_edges(nx.read_graphml("/Users/adrigarc/Downloads/Session1/gD.graphml"), "g'D.graphml")

    """
    (b) Two undirected graphs with weights (gwB and gwD) obtained from the similarity between the artists. To obtain them, it will be necessary to calculate the vector of average audio features for each artist (compute mean audio features), create a similarity graph with these features (create similarity graph), and prune the resulting graph (prune low weight edges) to achieve the desired size. Specifically, the size of the graph should be as similar as possible to the size of graphs g′B and g′D, respectively.
    """

    import pandas as pd

    # Read the CSV files into pandas DataFrames
    dfB = pd.read_csv('/Users/adrigarc/Downloads/Session2/needed/B.csv')
    dfD = pd.read_csv('/Users/adrigarc/Downloads/Session2/needed/C.csv')

    # Compute the mean audio features for each artist in DataFrame B
    B_ = compute_mean_audio_features(dfB)

    # Compute the mean audio features for each artist in DataFrame D
    D_ = compute_mean_audio_features(dfD)

    # Create a similarity graph for DataFrame B using Euclidean similarity measure
    B_sim = create_similarity_graph(B_, similarity='euclidean')

    # Create a similarity graph for DataFrame D using Euclidean similarity measure
    D_sim = create_similarity_graph(D_, similarity='euclidean')

    # Prune low-weight edges from the similarity graph of B and save to gwB.graphml
    gwB = prune_low_weight_edges(B_sim, min_weight=0.29525, out_filename='gwB.graphml')

    # Prune low-weight edges from the similarity graph of D and save to gwD.graphml
    gwD = prune_low_weight_edges(D_sim, min_weight=0.3, out_filename='gwD.graphml')

    # --- REPORT JUSTIFICATION ---

    """
    1. (0.5 points) Provide the order and size of the four obtained undirected graphs (g′B, g′D, gwB, and gwD).
    """

    # For graph gB_
    order_gB_ = gB_.number_of_nodes()
    size_gB_ = gB_.number_of_edges()

    # For graph gD_
    order_gD_ = gD_.number_of_nodes()
    size_gD_ = gD_.number_of_edges()

    # For graph gwB
    order_gwB = gwB.number_of_nodes()
    size_gwB = gwB.number_of_edges()

    # For graph gwD
    order_gwD = gwD.number_of_nodes()
    size_gwD = gwD.number_of_edges()

    print(f"Order and size of gB_: {order_gB_}, {size_gB_}")
    print(f"Order and size of gD_: {order_gD_}, {size_gD_}")
    print(f"Order and size of gwB: {order_gwB}, {size_gwB}")
    print(f"Order and size of gwD: {order_gwD}, {size_gwD}")

    """
    5. (0.5 points) Compute the size of the largest connected component from g′B and g′D. Which one is bigger? Justify the result.
    """

    # For graph gB_
    components_gB_ = nx.connected_components(gB_)
    largest_component_gB_ = max(components_gB_, key=len)
    size_largest_component_gB_ = len(largest_component_gB_)

    # For graph gD_
    components_gD_ = nx.connected_components(gD_)
    largest_component_gD_ = max(components_gD_, key=len)
    size_largest_component_gD_ = len(largest_component_gD_)

    print(f"Size of the largest connected component in gB_: {size_largest_component_gB_}")
    print(f"Size of the largest connected component in gD_: {size_largest_component_gD_}")

    if size_largest_component_gB_ > size_largest_component_gD_:
        print("The largest connected component of gB_ is bigger than gD_.")
    elif size_largest_component_gB_ < size_largest_component_gD_:
        print("The largest connected component of gD_ is bigger than gB_.")
    else:
        print("The largest connected components of gB_ and gD_ are of the same size.")

    # ------------------- END OF MAIN ------------------------ #
