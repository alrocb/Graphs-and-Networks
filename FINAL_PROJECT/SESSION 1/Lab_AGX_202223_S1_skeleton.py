import networkx as nx
import pandas as pd
import spotipy
from spotipy . oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
from statistics import median

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

# Define a function to enhance the input graph with additional metadata.
def enhance_graph_with_metadata(input_graph):
    # Initialize a dictionary to store artist attributes.
    artist_attributes_dict = {}

    # Loop through each node (artist) in the input graph.
    for artist_id in input_graph.nodes():
        # Fetch the artist's data from the Spotify API.
        artist_data = sp.artist(artist_id)

        # Organize the desired artist data into a dictionary.
        artist_metadata = {
            'artist_name': artist_data['name'],  # The name of the artist.
            'spotify_id': artist_data['id'],  # The Spotify ID of the artist.
            'follower_count': artist_data['followers']['total'],  # The total number of followers the artist has on Spotify.
            'popularity_score': artist_data['popularity'],  # The popularity score of the artist on Spotify.
            'associated_genres': str(artist_data['genres'])  # The genres associated with the artist.
        }

        # Store the artist metadata in the dictionary, using the artist's ID as the key.
        artist_attributes_dict[artist_id] = artist_metadata

    # Use the NetworkX function 'set_node_attributes' to add the artist metadata to the nodes of the input graph.
    nx.set_node_attributes(input_graph, artist_attributes_dict)

    # Return the graph with the newly added metadata.
    return input_graph

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def search_artist(artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    # Using Spotify's search function, we search for an artist by name.
    # This returns a JSON object containing matching results.
    results = sp.search(q=artist_name, type='artist')

    # We extract the artist's Spotify ID from the results.
    # The 'artists' key in the results dictionary contains a list of matching artists ('items').
    # We're assuming that the artist we want is the first one in the list (index 0).
    # Each artist in the list is a dictionary, and the 'id' key in that dictionary is the artist's Spotify ID.
    artist_id = results['artists']['items'][0]['id']

    # We return the artist's Spotify ID from the function.
    return artist_id

    # ----------------- END OF FUNCTION --------------------- #


def crawler(seed: str, max_nodes_to_crawl: int, strategy: str = "BFS", out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    artist_graph = nx.DiGraph()

    def add_artists(artist_id, crawled_nodes, traversal_order, visited_nodes):

        # Base case, when the crawled nodes = the nodes to crawl
        if crawled_nodes > max_nodes_to_crawl:
            return

        # Get the realted artist of the added artist
        related_artists = sp.artist_related_artists(artist_id)

        # Extract the realted artists in a list
        list_related = related_artists['artists']

        # --- Determine the scheduling algorithm ---

        # Breath first search
        if strategy == 'BFS':

                # Iterate over the related artists
                for related_artist in list_related:

                    # Obtain the ID of every artist
                    related_artist_id = related_artist['id']

                    # Check if the artist has already been visited.
                    if related_artist_id not in visited_nodes:

                      # Append the ID of the corresponding artist
                      traversal_order.append(related_artist_id)

                    # Add the corresponding node automatically, also add the adge which joins them
                    artist_graph.add_edge(artist_id, related_artist_id)
                
                next_one = traversal_order[crawled_nodes] # Indicate the next node to expand

        # Depth first search
        if strategy == 'DFS':
                
                # Create an auxiliary list for the related artist of the current crawled node.
                aux = list()

                # Iterate over the related artists
                for related_artist in list_related:

                    # Obtain the ID of every artist
                    related_artist_id = related_artist['id']

                     # Check if the artist has already been visited.
                    if related_artist_id not in visited_nodes:

                      # Append tha artist
                      aux.append(related_artist_id)

                    # Add the corresponding node automatically, also add the adge which joins them
                    artist_graph.add_edge(artist_id, related_artist_id)

                # In order to trasverse the list in the corresponding manner introduce the new nodes to search at the beggining
                traversal_order = aux + traversal_order
                # Get next the node to visit, first one in the traversal order by poping it from the traversal order list.
                next_one = traversal_order.pop(0)

        # Add to the list of visited nodes the node which has just been crawled
        visited_nodes.append(artist_id)
        # Recursivity call 
        add_artists(next_one, crawled_nodes + 1, traversal_order, visited_nodes)

    # List recursivity
    traversal_order = [seed]
    # Visited nodes
    visited_nodes = list()

    # Add the initial node
    add_artists(seed, 1, traversal_order, visited_nodes)

    # Add the corresponding properties to each node in the graph
    artist_graph = enhance_graph_with_metadata(artist_graph)

    nx.write_graphml(artist_graph, out_filename) # Save the graph with out_filename
    return artist_graph, visited_nodes # Return the graph

    # ----------------- END OF FUNCTION --------------------- #


def get_track_data(graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get track data for each visited artist in the graph.
    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.
    """
    # Initialize an empty dictionary to store the track data
    track_data = {}

    # Retrieve all unique artists from the input graphs
    visited_artists = {artist for graph in graphs 
                       for artist in graph.nodes if graph.out_degree(artist) > 0}

    # Iterate over each visited artist
    for artist_id in visited_artists:
        
        # Get the top tracks for the artist from Spotify
        top_tracks = sp.artist_top_tracks(artist_id, country='ES')

        # Iterate over each track in the top tracks
        for track in top_tracks["tracks"]:
            # Retrieve audio features for the track
            audio_features = sp.audio_features(track["id"])

            # Extract the desired track data
            track_info = {
                "id": track["id"],
                "duration_ms": track["duration_ms"],
                "name": track["name"],
                "popularity": track["popularity"]
            }

            # Extract the audio features for the track
            audio_info = {
                "danceability": audio_features[0]["danceability"],
                "energy": audio_features[0]["energy"],
                "loudness": audio_features[0]["loudness"],
                "speechiness": audio_features[0]["speechiness"],
                "acousticness": audio_features[0]["acousticness"],
                "instrumentalness": audio_features[0]["instrumentalness"],
                "liveness": audio_features[0]["liveness"],
                "valence": audio_features[0]["valence"],
                "tempo": audio_features[0]["tempo"]
            }

            # Extract album information for the track
            album_info = {
                "id": track["album"]["id"],
                "name": track["album"]["name"],
                "release_date": track["album"]["release_date"]
            }

            # Extract artist information
            artist_info = {
                "id": artist_id,
                "name": sp.artist(artist_id)["name"]
            }

            # Store the track data in the dictionary using the track ID as the key
            track_data[track["id"]] = {
                "track_info": track_info,
                "audio_info": audio_info,
                "album_info": album_info,
                "artist_info": artist_info
            }
        
    # Create a pandas DataFrame from the track data dictionary
    track_df = pd.DataFrame.from_dict(track_data, orient="index")

    # Save the DataFrame to a CSV file
    track_df.to_csv(out_filename)

    return track_df

    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    # Create an authentication manager using the SpotifyClientCredentials class.
    # client_id and client_secret are provided; these are specific to your application and would have been given when you registered your app with Spotify.
    auth_manager = SpotifyClientCredentials(client_id="2c397acf4c3c489dbdd63a0755bbd860", client_secret="f99adc9f87ad4145b67a648973b9212d")

    # Instantiate a Spotify API client using spotipy.
    # This client will use the previously created auth_manager to handle authentication with the Spotify API.
    sp = spotipy.Spotify(auth_manager=auth_manager)

    # --- EXERCISE 4 ---

    """
    (a) A graph of related artists starting with the artist Drake and exploring 200 artists with BFS (we will call this graph gB).
    """

    # Call the search_artist function with the string "Drake" as argument.
    # This function should return the Spotify ID of the artist Drake.
    drake_id = search_artist("Drake")

    # Set the number of nodes (artists) to crawl in the graph. In this case, 200.
    nodes_to_crawl = 200

    # Call the crawler function with the following arguments:
    # - seed (start point): Drake's Spotify ID,
    # - max_nodes_to_crawl: the number of nodes (artists) to crawl,
    # - strategy: "BFS", indicating that a Breadth-First Search strategy should be used.
    # This function should return a graph where the nodes represent artists and edges represent relationships between them.
    gB, crawled_nodes_gB = crawler(seed = drake_id, max_nodes_to_crawl = nodes_to_crawl, strategy= "BFS")

    """
    (b) A graph of related artists starting with the artist Drake and exploring 200 artists with DFS (we will call this graph gD).
    """

    # Call the search_artist function with the string "Drake" as argument.
    # This function should return the Spotify ID of the artist Drake.
    drake_id = search_artist("Drake")

    # Set the number of nodes (artists) to crawl in the graph. In this case, 200.
    nodes_to_crawl = 200

    # Call the crawler function with the following arguments:
    # - seed (start point): Drake's Spotify ID,
    # - max_nodes_to_crawl: the number of nodes (artists) to crawl,
    # - strategy: "DFS", indicating that a Depth-First Search strategy should be used.
    # This function should return a graph where the nodes represent artists and edges represent relationships between them.
    gD, crawled_nodes_gD = crawler(seed = drake_id, max_nodes_to_crawl = nodes_to_crawl, strategy= "DFS")

    """
    (c) A dataset of songs from all the explored artists that appear in any of the previous graphs (we will call this dataset D).
    """

    # Create a list with the previously created graphs
    graph_list = [gB, gD]

    # Call to the function get_track_data() in order to obtain the dataframe
    D = get_track_data(graph_list, out_filename = "D.csv")

    """
    (d) A graph of related artists starting with the artist French Montana and exploring 200 artists with BFS (we will call this graph hB).
    """

    # Call the search_artist function with the string "French Montana" as argument.
    # This function should return the Spotify ID of the artist French Montana.
    french_montana_id = search_artist("French Montana")

    # Set the number of nodes (artists) to crawl in the graph. In this case, 200.
    nodes_to_crawl = 200

    # Call the crawler function with the following arguments:
    # - seed (start point): Drake's Spotify ID,
    # - max_nodes_to_crawl: the number of nodes (artists) to crawl,
    # - strategy: "BFS", indicating that a Breadth-First Search strategy should be used.
    # This function should return a graph where the nodes represent artists and edges represent relationships between them.
    hB, crawled_nodes_hB = crawler(seed = french_montana_id, max_nodes_to_crawl = nodes_to_crawl, strategy= "BFS")

    """
    (e) A graph of related artists starting with the last crawled artist from gD and exploring 200 artists with BFS (we will call this graph fB).
    """

    last_crawled_gD = crawled_nodes_gD[-1]

    # Set the number of nodes (artists) to crawl in the graph. In this case, 200.
    nodes_to_crawl = 200

    # Call the crawler function with the following arguments:
    # - seed (start point): Last crawled node,
    # - max_nodes_to_crawl: the number of nodes (artists) to crawl,
    # - strategy: "BFS", indicating that a Breadth-First Search strategy should be used.
    # This function should return a graph where the nodes represent artists and edges represent relationships between them.
    fB, crawled_nodes_fB = crawler(seed = last_crawled_gD, max_nodes_to_crawl = nodes_to_crawl, strategy= "BFS", out_filename = "fB.graphml")

    # --- REPORT JUSTIFICATION ---

    order_gB = gB.number_of_nodes() # Order of gB
    order_gD = gD.number_of_nodes() # Order of gD

    size_gB = gB.number_of_edges() # Size of gB
    size_gD = gD.number_of_edges() # Size of gD

    print()
    print("--------------")
    print()
    print ("Order of gB: ", order_gB)
    print ("Order of gD: ", order_gD)
    print()
    print("--------------")
    print()
    print("Size of gB:", size_gB)
    print("Size of gD:", size_gD)
    print()
    print("--------------")
    print()

    # For graph gB
    in_degrees_gB = [d for n, d in gB.in_degree()]
    out_degrees_gB = [d for n, d in gB.out_degree()]

    # Compute minimum, maximum, median for in-degree and out-degree of graph gB
    min_in_degree_gB = min(in_degrees_gB)
    max_in_degree_gB = max(in_degrees_gB)
    median_in_degree_gB = median(in_degrees_gB)

    min_out_degree_gB = min(out_degrees_gB)
    max_out_degree_gB = max(out_degrees_gB)
    median_out_degree_gB = median(out_degrees_gB)

    print('gB in-degree: Min:', min_in_degree_gB, 'Max:', max_in_degree_gB, 'Median:', median_in_degree_gB)
    print('gB out-degree: Min:', min_out_degree_gB, 'Max:', max_out_degree_gB, 'Median:', median_out_degree_gB)
    print()
    print("--------------")
    print()

    # For graph gD
    in_degrees_gD = [d for n, d in gD.in_degree()]
    out_degrees_gD = [d for n, d in gD.out_degree()]

    # Compute minimum, maximum, median for in-degree and out-degree of graph gD
    min_in_degree_gD = min(in_degrees_gD)
    max_in_degree_gD = max(in_degrees_gD)
    median_in_degree_gD = median(in_degrees_gD)

    min_out_degree_gD = min(out_degrees_gD)
    max_out_degree_gD = max(out_degrees_gD)
    median_out_degree_gD = median(out_degrees_gD)

    print('gD in-degree: Min:', min_in_degree_gD, 'Max:', max_in_degree_gD, 'Median:', median_in_degree_gD)
    print('gD out-degree: Min', min_out_degree_gD, 'Max:', max_out_degree_gD, 'Median:', median_out_degree_gD)
    print()
    print("--------------")
    print()

    num_songs = len(D)
    print(f'The number of songs in the dataframe is {num_songs}.')

    # Unique artists
    unique_artists = D['artists'].apply(lambda x: x['id']).unique()
    num_unique_artists = len(unique_artists)
    print(f'The number of unique artists in the dataframe is {num_unique_artists}.')

    # Unique albums
    unique_albums = D['albums'].apply(lambda x: x['id']).unique()
    num_unique_albums = len(unique_albums)
    print(f'The number of unique albums in the dataframe is {num_unique_albums}.')
    print()
    print("--------------")
    print()

    # ------------------- END OF MAIN ------------------------ #
