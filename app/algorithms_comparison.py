from random import randint
import time
import snap
from pprint import pprint


def create_watts_strogatz(nodes, out_degree):
    """
    Generates and returns a random small-world graph using the Watts-Strogatz model.
    :param nodes: The number of nodes desired.
    :param out_degree: The out degree of each node desired. Since the generated graph is undirected, the out degree
    for each node, on average, will be twice this value.
    :return: A Snap.py undirected graph generated with the Watts-Strogatz model
    """
    rnd = snap.TRnd(1, 0)
    random_graph = snap.GenSmallWorld(nodes, out_degree, 0.7, rnd)

    return random_graph


def get_max_degree_node(input_graph):
    """
    Calculates the max degree of a graph
    :param input_graph: snap.py graph object
    :return: int, float, the id of the node with the highest degree
    along with the degree value
    """
    max_out_degree, max_node_id = 0, 0
    for node in input_graph.Nodes():
        if node.GetOutDeg() > max_out_degree:
            max_out_degree = node.GetOutDeg()
            max_node_id = node.GetId()

    return max_node_id, max_out_degree


def get_max_hub_authority(input_graph):
    """
    Calculates the maximum hub and authority scores based on Hits algorithm
    :param input_graph: snap.py graph object
    :return: float, int, float, int, the ids of the nodes with the highest hub
    and authority scores along with their values
    """
    max_hub_node, max_auth_node = 0, 0
    max_hub_score, max_auth_score = 0.0, 0.0

    hub_vec = snap.TIntFltH()
    authority_vec = snap.TIntFltH()
    snap.GetHits(input_graph, hub_vec, authority_vec)

    for node in hub_vec:
        if hub_vec[node] > max_hub_score:
            max_hub_node = node
            max_hub_score = hub_vec[node]

    for node in authority_vec:
        if authority_vec[node] > max_auth_score:
            max_auth_node = node
            max_auth_score = authority_vec[node]

    return max_hub_score, max_hub_node, max_auth_score, max_auth_node


def run_girvan_newman_algorithm(input_graph):
    """
    Girvan-Newman community detection algorithm
    :param input_graph: A Snap.py undirected graph.
    :return: float, the modularity of the network, float, float, minutes and seconds of the execution time
    """
    start = time.time()  # measure start time

    community_vec = snap.TCnComV()
    modularity = snap.CommunityGirvanNewman(input_graph, community_vec)  # measure modularity

    computation_time = time.time() - start  # measure execution time
    time_min, time_sec = divmod(computation_time, 60)

    return modularity, time_min, time_sec


def run_cnm_method(input_graph):
    """
    Clauset-Newman-Moore community detection method
    :param input_graph: A Snap.py undirected graph.
    :return: float, the modularity of the network, float, float, minutes and seconds of the execution time
    """
    start = time.time()  # measure start time

    community_vec = snap.TCnComV()
    modularity = snap.CommunityCNM(input_graph, community_vec)  # measure modularity

    computation_time = time.time() - start  # measure execution time
    time_min, time_sec = divmod(computation_time, 60)

    return modularity, time_min, time_sec


def get_top_page_rank(input_graph, top_number):
    """
    Computes the PageRank score of every node in Graph and returns the top nodes
    :param input_graph: A Snap.py graph
    :param top_number: int, the number of top nodes that will be returned
    :return: a list of tuples with the node_id and the PageRank score, a list of ints with the top ids
    """
    page_rank = snap.TIntFltH()  # a hash table of int keys and float values
    snap.GetPageRank(input_graph, page_rank)

    page_rank_list = list()
    for node in page_rank:
        page_rank_list.append((node, page_rank[node]))

    sorted_page_rank = sorted(page_rank_list, key=lambda x: x[1])

    top_node_ids = [x[0] for x in sorted_page_rank]

    return sorted_page_rank[:top_number], top_node_ids[:top_number]


def get_hub_authority_score(input_graph, top_nodes):
    """
    Calculates the hub and authority scores based on Hits algorithm for the given nodes
    :param input_graph: A Snap.py graph
    :param top_nodes: list, the nodes for which the metric will be returned
    :return: dict of dicts, with the top nodes as keys and their hub and authority dictionaries with their values
    """

    hub_vec = snap.TIntFltH()
    authority_vec = snap.TIntFltH()
    snap.GetHits(input_graph, hub_vec, authority_vec)

    top_dict = dict()
    for top in top_nodes:
        top_dict[top] = dict()
        top_dict[top]['authority'] = authority_vec[top]
        top_dict[top]['hub'] = authority_vec[top]

    return top_dict


def get_betweeness_score(input_graph, top_nodes):
    """
    Calculates and returns betweenness centrality of the given nodes
    :param input_graph: A Snap.py graph
    :param top_nodes: list, the nodes for which the metric will be returned
    :return: dict, with the top nodes as keys and their betweeness as values
    """
    nodes_vec = snap.TIntFltH()
    edges_vec = snap.TIntPrFltH()
    snap.GetBetweennessCentr(input_graph, nodes_vec, edges_vec, 1.0)

    top_dict = dict()
    for top in top_nodes:
        top_dict[top] = nodes_vec[top]

    return top_dict


def get_closeness_score(input_graph, top_nodes):
    """
    Calculates and returns closeness centrality of the given nodes
    :param input_graph: A Snap.py graph
    :param top_nodes: list, the nodes for which the metric will be returned
    :return: dict, with the top nodes as keys and their closeness as values
    """

    top_dict = dict()
    for top in top_nodes:
        close_centrality = snap.GetClosenessCentr(input_graph, top)
        top_dict[top] = close_centrality

    return top_dict


def run_iterative_process(node_increment):
    """
    Performs two community detection algorithms on an iterative process which increases the number of nodes of the
    graph and returns the last run graph parameters.
    :param node_increment: int, the number of nodes that each iteration will add to the new graph generation
    :return: parameters of the final graph
    """

    # Initialize nodes with 50 nodes
    nodes = 50

    # Generate random out-degree in [5,20]
    out_degree = randint(5, 20)

    # instantiating a graph variable
    graph_copy = 0

    total_computation_time, graph_computation_time = 0, 0

    # Run the script until computation time reaches "t-end" minutes
    while graph_computation_time < 60 * 10:
        try:
            print '-' * 30
            print 'Number of Nodes: {}, Out-degree: {}\n'.format(nodes, out_degree)

            # Generate the graph
            graph = create_watts_strogatz(nodes, out_degree)

            # Find the max degree node
            max_degree_id, max_degree = get_max_degree_node(graph)
            print 'Node id with max degree is: {}, with degree: {}'.format(max_degree_id, max_degree)

            # Find max hub and authority scores and the respective nodes
            hub_score, hub_node, auth_score, auth_node = get_max_hub_authority(graph)
            print 'Node id with max Hub score is: {}, with score: {}'.format(hub_node, hub_score)
            print 'Node id with max Authority score is: {}, with score: {}\n'.format(auth_node, auth_score)

            # Initialize time
            start_time = time.time()

            # Clauset-Newman-Moore community detection method
            modularity_cnm, cnm_min, cnm_sec = run_cnm_method(graph)
            print 'Clauset-Newman-Moore modularity is: {}, Computation time: {} min & {} sec'.format(modularity_cnm,
                                                                                                     int(cnm_min),
                                                                                                     cnm_sec)

            # Girvan-Newman community detection algorithm
            modularity_gn, gn_min, gn_sec = run_girvan_newman_algorithm(graph)
            print 'Girvan Newman modularity is: {}, Computation time: {} min & {} sec\n'.format(modularity_gn,
                                                                                                int(gn_min),
                                                                                                gn_sec)

            graph_computation_time = time.time() - start_time  # measure total execution time
            total_computation_time += graph_computation_time

            nodes += node_increment

            g_minutes, g_seconds = divmod(graph_computation_time, 60)

            print 'Total computation time: {} min & {} sec'.format(int(g_minutes), g_seconds)

        except MemoryError:
            print "\n\nMemory Error!"

    t_minutes, t_seconds = divmod(total_computation_time, 60)
    print '-' * 30
    print 'Final Graph had {} Nodes and Out-degree: {}\n' \
          'Node increment was: {}\n' \
          'Total script computation time was {} min & {} sec\n'.format(nodes - node_increment, out_degree,
                                                                       node_increment, int(t_minutes), t_seconds)

    return nodes, out_degree

if __name__ == '__main__':
    # run iterative process and get the final largest model
    final_nodes, final_out_degree = run_iterative_process(50)

    final_graph = create_watts_strogatz(final_nodes, final_out_degree)

    # calculate the top 30 nodes in terms of their Page Rank score
    top_30_pr, top_30_nodes = get_top_page_rank(final_graph, 30)

    # calculate Betweenness, Closeness, Authority and Hub scores for top 30 page_rank nodes
    betweeness_dict = get_betweeness_score(final_graph, top_30_nodes)
    closeness_dict = get_closeness_score(final_graph, top_30_nodes)
    hub_auth_dict = get_hub_authority_score(final_graph, top_30_nodes)

    # print(top_30_nodes)

    # create final dict results
    top_30_dict = dict()
    for id in top_30_pr:
        top_30_dict[id[0]] = dict()
        top_30_dict[id[0]]['pagerank'] = id[1]
        top_30_dict[id[0]]['betweeness'] = betweeness_dict[id[0]]
        top_30_dict[id[0]]['closeness'] = closeness_dict[id[0]]
        top_30_dict[id[0]]['authority'] = hub_auth_dict[id[0]]['authority']
        top_30_dict[id[0]]['hub'] = hub_auth_dict[id[0]]['hub']

    pprint(top_30_dict)
