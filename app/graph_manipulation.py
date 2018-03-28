from snap import *
import unittest
import snap


def get_egonet(graph, node):
    """
    Creates the egonet graph of a given node in a graph
    :param graph: snap.py graph object
    :param node: int, the node id that the egonet graph is based on
    :return: spap.py graph object of the egonet graph of the given node
    """
    node_vec = snap.TIntV()
    snap.GetNodesAtHop(graph, node, 1, node_vec, False)

    # add the node that the egonet graph is based on
    node_vec.Add(node)

    # Get subgraph induced by the neighbors of given node.
    ego = snap.GetSubGraph(graph, node_vec)

    return ego


def get_line_graph(initial_graph):
    """
    Creates the line graph of a given graph
    :param initial_graph: snap.py graph object
    :return: spap.py graph object of the line graph
    """
    line_graph = snap.TUNGraph.New()

    nodes = dict()
    counter = 0

    # adding nodes in the new line-graph and store the neighbour nodes of each edge of the input graph
    for edge in initial_graph.Edges():
        # add edge as a node
        line_graph.AddNode(counter)

        # check if a set for a node already exists, if not create it and then add the counter
        try:
            nodes[edge.GetDstNId()].add(counter)
        except KeyError:
            nodes[edge.GetDstNId()] = set()
            nodes[edge.GetDstNId()].add(counter)

        try:
            nodes[edge.GetSrcNId()].add(counter)
        except KeyError:
            nodes[edge.GetSrcNId()] = set()
            nodes[edge.GetSrcNId()].add(counter)

        counter += 1

    # Adding edges in the new line-graph
    for n in nodes:
        l = list(nodes[n])

        # get the pairs of the nodes that each new node has as neighbours, and create an edge in the new line graph
        for i in range(len(l)-1):
            for j in range(i+1, len(l)):
                line_graph.AddEdge(l[i], l[j])

    return line_graph


def get_line_graph_old(graph):
    """
    Creates the line graph of a given graph
    :param graph: snap.py graph object
    :return: spap.py graph object of the line graph
    """
    j = 0
    new_nodes = set()
    new_edges_dict = dict()
    nodes_mapper = dict()
    line_graph = snap.TUNGraph.New()

    # create a dictionary with the new names of nodes for each edge
    for edge in graph.Edges():
        edge_tuple = (edge.GetSrcNId(), edge.GetDstNId())
        nodes_mapper[edge_tuple] = j
        line_graph.AddNode(j)
        j += 1

    # S = snap.TIntStrH()
    # S.AddDat(j,"("+str(edge.GetSrcNId())+","+str(edge.GetDstNId())+")")

    # create line graph
    for edge in graph.Edges():
        edge_tuple = (edge.GetSrcNId(), edge.GetDstNId())

        for other_edge in graph.Edges():
            other_edge_tuple = (other_edge.GetSrcNId(), other_edge.GetDstNId())

            if edge_tuple != other_edge_tuple:
                if edge_tuple[0] == other_edge_tuple[0] or edge_tuple[0] == other_edge_tuple[1] or edge_tuple[1] == \
                        other_edge_tuple[0] or edge_tuple[1] == other_edge_tuple[1]:

                    new_edges_dict[str(sorted(edge_tuple))] = sorted(other_edge_tuple)
                    new_nodes.add(str(sorted(other_edge_tuple)))
                    line_graph.AddEdge(nodes_mapper[(edge_tuple)], nodes_mapper[(other_edge_tuple)])

    return line_graph


class TestFunctions(unittest.TestCase):
    # Initialize a test graph
    graph = snap.TUNGraph.New()

    # Add nodes
    graph.AddNode(1)
    graph.AddNode(2)
    graph.AddNode(3)
    graph.AddNode(4)
    graph.AddNode(5)
    graph.AddNode(6)

    # Add Edges
    graph.AddEdge(1, 3)
    graph.AddEdge(1, 2)
    graph.AddEdge(3, 2)
    graph.AddEdge(4, 2)
    graph.AddEdge(4, 5)
    graph.AddEdge(6, 4)
    graph.AddEdge(6, 5)

    def test_get_egonet(self):
        egonet = get_egonet(self.graph, 2)  # get the egonet of node 2

        self.assertEqual(egonet.GetNodes(), 4)  # make sure the egonet has 4 nodes
        self.assertEqual(egonet.GetEdges(), 4)  # make sure the egonet has 4 edges

    def test_get_egonet_on_large_graph(self):
        # create a fully connected graph
        random_graph = snap.GenFull(snap.PUNGraph, 100)
        egonet = get_egonet(random_graph, 2)  # get the egonet of node 2

        self.assertEqual(egonet.GetNodes(), random_graph.GetNodes())  # make sure the egonet has the same nodes as random graph
        self.assertEqual(egonet.GetEdges(), random_graph.GetEdges())  # make sure the egonet has the same edges as random graph

    def test_get_line_graph(self):
        line_graph = get_line_graph(self.graph)  # get the line graph

        self.assertEqual(self.graph.GetEdges(),
                         line_graph.GetNodes())  # make sure the nodes of the line graph are as much as the edges of the graph

        # test if edges of the line graph is the same as the sum
        # of the degree * (degree - 1)/2 for each node of the input graph
        total_degree = 0
        for node in self.graph.Nodes():
            total_degree += node.GetOutDeg() * (node.GetOutDeg() - 1) / 2

        self.assertEqual(total_degree, line_graph.GetEdges())

    def test_get_line_graph_on_large_graph(self):
        # create a fully connected graph
        random_graph = snap.GenFull(snap.PUNGraph, 100)
        line_graph = get_line_graph(random_graph)  #  get the line graph

        self.assertEqual(random_graph.GetEdges(), line_graph.GetNodes())   # make sure the nodes of the line graph are as much as the edges of the graph

        # test if edges of the line graph is the same as the sum
        # of the degree * (degree - 1)/2 for each node of the input graph
        total_degree = 0
        for node in random_graph.Nodes():
            total_degree += node.GetOutDeg() * (node.GetOutDeg() - 1) / 2

        self.assertEqual(total_degree, line_graph.GetEdges())




if __name__ == '__main__':
    unittest.main()
