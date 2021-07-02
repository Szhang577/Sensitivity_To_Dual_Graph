import json
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt

def main():
    suffix = lambda x: f'-{x}' if x != 0 else ''
    # path = os.path.join(config['PATH'], config['EXPERIMENT_NAME'] + suffix(config['num']), config['HIGH_SCORE'])
    path = os.path.join(config['EXPERIMENT_NAME'] + suffix(config['num']), config['MAX_SCORE'])
    # path = os.path.join(config['EXPERIMENT_NAME'], config['PICKLE_FILE'])
    # path = config['PATH']

    objectRep = open(path, "rb")

    graph = pickle.load(objectRep)
    print(graph)
    objectRep.close()

    cdict = {1: "blue", 0: "red"}
    line_style = nx.get_edge_attributes(graph, 'style').values()
    plt.figure()
    nx.draw(graph,
            pos={x: x for x in graph.nodes()},
            node_color=[cdict[graph.nodes[x]["blue"]] for x in graph.nodes()],
            edge_color=[graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()],
            node_size=1,
            style=line_style,
            cmap='magma')
    #nx.draw(graph, pos={x : x for x in graph.nodes()}, nodelist = vertex_list, node_color = "blue", node_size = 50, node_shape = "s")
    plt.show()

    path = os.path.join(config['EXPERIMENT_NAME'] + suffix(config['num']), config['ORIGINAL_FILE'])
    objectRep = open(path, "rb")

    graph = pickle.load(objectRep)
    print(graph)
    objectRep.close()

    plt.figure()
    nx.draw(graph,
            pos={x: x for x in graph.nodes()},
            node_color=[cdict[graph.nodes[x]["blue"]] for x in graph.nodes()],
            node_size=50,
            cmap='magma')
    plt.show()


if __name__ == '__main__':
    global config
    config = {
        "EXPERIMENT_NAME": 'experiments/grid_graph/edge_proposal',
        'PICKLE_FILE': "special_faces.pkl",
        'ORIGINAL_FILE': "original_graph.pkl",
        'HIGH_SCORE': "north_carolina_highest_found.pkl",
        'MAX_SCORE': "max_score",
        'PATH': 'Experiments/Markov_Chain_On_State_Dual_Graphs/north_carolina_analysis',
        'num': 26,
        'DUAL_FILE': "initial_dual_graph.pkl"
    }
    main()

