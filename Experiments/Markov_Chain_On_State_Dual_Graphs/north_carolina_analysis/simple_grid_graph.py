import math
import random
import tqdm
import copy
import numpy as np
import statistics
import os
import json
from collections import defaultdict

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from functools import partial
import networkx as nx


from gerrychain import MarkovChain, accept
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from Experiments.Markov_Chain_On_State_Dual_Graphs.north_carolina_analysis import chain_on_subsets_of_faces
from Experiments.Markov_Chain_On_State_Dual_Graphs.north_carolina_analysis import facefinder
import pickle

gn = 6
k = 5
ns = 50
p = 0.6
# BUILD GRAPH
def preprocessing(output_directory):

    graph = nx.grid_graph([k * gn, k * gn])

    for n in graph.nodes():
        graph.nodes[n]["population"] = 1

        graph.nodes[n]["pos"] = n
        graph.nodes[n]["C_X"] = n[0]
        graph.nodes[n]["C_Y"] = n[1]

        #if random.random() < p:
        # if n[0] < p * k * gn:  # vertical
        if n[1] < p * k * gn:
            graph.nodes[n]["blue"] = 1
            graph.nodes[n]["red"] = 0
        else:
            graph.nodes[n]["blue"] = 0
            graph.nodes[n]["red"] = 1
        if 0 in n or k * gn - 1 in n:
            graph.nodes[n]["boundary_node"] = True
            graph.nodes[n]["boundary_perim"] = 1

        else:
            graph.nodes[n]["boundary_node"] = False

    for u, v in graph.edges():
        graph[u][v]['style'] = "solid"
        graph[u][v]['cut_times'] = 0

    dual = facefinder.restricted_planar_dual(graph)

    cdict = {1: "blue", 0: "red"}

    plt.figure()
    nx.draw(
        graph,
        pos={x: x for x in graph.nodes()},
        node_color=[cdict[graph.nodes[x]["blue"]] for x in graph.nodes()],
        node_size=ns,
        node_shape="s",
    )
    plt.show()

    plt.figure()
    nx.draw(
        dual,
        pos=nx.get_node_attributes(dual, 'pos'),
        #node_color=[cdict[graph.nodes[x]["pink"]] for x in graph.nodes()],
        node_size=ns,
        node_shape="s",
    )
    plt.show()

    save_obj(graph, output_directory + '/', "original_graph")
    save_obj(dual, output_directory + '/', "initial_dual_graph")
    return graph, dual


def main():
    """ Contains majority of expermiment. Runs a markov chain on the state dual graph, determining how the distribution is affected to changes in the
     state dual graph.
     Raises:
        RuntimeError if PROPOSAL_TYPE of config file is neither 'sierpinski'
        nor 'convex'
    """
    output_directory = createDirectory(config)
    epsilon = config["epsilon"]
    k = config["NUM_DISTRICTS"]
    updaters = {'population': Tally('population'),
                'cut_edges': cut_edges,
                "Blue-Red": Election("Blue-Red", {"Blue": "blue", "Red": "red"})
                }
    graph, dual = preprocessing(output_directory)
    cddict = {x: int(x[0] / gn) for x in graph.nodes()}

    plt.figure()
    nx.draw(
        graph,
        pos={x: x for x in graph.nodes()},
        node_color=[cddict[x] for x in graph.nodes()],
        node_size=ns,
        node_shape="s",
        cmap="tab20",
    )
    plt.show()

    ideal_population = sum(graph.nodes[x]["population"] for x in graph.nodes()) / k
    faces = graph.graph["faces"]
    faces = list(faces)
    square_faces = [face for face in faces if len(face) == 4]
    totpop = 0
    for node in graph.nodes():
        totpop += int(graph.nodes[node]['population'])
    # length of chain
    steps = config["CHAIN_STEPS"]

    # length of each gerrychain step
    gerrychain_steps = config["GERRYCHAIN_STEPS"]
    # faces that are currently modified. Code maintains list of modified faces, and at each step selects a face. if face is already in list,
    # the face is un-modified, and if it is not, the face is modified by the specified proposal type.
    special_faces = set([face for face in square_faces if np.random.uniform(0, 1) < .5])
    #chain_output = {'dem_seat_data': [], 'rep_seat_data': [], 'score': []}
    chain_output = defaultdict(list)

    # start with small score to move in right direction
    print("Choosing", math.floor(len(faces) * config['PERCENT_FACES']), "faces of the dual graph at each step")
    max_score = -math.inf
    # this is the main markov chain
    for i in tqdm.tqdm(range(1, steps + 1), ncols=100, desc="Chain Progress"):
        special_faces_proposal = copy.deepcopy(special_faces)
        proposal_graph = copy.deepcopy(graph)
        if (config["PROPOSAL_TYPE"] == "sierpinski"):
            for i in range(math.floor(len(faces) * config['PERCENT_FACES'])):
                face = random.choice(faces)
                ##Makes the Markov chain lazy -- this just makes the chain aperiodic.
                if random.random() > .5:
                    if not (face in special_faces_proposal):
                        special_faces_proposal.append(face)
                    else:
                        special_faces_proposal.remove(face)
            chain_on_subsets_of_faces.face_sierpinski_mesh(proposal_graph, special_faces_proposal)
        elif (config["PROPOSAL_TYPE"] == "add_edge"):
            for j in range(math.floor(len(square_faces) * config['PERCENT_FACES'])):
                face = random.choice(square_faces)
                ##Makes the Markov chain lazy -- this just makes the chain aperiodic.
                if random.random() > .5:
                    if not (face in special_faces_proposal):
                        special_faces_proposal.add(face)
                    else:
                        special_faces_proposal.remove(face)
            chain_on_subsets_of_faces.add_edge_proposal(proposal_graph, special_faces_proposal)
        else:
            raise RuntimeError('PROPOSAL TYPE must be "sierpinski" or "convex"')

        initial_partition = Partition(proposal_graph, assignment=cddict, updaters=updaters)

        # Sets up Markov chain
        popbound = within_percent_of_ideal_population(initial_partition, epsilon)
        tree_proposal = partial(recom, pop_col=config['POP_COL'], pop_target=ideal_population, epsilon=epsilon,
                                node_repeats=1)

        # make new function -- this computes the energy of the current map
        exp_chain = MarkovChain(tree_proposal, Validator([popbound]), accept=accept.always_accept,
                                initial_state=initial_partition, total_steps=gerrychain_steps)
        seats_won_for_republicans = []
        seats_won_for_democrats = []
        for part in exp_chain:
            for u, v in part["cut_edges"]:
                proposal_graph[u][v]["cut_times"] += 1

            rep_seats_won = 0
            dem_seats_won = 0
            for j in range(k):
                rep_votes = 0
                dem_votes = 0
                for n in graph.nodes():
                    if part.assignment[n] == j:
                        rep_votes += graph.nodes[n]["blue"]
                        dem_votes += graph.nodes[n]["red"]
                total_seats_dem = int(dem_votes > rep_votes)
                total_seats_rep = int(rep_votes > dem_votes)
                rep_seats_won += total_seats_rep
                dem_seats_won += total_seats_dem
            seats_won_for_republicans.append(rep_seats_won)
            seats_won_for_democrats.append(dem_seats_won)

        seat_score = statistics.mean(seats_won_for_republicans)

        # implement mattingly simulated annealing scheme, from evaluating partisan gerrymandering in wisconsin
        if i <= math.floor(steps * .67):
            beta = i / math.floor(steps * .67)
        else:
            beta = (i / math.floor(steps * 0.67)) * 100
        temperature = 1 / (beta)

        weight_seats = 1
        weight_flips = -.2
        config['PERCENT_FACES'] = config['PERCENT_FACES']  # what is this?
        flip_score = len(special_faces)  # This is the number of edges being swapped

        score = weight_seats * seat_score + weight_flips * flip_score

        ##This is the acceptance step of the Metropolis-Hasting's algorithm. Specifically, rand < min(1, P(x')/P(x)), where P is the energy and x' is proposed state
        # if the acceptance criteria is met or if it is the first step of the chain
        def update_outputs():
            chain_output['dem_seat_data'].append(seats_won_for_democrats)
            chain_output['rep_seat_data'].append(seats_won_for_republicans)
            chain_output['score'].append(score)
            chain_output['seat_score'].append(seat_score)
            chain_output['flip_score'].append(flip_score)


        def propagate_outputs():
            for key in chain_output.keys():
                chain_output[key].append(chain_output[key][-1])

        if i == 1:
            update_outputs()
            special_faces = copy.deepcopy(special_faces_proposal)
        # this is the simplified form of the acceptance criteria, for intuitive purposes
        # exp((1/temperature) ( proposal_score - previous_score))
        elif np.random.uniform(0, 1) < (math.exp(score) / math.exp(chain_output['score'][-1])) ** (1 / temperature):
            update_outputs()
            special_faces = copy.deepcopy(special_faces_proposal)
        else:
            propagate_outputs()

        # if score is highest seen, save map.
        if score > max_score:
            # todo: all graph coloring for graph changes that produced this score
            nx.write_gpickle(proposal_graph, "obj/graphs/"+str(score)+'sc_'+str(config['CHAIN_STEPS'])+'mcs_'+ str(config["GERRYCHAIN_STEPS"])+ "gcs_" +
                config['PROPOSAL_TYPE']+'_'+ str(len(special_faces)), pickle.HIGHEST_PROTOCOL)
            save_obj(special_faces, output_directory, 'north_carolina_highest_found')
            nx.write_gpickle(proposal_graph, output_directory + '/' +  "max_score", pickle.HIGHEST_PROTOCOL)
            f=open(output_directory + "/max_score_data.txt","w+")
            f.write("maximum score: " + str(score) + "\n" + "edges changed: " + str(len(special_faces)) + "\n" + "Seat Score: " + str(seat_score))
            save_obj(special_faces, output_directory + '/', "special_faces")
            max_score = score

    plt.plot(range(len(chain_output['score'])), chain_output['score'])
    plt.xlabel("Meta-Chain Step")
    plt.ylabel("Score")
    plt.show()
    plt.close()

    plt.plot(range(len(chain_output['seat_score'])), chain_output['seat_score'])
    plt.xlabel("Meta-Chain Step")
    plt.ylabel("Number of average seats republicans won")
    plt.show()
    plt.close()


def save_obj(obj, output_directory, name):
    with open(output_directory + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def createDirectory(config):
    num = 0
    suffix = lambda x: f'-{x}' if x != 0 else ''
    while os.path.exists(config['EXPERIMENT_NAME'] + suffix(num)):
        num += 1
    os.makedirs(config['EXPERIMENT_NAME'] + suffix(num))
    metadataFile = os.path.join(config['EXPERIMENT_NAME'] + suffix(num), config['METADATA_FILE'])
    with open(metadataFile, 'w') as metaFile:  # should we have json file format in here?
        json.dump(config, metaFile, indent=2)
    return config['EXPERIMENT_NAME'] + suffix(num)


if __name__ == '__main__':
    global config
    config = {
        "X_POSITION": "C_X",
        "Y_POSITION": "C_Y",
        'PARTY_A_COL': "pink",
        'PARTY_B_COL': "purple",
        "UNDERLYING_GRAPH_FILE": "./plots/UnderlyingGraph.png",
        "WIDTH": 1,
#        "ASSIGN_COL": "part",
        "ASSIGN_COL": "cddict",
        "POP_COL": "population",
        'SIERPINSKI_POP_STYLE': 'random',
        'GERRYCHAIN_STEPS': 25,
        'CHAIN_STEPS': 50,
        "NUM_DISTRICTS": 5,
        'STATE_NAME': 'north_carolina',
        'PERCENT_FACES': .05,
        'PROPOSAL_TYPE': "add_edge",
        'epsilon': .01,
        "EXPERIMENT_NAME": 'experiments/grid_graph/edge_proposal',
        'METADATA_FILE': "experiment_data"
    }
    main()