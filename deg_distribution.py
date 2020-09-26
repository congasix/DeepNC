# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 11:40:46 2019

@author: HP
"""

import collections
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from networkx.algorithms.community import LFR_benchmark_graph

# load a list of graphs
def load_graph_list(fname,is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
        """
    for i in range(len(graph_list)):
        edges_with_selfloops = graph_list[i].selfloop_edges()
        if len(edges_with_selfloops)>0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
            """
    return graph_list

#G = nx.read_edgelist("ground_truth_protein.txt")
"""
graphs2 = load_graph_list('graphs/' + 'GraphRNN_MLP_lfr_node2vec_4_128_test_0.dat')
G=graphs2[0].copy()
tau1 = 3
tau2 = 1.5
mu = 0.1
n = 2000
G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=20, min_community=690, seed=10)
"""
G = nx.read_edgelist("ground_truth_ba.txt")
print("This graph is connected?", nx.is_connected(G))
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

plt.loglog(deg, cnt,'b-',marker='o')
plt.title("Degree distribution plot")
plt.ylabel("count")
plt.xlabel("degree")

# draw graph in inset
plt.axes([0.45,0.45,0.45,0.45])
Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
pos=nx.circular_layout(Gcc)
plt.axis('off')
#nx.draw_networkx_nodes(Gcc,pos,node_size=20)
#nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

plt.savefig("degree_histogram.png")
plt.show()