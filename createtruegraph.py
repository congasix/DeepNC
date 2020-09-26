# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:31:25 2019

@author: HP
"""

import networkx as nx
import numpy as np

G=nx.barabasi_albert_graph(200,8)
nx.write_edgelist(G, "ground_truth_ba200n.txt")
max_node = len(G)
while (len(G) > 0.7*max_node):
    random_node = np.random.choice(G.nodes())
    randnum = np.random.random_sample()
    if randnum < 1/G.number_of_nodes():
    #if randnum < 1/10:
        print(random_node)
        print(len(G))
        G.remove_node(random_node)
        
print(len(G))
nx.write_edgelist(G, "delgraph_ba200n_RN.txt")