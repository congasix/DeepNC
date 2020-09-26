# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 22:03:34 2018

@author: cong
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from fastPFP import fastPFP_faster, loss
from ged4py.algorithm import graph_edit_dist

#delGraph=nx.read_edgelist("delgraph_lfr_n2v_50.txt")
#print(len(delGraph))
outputEvo=nx.read_edgelist("recgraph_lfr_FF_Kron.txt")

#print(outputEvo.nodes())
print(len(outputEvo))

lam = 0.0
ground_truth=nx.read_edgelist("ground_truth_lfr2000.txt")
print(len(ground_truth))
nodelist = list(map(int, outputEvo.nodes()))

while (len(outputEvo) > len(ground_truth)):
    outputEvo.remove_node(max(outputEvo.nodes()))

print(len(outputEvo))
#B2 = nx.to_numpy_matrix(ground_truth)
#B3 = nx.to_numpy_matrix(outputEvo)
#X3 = fastPFP_faster(B3, B2, lam=lam, alpha=0.5,
                   threshold1=1.0e-4, threshold2=1.0e-4,verbose=False)
#loss_X = loss(B2, B3, X3)/(len(ground_truth)**2)
#print("MAE = %s" % loss_X)
print("GED = %s" % graph_edit_dist.compare(outputEvo,ground_truth, False)) 
"""
best=nx.read_edgelist("bestgraph-980.7232082116244.txt")
print(best.nodes())
print(len(best))

while (len(best) > len(ground_truth)):
    best.remove_node(max(best.nodes()))

print(len(best))
B4 = nx.to_numpy_matrix(best)
X4 = fastPFP_faster(B4, B2, lam=lam, alpha=0.5,
                   threshold1=1.0e-4, threshold2=1.0e-4,verbose=False)
loss_X = loss(B4, B2, X4)
print("Loss(X) new proposed graph = %s" % loss_X)
"""