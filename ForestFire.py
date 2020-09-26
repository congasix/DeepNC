from .snap import snap
import random
import sys
import signal
import csv
from operator import add
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    """
    G1 = snap.LoadEdgeList(snap.PUNGraph, "/root/Downloads/binary_networks/Youtube/del_Youtubetest.csv", 0, 1, '\t')
    print("G1: Nodes %d, Edges %d" % (G1.GetNodes(), G1.GetEdges()))
    numdel = 5000
    #snap.PlotInDegDistr(LoadGraph, "In Degree", "Twitter graph in degree")
    #Get nodes with high degree and then delete them
    InDegV = snap.TIntPrV()
    snap.GetNodeInDegV(G1, InDegV)
    del_node = 0
    V = snap.TIntV()
    for item in InDegV:
        if random.random() < 1:
            i = item.GetVal1()
            #print isinstance(i,int)
            V.Add(i)
            del_node = del_node + 1
        if del_node >= numdel:
            break
    snap.DelNodes(G1, V)
    print("G1: Nodes %d, Edges %d" % (G1.GetNodes(), G1.GetEdges()))
    snap.SaveEdgeList(G1, "/root/Downloads/binary_networks/Youtube/del300k_YoutubeRN2.csv")
    """
    """
    G2 = snap.LoadEdgeList(snap.PUNGraph, "/root/Downloads/binary_networks/10ksyn/network10k_1.dat", 0, 1, '\t')
    print "G2: Nodes %d, Edges %d" % (G2.GetNodes(), G2.GetEdges())
    numdel = 3000
    # snap.PlotInDegDistr(LoadGraph, "In Degree", "Twitter graph in degree")
    # Get nodes with high degree and then delete them
    NodeVec = snap.TIntV()
    for NI in G2.Nodes():
        for Id in NI.GetOutEdges():
            #print "edge (%d %d)" % (NI.GetId(),Id)
            if Id not in NodeVec:
                NodeVec.Add(Id)
        if len(NodeVec) > 7000: break
    print len(NodeVec)
    V = snap.TIntV()
    for node in G2.Nodes():
        if node.GetId() not in NodeVec:
            V.Add(node.GetId())
    snap.DelNodes(G2, V)

    print "G2: Nodes %d, Edges %d" % (G2.GetNodes(), G2.GetEdges())
    snap.SaveEdgeList(G2, "/root/Downloads/binary_networks/10ksyn/del_network10k_1_BFS.csv")

    def dfs(G,start):
        visited, stack = set(), [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                V = set()
                for node in G.Nodes():
                    if node.IsNbrNId(vertex):
                       V.add(node.GetId())
                stack.extend(V)
            if len(visited) > 6999:
                break
        return visited

    G3 = snap.LoadEdgeList(snap.PUNGraph, "/root/Downloads/binary_networks/10ksyn/network10k_1.dat", 0, 1, '\t')
    print "G3: Nodes %d, Edges %d" % (G3.GetNodes(), G3.GetEdges())
    NodeVec = dfs(G3,1)
    print len(NodeVec)
    V = snap.TIntV()
    for node in G3.Nodes():
        if node.GetId() not in NodeVec:
            V.Add(node.GetId())
    snap.DelNodes(G3, V)

    print "G3: Nodes %d, Edges %d" % (G3.GetNodes(), G3.GetEdges())
    snap.SaveEdgeList(G3, "/root/Downloads/binary_networks/10ksyn/del_network10k_1_DFS.csv")
"""

    # a function to perform Metropolis-Hasting random walk
    def ForestFire(G, seed, sampling):

        if seed.GetId() not in sampling:
            sampling.Add(seed.GetId())
        random.seed()
        # select a random number of neighbors of seed
        randNbr = random.randint(0,node.GetDeg()-1)
        for nbr in range(randNbr):
            neighborID = node.GetNbrNId(nbr)
            neighbor = G.GetNI(neighborID)
            if neighborID not in sampling:
                return ForestFire(G, neighbor,  sampling)

        return sampling

    G4 = snap.LoadEdgeList(snap.PUNGraph, "ground_truth_protein.txt", 0, 1, '\t')
    print("G4: Nodes %d, Edges %d" % (G4.GetNodes(), G4.GetEdges()))
    NodeVec = snap.TIntV()
    numiter=0
    size = 0.7*G4.GetNodes()
    while len(NodeVec) < size:
        numnode = len(NodeVec)
        random.seed()
        Rnd = snap.TRnd(42)
        Rnd.Randomize()
        nodeID = G4.GetRndNId(Rnd)
        node = G4.GetNI(nodeID)
        NodeVec = ForestFire(G4, node, NodeVec)
        numiter=numiter+1
        print(numiter)
        print(len(NodeVec))
        if numnode == len(NodeVec):
            break


    V = snap.TIntV()
    for node in G4.Nodes():
        if node.GetId() not in NodeVec:
            V.Add(node.GetId())
    snap.DelNodes(G4, V)

    print("G4: Nodes %d, Edges %d" % (G4.GetNodes(), G4.GetEdges()))
    snap.SaveEdgeList(G4, "delgraph_protein_FF.txt")
