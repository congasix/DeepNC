import networkx as nx
import numpy as np
from networkx.generators.community import LFR_benchmark_graph
import node2vec
from utils import *
from data import *
import random

def create(args):
### load datasets
    graphs=[]
    # synthetic graphs
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='ladder_small':
        graphs = []
        for i in range(2, 11):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    elif args.graph_type=='lfr':
        print('dung roi')
        graphs = []
        tau1 = 3
        tau2 = 1.5
        mu = 0.1
        for i in range(0, 50,1):
            n = 200
            G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)
            graphs.append(G)
            args.max_prev_node = 200
            print(len(G))
            print(len(graphs))
    elif args.graph_type=='lfr_single':
        print('lfr_single')
        graphs = []
        tau1 = 3
        tau2 = 1.5
        mu = 0.1
        for i in range(0, 2):
            n = 200
            G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=5, min_community=20, seed=10)
            graphs.append(G)
            args.max_prev_node = 200
            print(len(G))
            print(len(graphs))
    elif args.graph_type=='lfr2000':
        print('lfr2000')
        graphs = []
        tau1 = 3
        tau2 = 1.5
        mu = 0.1
        for i in range(1600,1601):
            for j in range(100):
                n = 1600
                G = LFR_benchmark_graph(1600, 3, 1.5, 0.1, average_degree=5, min_community=20, seed=10)
                graphs.append(G)
                args.max_prev_node = n
                print(len(G))
                print(len(graphs))
        #args.max_prev_node = 1600
    elif args.graph_type=='lfr_node2vec2':
        print('lfr2')
        args.max_prev_node = 0
        graphs = []
        tau1 = 3
        tau2 = 1.5
        mu = 0.1
        n = 2000
        G = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=20, min_community=690, seed=10)
        G2 = G
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        n2v_G = node2vec.Graph(G, 0, 1, 1)
        n2v_G.preprocess_transition_probs()
        nodes = list(G.nodes())
        random.shuffle(nodes)
        count = 0
        for count in range(500):
            walk = n2v_G.node2vec_walk(walk_length=200, start_node=nodes[0])
            count = count + 1
            H = G2.subgraph(walk)
            graphs.append(H)
            if len(H) > args.max_prev_node:
                args.max_prev_node = len(H)
            #print(len(H))
        print(len(graphs))
    elif args.graph_type=='gnutella':
        print('gnutella dataset')
        G = nx.read_edgelist("p2p-Gnutella04.txt")
        G2 = G
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        n2v_G = node2vec.Graph(G, 0, 1, 1)
        n2v_G.preprocess_transition_probs()
        nodes = list(G.nodes())
        random.shuffle(nodes)
        count = 0
        for node in nodes:
            walk = n2v_G.node2vec_walk(walk_length=300, start_node=node)
            count = count + 1
            if count > 200:
                break
            H = G2.subgraph(walk)
            for (n1, n2, d) in H.edges(data=True):
                d.clear()
            graphs.append(H)
            args.max_prev_node = len(H)
            print(len(H))
        print(len(graphs))
    elif args.graph_type=='facebook':
        print('facebook dataset')
        args.max_prev_node = 0
        for i in range(0,10):
            G = nx.read_edgelist("dataset/facebook/"+str(i)+".edges")
            graphs.append(G)
            args.max_prev_node = max(args.max_prev_node,len(G))
        print(len(graphs))
    elif args.graph_type=='facebook_single':
        print('facebook dataset')
        args.max_prev_node = 0
        for i in range(0,10):
            G = nx.read_edgelist("dataset/facebook/"+str(i)+".edges")
            if len(G) > 700:
                graphs.append(G)
            args.max_prev_node = max(args.max_prev_node,len(G))
        print(len(graphs))
    elif args.graph_type=='whois':
        print('whois dataset')
        G = nx.read_edgelist("tech-WHOIS.mtx")
        G2 = G
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        n2v_G = node2vec.Graph(G, 0, 1, 1)
        n2v_G.preprocess_transition_probs()
        nodes = list(G.nodes())
        random.shuffle(nodes)
        count = 0
        for node in nodes:
            walk = n2v_G.node2vec_walk(walk_length=300, start_node=node)
            count = count + 1
            if count > 200:
                break
            H = G2.subgraph(walk)
            for (n1, n2, d) in H.edges(data=True):
                d.clear()
            graphs.append(H)
            args.max_prev_node = len(H)
            print(len(H))
        print(len(graphs))
    elif args.graph_type=='celegans':
        print('celegans dataset')
        G = nx.read_edgelist("bio-celegans.mtx")
        G2 = G
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        n2v_G = node2vec.Graph(G, 0, 1, 1)
        n2v_G.preprocess_transition_probs()
        nodes = list(G.nodes())
        random.shuffle(nodes)
        count = 0
        for node in nodes:
            walk = n2v_G.node2vec_walk(walk_length=100, start_node=node)
            count = count + 1
            if count > 10:
                break
            H = G2.subgraph(walk)
            for (n1, n2, d) in H.edges(data=True):
                d.clear()
            graphs.append(H)
            args.max_prev_node = len(H)
            print(len(H))
        print(len(graphs))
    elif args.graph_type=='bio-grid':
        print('bio-grid dataset')
        G = nx.read_edgelist("bio-grid-human.edges",delimiter=',')
        G2 = G
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        n2v_G = node2vec.Graph(G, 0, 1, 1)
        n2v_G.preprocess_transition_probs()
        nodes = list(G.nodes())
        random.shuffle(nodes)
        count = 0
        for node in nodes:
            walk = n2v_G.node2vec_walk(walk_length=30, start_node=node)
            count = count + 1
            if count > 10:
                break
            H = G2.subgraph(walk)
            for (n1, n2, d) in H.edges(data=True):
                d.clear()
            graphs.append(H)
            args.max_prev_node = len(H)
            print(len(H))
        print(len(graphs))
    elif args.graph_type=='tree':
        graphs = []
        for i in range(2,5):
            for j in range(3,5):
                graphs.append(nx.balanced_tree(i,j))
        args.max_prev_node = 256
    elif args.graph_type=='caveman':
        # graphs = []
        # for i in range(5,10):
        #     for j in range(5,25):
        #         for k in range(5):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(30, 81):
                for k in range(10):
                    graphs.append(caveman_special(i,j, p_edge=0.3))
        args.max_prev_node = 100
    elif args.graph_type=='caveman_small':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(6, 11):
                for k in range(20):
                    graphs.append(caveman_special(i, j, p_edge=0.8)) # default 0.8
        args.max_prev_node = 20
    elif args.graph_type=='caveman_small_single':
        # graphs = []
        # for i in range(2,5):
        #     for j in range(2,6):
        #         for k in range(10):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        for i in range(2, 3):
            for j in range(8, 9):
                for k in range(100):
                    graphs.append(caveman_special(i, j, p_edge=0.5))
        args.max_prev_node = 20
    elif args.graph_type.startswith('community'):
        num_communities = int(args.graph_type[-1])
        print('Creating dataset with ', num_communities, ' communities')
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        #c_sizes = [15] * num_communities
        for k in range(3000):
            graphs.append(n_community(c_sizes, p_inter=0.01))
        args.max_prev_node = 80
    elif args.graph_type=='grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 40
    elif args.graph_type=='grid_small':
        graphs = []
        for i in range(2,5):
            for j in range(2,6):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 15
    elif args.graph_type=='barabasi':
        graphs = []
        for i in range(100,200):
             for j in range(4,5):
                 for k in range(5):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 199
    elif args.graph_type=='barabasi2000':
        graphs = []
        for i in range(1600,2001,20):
             for j in range(4,5):
                 for k in range(2):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 1800
    elif args.graph_type=='barabasi_95':
        graphs = load_graph_list('graphs/' + 'GraphRNN_MLP_ba95_4_128_test_0.dat')
        args.max_prev_node = 199
    elif args.graph_type=='barabasi_single':
        graphs = []
        for i in range(200,201):
             for j in range(4,5):
                 for k in range(1):
                    graphs.append(nx.barabasi_albert_graph(i,j))
    elif args.graph_type=='barabasi3':
        graphs = []
        for i in range(100,200):
             for j in range(4,5):
                 for k in range(8):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 130
    elif args.graph_type=='barabasi_small':
        graphs = []
        for i in range(4,21):
             for j in range(3,4):
                 for k in range(10):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        args.max_prev_node = 20
    elif args.graph_type=='grid_big':
        graphs = []
        for i in range(36, 46):
            for j in range(36, 46):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 90

    elif 'barabasi_noise' in args.graph_type:
        graphs = []
        for i in range(100,101):
            for j in range(4,5):
                for k in range(500):
                    graphs.append(nx.barabasi_albert_graph(i,j))
        graphs = perturb_new(graphs,p=args.noise/10.0)
        args.max_prev_node = 99

    # real graphs
    elif args.graph_type == 'enzymes':
        graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        args.max_prev_node = 25
    elif args.graph_type == 'enzymes_small':
        graphs_raw = Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        graphs = []
        for G in graphs_raw:
            if G.number_of_nodes()<=20:
                graphs.append(G)
        args.max_prev_node = 15
    elif args.graph_type == 'protein':
        graphs = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
        args.max_prev_node = 80
    elif args.graph_type == 'protein_95':
        graphs = load_graph_list('graphs/' + 'GraphRNN_MLP_pro95_4_128_test_0.dat')
        args.max_prev_node = 72
    elif args.graph_type == 'protein_single':
        graphsfull = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
        for i in range(1,100):
            if len(graphsfull[i]) > 100:
                print(len(graphsfull[i]))
                graphs.append(graphsfull[i])
                if len(graphs) > 1:
                    break
        args.max_prev_node = 80
    elif args.graph_type == 'DD':
        graphs = Graph_load_batch(min_num_nodes=100, max_num_nodes=500, name='DD',node_attributes=False,graph_labels=True)
        args.max_prev_node = 230
    elif args.graph_type == 'citeseer':
        _, _, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
        args.max_prev_node = 250
    elif args.graph_type == 'citeseer_single':
        _, _, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=3)
            if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
                graphs.append(G_ego)
                if len(graphs) > 1:
                    break
        args.max_prev_node = 250
    elif args.graph_type == 'citeseer_small':
        _, _, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        shuffle(graphs)
        graphs = graphs[0:200]
        args.max_prev_node = 15

    return graphs


