import networkx as nx
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm
from ged4py.algorithm import graph_edit_dist
import statistics 
from utils import *
from model import *
from data import *
from args import Args
import create_graphs
#from pyswarm import pso

import matplotlib.pyplot as plt
from fastPFP import fastPFP_faster, loss
from timeit import default_timer as timer

def train_vae_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred,z_mu,z_lsgms = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
        loss = loss_bce + loss_kl
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss_bce.data[0], loss_kl.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_'+args.fname, loss_bce.data[0], epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' +args.fname, loss_kl.data[0], epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)

def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time = 1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)


    return G_pred_list


def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list



def train_mlp_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data, args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        #log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data
    return loss_sum/(batch_idx+1)


def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False,sample_time=1):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)


    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list



def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        for i in range(max_num_node):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            print('h:',h)
            print('y_pred_step:',y_pred_step)
            print('y_pred:',y_pred)
            x_step = sample_sigmoid_supervised_simple(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)
            print('x_step:',x_step)
            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda()
        y = Variable(y).cuda()

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j+1,y.size(2))
            loss += binary_cross_entropy_weight(y_pred[:,j,0:end_idx], y[:,j,0:end_idx])*end_idx


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.data[0]
    return loss_sum/(batch_idx+1)





## too complicated, deprecated
# def test_mlp_partial_bfs_epoch(epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):
#     rnn.eval()
#     output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#         for i in range(max_num_node):
#             # 1 back up hidden state
#             hidden_prev = Variable(rnn.hidden.data).cuda()
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)
#             y_pred_long[:, i:i + 1, :] = x_step
#
#             rnn.hidden = Variable(rnn.hidden.data).cuda()
#
#             print('finish node', i)
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()
#
#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()
        # update deterministic and lstm
        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
        feature_dim = y.size(1)*y.size(2)
        loss_sum += loss.data[0]*feature_dim
    return loss_sum/(batch_idx+1)



def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    for i in range(max_num_node):
        h = rnn(x_step)
        # output.hidden = h.permute(1,0,2)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(2))).cuda()
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = Variable(torch.zeros(test_batch_size,1,args.max_prev_node)).cuda()
        output_x_step = Variable(torch.ones(test_batch_size,1,1)).cuda()
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = Variable(output.hidden.data).cuda()
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).cuda()
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list




def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)

        output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),y_reshape[:,0:-1,0:1]),dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1,0,-1):
            count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
            output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        output_x = Variable(output_x).cuda()
        output_y = Variable(output_y).cuda()
        # print(output_y_len)
        # print('len',len(output_y_len))
        # print('y',y.size())
        # print('output_y',output_y.size())


        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).cuda()
        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
        output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
        output_y = pad_packed_sequence(output_y,batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
        # print(y_pred.size())
        feature_dim = y_pred.size(0)*y_pred.size(1)
        loss_sum += loss.data[0]*feature_dim/y.size(0)
    return loss_sum/(batch_idx+1)


########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output):
    #configure("runs/run-1234")
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()
        # train
        if 'GraphRNN_VAE' in args.note:
            train_vae_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_MLP' in args.note:
            train_mlp_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        elif 'GraphRNN_RNN' in args.note:
            train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                            optimizer_rnn, optimizer_output,
                            scheduler_rnn, scheduler_output)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    if 'GraphRNN_VAE' in args.note:
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_MLP' in args.note:
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size,sample_time=sample_time)
                    elif 'GraphRNN_RNN' in args.note:
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)

def compute_prob(x_step,y_pred_data):
    #compute the log probability
    p = 0
    for i,row in enumerate(x_step):
        for j, data in enumerate(row):
            if j <= i:
                if data == 1:
                    p = p+math.log10(y_pred_data[i,j])
                else:
                    p=p+math.log10(1-y_pred_data[i,j])
    return (p)
        
def compute_prob2(x_step,y_pred_data,i,idx):
    #compute the log probability
    p = 0
    y_temp = y_pred_data[idx][0]
    if np.count_nonzero(x_step[idx][0]):
        return -10000
    for j,data in enumerate(x_step[idx][0]):
        if j <= i:
            if data == 1:
                p = p+math.log10(y_temp[j])
            else:
                p=p+math.log10(1-y_temp[j])
    return (p)

def compute_distance(filename1,ground_truth):
    lam = 0.0
    G_pro1 = nx.read_edgelist(filename1)
    
    # compute MAE
    B2 = nx.to_numpy_matrix(ground_truth)
    B3 = nx.to_numpy_matrix(G_pro1)
    if (len(ground_truth) == len(G_pro1)):
        
        # compute MAE
        #X3 = fastPFP_faster(B3, B2, lam=lam, alpha=0.5,
        #               threshold1=1.0e-4, threshold2=1.0e-4,verbose=False)
        #loss_X = loss(B2, B3, X3)/(len(ground_truth)**2)

        #print("MAE = %s" % loss_X)
        # compute GED
        GED = graph_edit_dist.compare(G_pro1,ground_truth, False)
        #print("GED = %s" % graph_edit_dist.compare(G_pro1,ground_truth, False))        
        #for v in nx.optimize_graph_edit_distance(G_pro1, ground_truth, node_subst_cost=0):
        #    minv = v
        #GED = minv/(i*2)

    else:
        #print("loi: ",len(ground_truth), len(G_pro1))
        for i in range(len(ground_truth) - len(G_pro1)):
            G_pro1.add_node("dummy"+str(i))
        #B3 = nx.to_numpy_matrix(G_pro1)
        #X3 = fastPFP_faster(B3, B2, lam=lam, alpha=0.5, 
        #                    threshold1=1.0e-4, threshold2=1.0e-4,verbose=False)
        #loss_X = loss(B2, B3, X3)/(len(ground_truth)**2)
        # compute MAE
        #print("MAE = %s" % loss_X)
        # compute GED
        GED = graph_edit_dist.compare(G_pro1,ground_truth, False)
        #print("GED = %s" % GED)        
        
    return GED

def graphcomplete(truth, obs_graph, epoch, args, rnn, output, save_histogram=False,sample_time=1):
    rnn.eval()
    output.eval()
    G_pred_list = []
    size = len(truth)
    max_num_node = int(args.max_num_node)
    max_node = size
    currentprob = -10000
    for numiter in range(1):
        test_batch_size = 200
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        for i in range(max_node-1):
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            x_step = sample_sigmoid(y_pred_step, sample=True, thresh=0.5, sample_time=2)
            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()
    
        # save graphs as pickle
        for i in range(test_batch_size): 
            prob = compute_prob(y_pred_long_data[i].cpu().numpy(),y_pred_data[i].cpu().numpy())
            if prob > currentprob:
                adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
                G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
                G_pred_list.append(G_pred)
                filename1 = "bestgraph-"+str(i)+".txt"
                nx.write_edgelist(G_pred, filename1)
                print("i = ",i)
                compute_distance(filename1,truth)
                currentprob = prob
    return G_pred_list

def SPV(permut):
    nodelist = [];
    for i in range(len(permut)):
        pos = permut.tolist().index(min(permut))
        permut[pos]=100
        nodelist.append(pos)
    return nodelist
    

def optimized_permut(permut,*per_args):
    obs_graph = per_args[0]
    epoch = per_args[1]
    args = per_args[2]
    rnn = per_args[3] 
    output = per_args[4]
    data_loader = per_args[5]
    save_histogram = per_args[6]
    sample_time = per_args[7]
    nodedic = per_args[8]
    rnn.eval()
    output.eval()
    #graphs = [obs_graph]
    total_prob = 0
    G_pred_list = []
    max_num_node = int(args.max_num_node)
    node_odr = SPV(permut)
    nodelist = []
    
    # Convert to true node order
    for i in node_odr:
        nodelist.append(nodedic[i])
    
    # Adjacency matrix according to true node order
    y_raw = nx.adjacency_matrix(obs_graph, nodelist=nodelist)
    y_raw = encode_adj(y_raw.toarray(), is_full=True)
    y_raw_all = [[] for i in range(32)]
    for i in range(32):
        for j in range(max_num_node-1):
            y_raw_all[i].append(y_raw[j][0:j+1])
        
    for batch_idx, data in enumerate(data_loader):

        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        #print('y_pred_long:',y_pred_long)
        #print('y_pred:',y_pred)
        #print('x_step:',x_step)
        
        for i in range(max_num_node-1):
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            #print('h:',h)
            
            x_step = sample_sigmoid_supervised_simple(y_raw_all, y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)
            #for j in range(max_num_node-1):
            #    if y_raw_all[batch_idx][i][j] != 100:
            #        x_step[batch_idx][i][j] = y_raw_all[batch_idx][i][j]
            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).cuda()
            prob = compute_prob(y_pred_long[i].cpu().numpy(),y_pred[i].cpu().numpy())
            print(prob)
        y_pred_data = y_pred.data
        print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size): 
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            prob = compute_prob(y_pred_long_data[i].cpu().numpy(),y_pred_data[i].cpu().numpy())
            total_prob = total_prob + prob
    avg_prob = total_prob/test_batch_size
    return avg_prob

def graphcomplete_pso(obs_graph, epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):

    max_num_node = int(args.max_num_node)
    obs_graph=nx.read_edgelist("testdelgraph.txt")
    G_pred_list=[]
    print(len(obs_graph))
    
    # Add dummy nodes to the partially observable graph
    for nodeidx in range(max_num_node-len(obs_graph)):
        obs_graph.add_node(100+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(100+nodeidx,node,weight=100)
    print(len(obs_graph))
    nodelist=list(obs_graph)
    nodedic = {nodelist.index(x): x for x in nodelist}
    #A_pi = encode_adj_flexible(A)
    
    # Pass variables to PSO optimizer
    pyswarm_args = (obs_graph, epoch, args, rnn, output, data_loader, save_histogram,sample_time, nodedic)
    lb = np.empty(max_num_node)
    lb.fill(0)
    ub = np.empty(max_num_node)
    ub.fill(4)
    
    # Run PSO optimization
    xopt, fopt = pso(optimized_permut, lb, ub, f_ieqcons=None,args=pyswarm_args,minfunc=1e-1)
    
    print(xopt)
    print(fopt)
    # Return the best graphs
    return G_pred_list


def graphcomplete_greedy(size, obs_graph, epoch, args, rnn, output, data_loader, save_histogram=False,sample_time=1):

    rnn.eval()
    output.eval()
    G_pred_list = []
    
    max_num_node = int(args.max_num_node)
    max_node = size
    for e in obs_graph.edges(): 
        obs_graph[e[0]][e[1]]['weight'] = 1
    num_obs = len(obs_graph)
    obslist=list(obs_graph)
   # Add dummy nodes to the partially observable graph
    for nodeidx in range(max_node-len(obs_graph)):
        obs_graph.add_node(1000+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(1000+nodeidx,node,weight=100)
    print(len(obs_graph))
    V_unused=list(obs_graph)
    
    for numiter in range (5):
        obslist2=shuffle(V_unused)
    #nodedic = {nodelist.index(x): x for x in nodelist}
    #A_pi = encode_adj_flexible(A)
    # Adjacency matrix according to node order
        y_raw = nx.adjacency_matrix(obs_graph, nodelist=V_unused)
        y_raw = encode_adj(y_raw.toarray(), is_full=True)
        y_raw_all = []
        for m in range(max_node-1):
            y_raw_all.append(y_raw[m][0:m+1])
        maxlogprob = -1000000
        test_batch_size = 8
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        
        y_pred = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # discrete prediction
        x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
        #print('y_pred_long:',y_pred_long)
        #print('y_pred:',y_pred)
        #print('x_step:',x_step)
        reslist = []
        for i in range(max_node-1):
            #print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            y_step = F.sigmoid(y_pred_step)
            #print('h:',h)
            nodeid = V_unused[i]

            #x_step = sample_sigmoid(y_pred_step, True, thresh=0.5, sample_time=sample_time)
            #x_step = sample_sigmoid_supervised_greedy(y_raw_all[i], y_pred_step, current=i, sample_time=sample_time)
            x_step = sample_sigmoid_supervised_greedy2(obs_graph, reslist, nodeid, y_pred_step, current=i, sample_time=sample_time)
            reslist.append(nodeid)
            #for j in range(max_num_node-1):
            #    if y_raw_all[batch_idx][i][j] != 100:
            #        x_step[batch_idx][i][j] = y_raw_all[batch_idx][i][j]
            y_pred_long[:, i:i + 1, :] = x_step
            
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        
        y_pred_data = y_pred.data
        #print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size): 
            
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            prob = compute_prob(y_pred_long_data[i].cpu().numpy(),y_pred_data[i].cpu().numpy())
            if prob > maxlogprob:
                maxlogprob = prob
                filename1 = "bestgraph-"+str(i)+".txt"
                nx.write_edgelist(G_pred, filename1)
                filename2 = "ground_truth_ba.txt"
                print("i = ",i)
                compute_distance(filename1,filename2)
    return G_pred_list

def graphcomplete_greedy2(size, obs_graph, epoch, args, rnn, output, save_histogram=False,sample_time=1):

    rnn.eval()
    output.eval()
    G_pred_list = []
    
    max_num_node = int(args.max_num_node)
    max_node = size
    
    num_obs = len(obs_graph)
    obslist_org=list(obs_graph)
   # Add dummy nodes to the partially observable graph
    for e in obs_graph.edges(): 
        obs_graph[e[0]][e[1]]['weight'] = 1
    for nodeidx in range(max_node-len(obs_graph)):
        obs_graph.add_node(1000+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(1000+nodeidx,node,weight=100)
    print(len(obs_graph))
    V_unused=list(obs_graph)

    test_batch_size = 1
    rnn.hidden = rnn.init_hidden(test_batch_size)
    # generate graphs
    
    y_pred = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    #print('y_pred_long:',y_pred_long)
    #print('y_pred:',y_pred)
    #print('x_step:',x_step)
    for numiter in range(1):
        obslist = V_unused[:num_obs]
        reslist=[]
        unobslist = V_unused[num_obs:]
        print(obslist)
        print(unobslist)
        for i in range(max_node-1):
            maxlogprob = -1000000
            print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            y_step = F.sigmoid(y_pred_step)
            #print('h:',h)
            if (unobslist == []):
                #nodedic = {nodelist.index(x): x for x in nodelist}
                #A_pi = encode_adj_flexible(A)
                # Adjacency matrix according to node order
                for nodeid in obslist:
                    
                #x_step = sample_sigmoid(y_pred_step, True, thresh=0.5, sample_time=sample_time)
                    x_step = sample_sigmoid_supervised_greedy2(obs_graph, reslist, nodeid, y_pred_step, current=i, sample_time=sample_time)
                    #for j in range(max_num_node-1):
                    #    if y_raw_all[batch_idx][i][j] != 100:
                    #        x_step[batch_idx][i][j] = y_raw_all[batch_idx][i][j]
                    prob = compute_prob2(x_step.data.cpu().numpy(),y_step.data.cpu().numpy(),i,0)
                    #print(prob)
                    if prob > maxlogprob:
                        maxlogprob = prob
                        maxnode = nodeid
                        y_pred_long[:, i:i + 1, :] = x_step
                obslist.remove(maxnode)
            else:
                x_step = sample_sigmoid_greedy(y_pred_step,sample=True,thresh=0.5, sample_time=1)
                y_pred_long[:, i:i + 1, :] = x_step
                maxnode = random.choice(unobslist)
                unobslist.remove(maxnode)
            reslist.append(maxnode)          
            #print(len(reslist))            
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        
        y_pred_data = y_pred.data
        #print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()
    
        # save graphs as pickle
        for i in range(test_batch_size): 
            
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            filename1 = "bestgraph-"+str(i)+".txt"
            nx.write_edgelist(G_pred, filename1)
            filename2 = "ground_truth_ba.txt"
            print("i = ",i)
            compute_distance(filename1,filename2)
    return G_pred_list

def graphcomplete_greedy3(size, obs_graph, epoch, args, rnn, output, save_histogram=False,sample_time=1):

    rnn.eval()
    output.eval()
    G_pred_list = []
    
    max_num_node = int(args.max_num_node)
    max_node = size
    
    num_obs = len(obs_graph)
    obslist_org=list(obs_graph)
   # Add dummy nodes to the partially observable graph
    for e in obs_graph.edges(): 
        obs_graph[e[0]][e[1]]['weight'] = 1
    for nodeidx in range(max_node-len(obs_graph)):
        obs_graph.add_node(1000+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(1000+nodeidx,node,weight=100)
    print(len(obs_graph))
    V_unused=list(obs_graph)

    test_batch_size = 1
    rnn.hidden = rnn.init_hidden(test_batch_size)
    # generate graphs
    
    y_pred = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    #print('y_pred_long:',y_pred_long)
    #print('y_pred:',y_pred)
    #print('x_step:',x_step)
    for numiter in range(25):
        obslist = V_unused[:num_obs]
        reslist=[]
        unobslist = V_unused[num_obs:]
        print(obslist)
        print(unobslist)
        for i in range(max_node-1):
            maxlogprob = -1000000
            #print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            y_step = F.sigmoid(y_pred_step)
            #print('h:',h)
            if ((i != 0) & (random.random() < len(obslist)/(len(V_unused)-i)) & (obslist != [])) | (unobslist == []):
                #nodedic = {nodelist.index(x): x for x in nodelist}
                #A_pi = encode_adj_flexible(A)
                # Adjacency matrix according to node order
                for nodeid in obslist:
                    
                #x_step = sample_sigmoid(y_pred_step, True, thresh=0.5, sample_time=sample_time)
                    x_step = sample_sigmoid_supervised_greedy2(obs_graph, reslist, nodeid, y_pred_step, current=i, sample_time=sample_time)
                    #for j in range(max_num_node-1):
                    #    if y_raw_all[batch_idx][i][j] != 100:
                    #        x_step[batch_idx][i][j] = y_raw_all[batch_idx][i][j]
                    prob = compute_prob2(x_step.data.cpu().numpy(),y_step.data.cpu().numpy(),i,0)
                    #print(prob)
                    if prob > maxlogprob:
                        maxlogprob = prob
                        maxnode = nodeid
                        y_pred_long[:, i:i + 1, :] = x_step
                obslist.remove(maxnode)
            else:
                x_step = sample_sigmoid_greedy(y_pred_step,sample=True,thresh=0.5, sample_time=1)
                y_pred_long[:, i:i + 1, :] = x_step
                maxnode = random.choice(unobslist)
                unobslist.remove(maxnode)
            reslist.append(maxnode)          
            #print(len(reslist))            
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        
        y_pred_data = y_pred.data
        #print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()
    
        # save graphs as pickle
        for i in range(test_batch_size): 
            
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            filename1 = "bestgraph-"+str(i)+".txt"
            nx.write_edgelist(G_pred, filename1)
            filename2 = "ground_truth_lfr_n2v_100.txt"
            print("i = ",i)
            compute_distance(filename1,filename2)
    return G_pred_list

def compute_prob_lc(obs_graph, reslist, nodeid, y_pred_data,i,idx):
    '''
    do low complexity computation of probability
    '''
    dv = 0
    y_temp = y_pred_data[idx][0]

    # using supervision
    update_values = np.zeros(len(reslist))
    j = 0
    if reslist == []:
        return -10000
    for node in reslist:
        if obs_graph.has_edge(nodeid,node):
            if obs_graph[node][nodeid]['weight'] != 100:
                update_values[j] = y_temp[j]
                dv = dv+math.log10((y_temp[j])/(1-y_temp[j]))
        j = j+1
        #y_result[i] = y[i]
    # supervision done
    #print(update_values)
    #print(dv)
    return (dv)

def graphcomplete_greedy_lc(truth, obs_graph, epoch, args, rnn, output, save_histogram=False,sample_time=1):
    start = timer()
    rnn.eval()
    output.eval()
    G_pred_list = []
    size = len(truth)
    max_num_node = int(args.max_num_node)
    max_node = size
    
    num_obs = len(obs_graph)
    obslist_org=list(obs_graph)
   # Add dummy nodes to the partially observable graph
    for e in obs_graph.edges(): 
        obs_graph[e[0]][e[1]]['weight'] = 1
    for nodeidx in range(max_node-len(obs_graph)):
        obs_graph.add_node(10000+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(10000+nodeidx,node,weight=100)
    #print(len(obs_graph))
    V_unused=list(obs_graph)

    test_batch_size = 1
    rnn.hidden = rnn.init_hidden(test_batch_size)
    # generate graphs
    
    y_pred = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    #print('y_pred_long:',y_pred_long)
    #print('y_pred:',y_pred)
    #print('x_step:',x_step)
    for numiter in range(1):
        obslist = V_unused[:num_obs]
        reslist=[]
        unobslist = V_unused[num_obs:]
        #print(obslist)
        #print(unobslist)
        L = []
        for i in range(max_node-1):
            maxlogprob = -1000000
            #print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            y_step = F.sigmoid(y_pred_step)
            #print('h:',h)
            if (unobslist == []):
            #if (obslist != []):
                #nodedic = {nodelist.index(x): x for x in nodelist}
                #A_pi = encode_adj_flexible(A)
                # Adjacency matrix according to node order
                for nodeid in L:
                    
                #x_step = sample_sigmoid(y_pred_step, True, thresh=0.5, sample_time=sample_time)
                    
                    #for j in range(max_num_node-1):
                    #    if y_raw_all[batch_idx][i][j] != 100:
                    #        x_step[batch_idx][i][j] = y_raw_all[batch_idx][i][j]
                    prob = compute_prob_lc(obs_graph, reslist, nodeid,y_step.data.cpu().numpy(),i,0)
                    #print(prob)
                    if prob > maxlogprob:
                        maxlogprob = prob
                        maxnode = nodeid
                if (maxlogprob < 0 and set(obslist) != set(L)):
                    maxnode = random.choice(list(set(obslist) - set(L)))

                x_step = sample_sigmoid_supervised_greedy2(obs_graph, reslist, maxnode, y_pred_step, current=i, sample_time=sample_time)
                while np.count_nonzero(x_step.cpu().numpy()) == 0:
                    #print("all zeros, test again!")
                    x_step = sample_sigmoid_supervised_greedy2(obs_graph, reslist, maxnode, y_pred_step, current=i, sample_time=sample_time)
                y_pred_long[:, i:i + 1, :] = x_step
                obslist.remove(maxnode)
                #print("max probability:", maxlogprob)
            else:
                x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                while np.count_nonzero(x_step.cpu().numpy()) == 0:
                    print("all zeros, again!")
                    x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                y_pred_long[:, i:i + 1, :] = x_step
                maxnode = random.choice(unobslist)
                unobslist.remove(maxnode)
            reslist.append(maxnode)
            L = list(set(L + list(obs_graph.neighbors(maxnode))) & set(obslist))
            #print(L)
            #print(len(reslist))            
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        
        y_pred_data = y_pred.data
        #print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()
    
        # save graphs as pickle
        for i in range(test_batch_size): 
            
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            filename1 = "bestgraph-"+str(i)+".txt"
            nx.write_edgelist(G_pred, filename1)
            print("i = ",i)
            compute_distance(filename1,truth)
    end = timer()
    print(end - start)
    return G_pred_list

def graphcomplete_greedy_lc_lp(truth, obs_graph, epoch, args, rnn, output, save_histogram=False,sample_time=1):
    start = timer()
    rnn.eval()
    output.eval()
    G_pred_list = []
    size = len(truth)
    max_num_node = int(args.max_num_node)
    max_node = size
    
    num_obs = len(obs_graph)
    obslist_org=list(obs_graph)
   # Add dummy nodes to the partially observable graph
    for e in obs_graph.edges(): 
        obs_graph[e[0]][e[1]]['weight'] = 1
    for nodeidx in range(max_node-len(obs_graph)):
        obs_graph.add_node(10000+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(10000+nodeidx,node,weight=100)
    #print(len(obs_graph))
    V_unused=list(obs_graph)
    #print(len(V_unused))
    test_batch_size = 1
    rnn.hidden = rnn.init_hidden(test_batch_size)
    # generate graphs
    
    y_pred = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    #print('y_pred_long:',y_pred_long)
    #print('y_pred:',y_pred)
    #print('x_step:',x_step)
    reslist=[]
    
    for numiter in range(1):
        obslist = V_unused[:num_obs]

        unobslist = V_unused[num_obs:]
        maxnode = random.choice(unobslist)
        unobslist.remove(maxnode)
        reslist.append(maxnode)
        #print(obslist)
        #print(unobslist)
        L = []
        for i in range(max_node-1):
            maxlogprob = -1000000
            #print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            y_step = F.sigmoid(y_pred_step)
            #print('h:',h)
            #print("Unobserved list: ",unobslist)
            if (unobslist == []):
            #if (obslist != []):
                #nodedic = {nodelist.index(x): x for x in nodelist}
                #A_pi = encode_adj_flexible(A)
                # Adjacency matrix according to node order
                for nodeid in L:
                    
                #x_step = sample_sigmoid(y_pred_step, True, thresh=0.5, sample_time=sample_time)
                    
                    #for j in range(max_num_node-1):
                    #    if y_raw_all[batch_idx][i][j] != 100:
                    #        x_step[batch_idx][i][j] = y_raw_all[batch_idx][i][j]
                    prob = compute_prob_lc(obs_graph, reslist, nodeid,y_step.data.cpu().numpy(),i,0)
                    #print(prob)
                    if prob > maxlogprob:
                        maxlogprob = prob
                        maxnode = nodeid
                if (maxlogprob < 0 and set(obslist) != set(L)):
                    maxnode = random.choice(list(set(obslist) - set(L)))
                # Link prediction come here
                    
                x_step = sample_sigmoid_supervised_greedy_lp(obs_graph, reslist, maxnode, y_pred_step, current=i, sample_time=sample_time)
                while np.count_nonzero(x_step.cpu().numpy()) == 0:
                    #print("all zeros, test again!")
                    x_step = sample_sigmoid_supervised_greedy_lp(obs_graph, reslist, maxnode, y_pred_step, current=i, sample_time=sample_time)
                y_pred_long[:, i:i + 1, :] = x_step
                obslist.remove(maxnode)
                #print("max probability:", maxlogprob)
            else:
                x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                #while np.count_nonzero(x_step.cpu().numpy()) == 0:
                #    print("all zeros, again!")
                #    x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                y_pred_long[:, i:i + 1, :] = x_step
                maxnode = random.choice(unobslist)
                unobslist.remove(maxnode)
            reslist.append(maxnode)
            
            #print("reslist:",reslist)
            L = list(set(L + list(obs_graph.neighbors(maxnode))) & set(obslist))
            #print(L)
            #print(len(reslist))            
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        
        y_pred_data = y_pred.data
        #print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()
    
        # save graphs as pickle
        """
        for i in range(test_batch_size): 
            
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            filename1 = "bestgraph-"+str(i)+".txt"
            nx.write_edgelist(G_pred, filename1)
            print("i = ",i)
            compute_distance(filename1,truth)
        """    
    end = timer()
    print(end - start)
    #print(len(reslist))
    #print(reslist)
    return (reslist, y_pred_data)

def graphcomplete_greedy_lc_lp_wc(truth, obs_graph, epoch, args, rnn, output, save_histogram=False,sample_time=1):
    start = timer()
    rnn.eval()
    output.eval()
    G_pred_list = []
    size = len(truth)
    max_num_node = int(args.max_num_node)
    max_node = size
    
    num_obs = len(obs_graph)
    obslist_org=list(obs_graph)
   # Add dummy nodes to the partially observable graph
    for e in obs_graph.edges(): 
        obs_graph[e[0]][e[1]]['weight'] = 1
    for nodeidx in range(max_node-len(obs_graph)):
        obs_graph.add_node(10000+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(10000+nodeidx,node,weight=100)
    #print(len(obs_graph))
    V_unused=list(obs_graph)
    #print(len(V_unused))
    test_batch_size = 1
    rnn.hidden = rnn.init_hidden(test_batch_size)
    # generate graphs
    
    y_pred = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    #print('y_pred_long:',y_pred_long)
    #print('y_pred:',y_pred)
    #print('x_step:',x_step)
    reslist=[]
    
    for numiter in range(1):
        obslist = V_unused[:num_obs]

        unobslist = V_unused[num_obs:]
        maxnode = random.choice(unobslist)
        unobslist.remove(maxnode)
        reslist.append(maxnode)
        #print(obslist)
        #print(unobslist)
        L = []
        for i in range(max_node-1):
            maxlogprob = -1000000
            #print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            y_step = F.sigmoid(y_pred_step)
            #print('h:',h)
            #print("Unobserved list: ",unobslist)
            if (unobslist == []):
            #if (obslist != []):
                #nodedic = {nodelist.index(x): x for x in nodelist}
                #A_pi = encode_adj_flexible(A)
                # Adjacency matrix according to node order
                for nodeid in L:
                    
                #x_step = sample_sigmoid(y_pred_step, True, thresh=0.5, sample_time=sample_time)
                    
                    #for j in range(max_num_node-1):
                    #    if y_raw_all[batch_idx][i][j] != 100:
                    #        x_step[batch_idx][i][j] = y_raw_all[batch_idx][i][j]
                    prob = compute_prob_lc(obs_graph, reslist, nodeid,y_step.data.cpu().numpy(),i,0)
                    #print(prob)
                    if prob > maxlogprob:
                        maxlogprob = prob
                        maxnode = nodeid
                if (maxlogprob < 0 and set(obslist) != set(L)):
                    maxnode = random.choice(list(set(obslist) - set(L)))
                # Link prediction come here
                    
                x_step = sample_sigmoid_supervised_greedy_lp(obs_graph, reslist, maxnode, y_pred_step, current=i, sample_time=sample_time)
                while np.count_nonzero(x_step.cpu().numpy()) == 0:
                    #print("all zeros, test again!")
                    x_step = sample_sigmoid_supervised_greedy_lp(obs_graph, reslist, maxnode, y_pred_step, current=i, sample_time=sample_time)
                y_pred_long[:, i:i + 1, :] = x_step
                obslist.remove(maxnode)
                #print("max probability:", maxlogprob)
            else:
                x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                #while np.count_nonzero(x_step.cpu().numpy()) == 0:
                #    print("all zeros, again!")
                #    x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                y_pred_long[:, i:i + 1, :] = x_step
                maxnode = random.choice(unobslist)
                unobslist.remove(maxnode)
            reslist.append(maxnode)
            
            #print("reslist:",reslist)
            L = list(set(L + list(obs_graph.neighbors(maxnode))) & set(obslist))
            #print(L)
            #print(len(reslist))            
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        
        y_pred_data = y_pred.data
        #print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()
    
        # save graphs as pickle
        
        for i in range(test_batch_size): 
            
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            filename1 = "bestgraph-DeepNC-"+str(i)+".txt"
            nx.write_edgelist(G_pred, filename1)
            #print("i = ",i)
            GED = compute_distance(filename1,truth)
            
    end = timer()
    print(end - start)
    #print(len(reslist))
    #print(reslist)
    return (reslist, y_pred_data, GED)

def graphcomplete_naive(truth, obs_graph_true, epoch, args, rnn, output, save_histogram=False,sample_time=1):
    start = timer()
    rnn.eval()
    output.eval()
    G_pred_list = []
    obs_graph = obs_graph_true.copy()
    size = len(truth)
    max_num_node = int(args.max_num_node)
    max_node = size
    
    num_obs = len(obs_graph)
    obslist_org=list(obs_graph)
   # Add dummy nodes to the partially observable graph
    for e in obs_graph.edges(): 
        obs_graph[e[0]][e[1]]['weight'] = 1
    for nodeidx in range(max_node-len(obs_graph)):
        obs_graph.add_node(10000+nodeidx)
        for node in obs_graph.nodes():
            obs_graph.add_edge(10000+nodeidx,node,weight=100)
    #print(len(obs_graph))
    V_unused=list(obs_graph)

    test_batch_size = 1
    rnn.hidden = rnn.init_hidden(test_batch_size)
    # generate graphs
    
    y_pred = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_node, args.max_prev_node)).cuda() # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
    #print('y_pred_long:',y_pred_long)
    #print('y_pred:',y_pred)
    #print('x_step:',x_step)
    for numiter in range(1):
        obslist = V_unused[:num_obs]
        reslist=[]
        unobslist = V_unused[num_obs:]
        #print(obslist)
        #print(unobslist)
        L = []
        for i in range(max_node-1):
            #print('finish node',i)
            h = rnn(x_step)
            y_pred_step = output(h)
            y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
            y_step = F.sigmoid(y_pred_step)
            #print('h:',h)
            #print("Unobserved list: ",unobslist)
            if (obslist != []):
                maxnode = random.choice(list(set(obslist)))
                x_step = sample_sigmoid_supervised_greedy2(obs_graph, reslist, maxnode, y_pred_step, current=i, sample_time=sample_time)
                y_pred_long[:, i:i + 1, :] = x_step
                obslist.remove(maxnode)
            else:
                x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                #while np.count_nonzero(x_step.cpu().numpy()) == 0:
                #    print("all zeros, again!")
                #    x_step = sample_sigmoid(y_pred_step,sample=True,thresh=0.01, sample_time=1)
                y_pred_long[:, i:i + 1, :] = x_step
                maxnode = random.choice(unobslist)
                unobslist.remove(maxnode)
            reslist.append(maxnode)
            #print("reslist:",reslist)
         
            #print(L)
            #print(len(reslist))            
            rnn.hidden = Variable(rnn.hidden.data).cuda()
        
        y_pred_data = y_pred.data
        #print('y_pred_data:',y_pred_data)
        y_pred_long_data = y_pred_long.data.long()
    
        # save graphs as pickle
        for i in range(test_batch_size): 
            
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
            filename1 = "bestgraph-naive-"+str(i)+".txt"
            nx.write_edgelist(G_pred, filename1)
            GED = compute_distance(filename1,truth)
            print("Naive GED = ",GED)
    end = timer()
    print(end - start)
    return (G_pred_list, GED)

def graphcomplete_em(truth, obs_graph,pi,phi, samples, epoch, args, rnn, output,sample_time=1):
    start = timer()
    obs_temp = obs_graph.copy()
    obs_graph_samples = []
    pi_list = []
    phi_list = []
    GED = []
    samples = 10
    for i in range(samples):
        obs_graph_samples.append(obs_temp)
        pi_list.append(pi)
        phi_list.append(phi)
        GED.append(0)
    for e in obs_graph.edges(): 
        obs_temp[e[0]][e[1]]['weight'] = 1
    for interation in range(5):
        for non_e in nx.non_edges(obs_graph):
            weight = 0
            # update weight matrix
            for i in range(len(obs_graph_samples)):
                sourceindex = pi_list[i].index(non_e[0])
                desindex = pi_list[i].index(non_e[1])
                weightmatrix = phi_list[i].data[0].cpu().numpy()
                #print(weightmatrix)
                #print(weightmatrix.shape)
                
                if sourceindex < desindex:
                    weight = weight + weightmatrix[sourceindex][desindex]
                else:
                    weight = weight + weightmatrix[desindex][sourceindex]
            weight = weight/len(obs_graph_samples)
                # sampling
            for i in range(samples):
                obs_graph_samples[i] = obs_graph.copy()
                if random.random() > weight:
                    obs_graph_samples[i].add_edge(sourceindex,desindex)
            #print(weight)
        totalGED = 0
        for i in range(samples):
            #adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            (pi_list[i],phi_list[i], GED[i]) = graphcomplete_greedy_lc_lp_wc(truth, obs_graph_samples[i],epoch, args, rnn, output, sample_time=sample_time)
            totalGED = totalGED + GED[i]
        GED_avg = totalGED/samples
        print("Average GED = ", GED_avg)
        print("Stdev = ",statistics.stdev(GED))
    for non_e in nx.non_edges(obs_graph):
        weight = 0
        # update weight matrix
        for i in range(len(obs_graph_samples)):
            sourceindex = pi_list[i].index(non_e[0])
            desindex = pi_list[i].index(non_e[1])
            weightmatrix = phi_list[i].data[0].cpu().numpy()
            #print(weightmatrix)
            #print(weightmatrix.shape)
            
            if sourceindex < desindex:
                weight = weight + weightmatrix[sourceindex][desindex]
            else:
                weight = weight + weightmatrix[desindex][sourceindex]
        weight = weight/len(obs_graph_samples)
            # sampling
        for i in range(samples):
            obs_graph_samples[i] = obs_graph.copy()
            if random.random() > weight:
                obs_graph_samples[i].add_edge(sourceindex,desindex)
        #print(weight)
    for i in range(samples):
            #adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        (pi_list[i],phi_list[i], GED[i]) = graphcomplete_greedy_lc_lp_wc(truth, obs_graph_samples[i],epoch, args, rnn, output, sample_time=sample_time)        
    end = timer()
    print(end - start)
    return (reslist, phi)

########### for graph completion task
def train_graph_completion(args, dataset_test, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))

    for sample_time in range(1,2):
        if 'GraphRNN_MLP' in args.note:
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
            #G_pred = graphcomplete(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        if 'GraphRNN_VAE' in args.note:
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')

########### for graph completion task
def graph_completion(truth, obs_graph, args, rnn, output):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))
    
    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    obs_graph2 = obs_graph.copy()
    for sample_time in range(1,2):
        if 'GraphRNN_MLP' in args.note:
            #G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
            
            # naive approach suggest by a reviewer
            GED_naive = np.zeros(3)
            sumGEDnaive = 0
            for i in range(3):
                G_pred_naive, GED_naive[i] = graphcomplete_naive(truth, obs_graph,epoch, args, rnn, output, sample_time=sample_time)
                sumGEDnaive = sumGEDnaive + GED_naive[i]
            print("Average GED naive = ",sumGEDnaive/3)
            print("Stdev naive = ",statistics.stdev(GED_naive))
            #for original GraphRNN comparison
            #G_pred = graphcomplete(truth, obs_graph,epoch, args, rnn, output, sample_time=sample_time)
            #G_pred = graphcomplete_pso(obs_graph,epoch, args, rnn, output, dataset_test,sample_time=sample_time)
            #The greedy algorithm
            #G_pred = graphcomplete_greedy(size, obs_graph,epoch, args, rnn, output, dataset_test,sample_time=sample_time)
            #G_pred = graphcomplete_greedy2(size, obs_graph,epoch, args, rnn, output, sample_time=sample_time)
            #G_pred = graphcomplete_greedy3(size, obs_graph,epoch, args, rnn, output, sample_time=sample_time)
            #G_pred = graphcomplete_greedy_lc(truth, obs_graph,epoch, args, rnn, output, sample_time=sample_time)
            (pi,phi) = graphcomplete_greedy_lc_lp(truth, obs_graph2,epoch, args, rnn, output, sample_time=sample_time)
            
            (reslist, phi) = graphcomplete_em(truth, obs_graph,pi,phi, 2, epoch, args, rnn, output, sample_time=sample_time)
        #if 'GraphRNN_VAE' in args.note:
            #G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test,sample_time=sample_time)
        # save graphs
        fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat'
        save_graph_list(G_pred, fname)
    print('graph completion done, graphs saved')

########### for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output,graph_validate_len,graph_test_len, max_iter = 1000):
    fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
    rnn.load_state_dict(torch.load(fname))
    fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
    output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print('model loaded!, epoch: {}'.format(args.load_epoch))
    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len)+','+str(graph_test_len)+'\n')
        f.write('train,test\n')
        for iter in range(max_iter):
            if 'GraphRNN_MLP' in args.note:
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test)
            if 'GraphRNN_RNN' in args.note:
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test)
            print('train',nll_train,'test',nll_test)
            f.write(str(nll_train)+','+str(nll_test)+'\n')

    print('NLL evaluation done')
