# export LD_PRELOAD="/lib64/libstdc++.so.6.0.21"
import gc
import time
import sys
import argparse
import warnings

import torch
# import torch_scatter
from resgcn_model import ResGCN_graphcl
from augmentation import augment
from pathlib import Path
import torch.nn.functional as F
from model.my_tgnh import TGN
from model.my_sgnh import SGN
from model.Transformer import *
from utils.utils import EarlyStopMonitor, get_neighbor_finder
from utils.dataset import loadTree, loadUdData, loadFNNData, loadFNNTestData
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import numpy as np
from torch_scatter import scatter_mean
from torch import nn
from sklearn import metrics
import torch.nn.init as init
import copy
from Transformer_utils import *
# Tensorboard
from torch.utils.tensorboard import SummaryWriter
# from graph_augs import Graph_Augmentor
writer = SummaryWriter()
from torch_geometric.nn import global_mean_pool, GATConv, SAGEConv, GCNConv, \
    global_max_pool, global_add_pool, GINConv, TopKPooling, global_sort_pool, GlobalAttention
from DisenGCN import DisenGCN, Cor_loss
from torch.nn import Linear, ReLU, Tanh, LeakyReLU

class hawkes(nn.Module):
    def __init__(self, d_model=770, num_types=1):
        super(hawkes, self).__init__()
        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)
        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        pass


class Gated_fusion(nn.Module):
    def __init__(self, input_size, out_size=1, dropout=0.2):
        super(Gated_fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, X1, X2):
        emb = torch.cat([X1.unsqueeze(dim=0), X2.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


class Net(nn.Module):
    def __init__(self, args, device):
        super(Net, self).__init__()
        self.device = device

        self.tgn = TGN(device=device,
                       n_layers=args.n_layer,
                       n_heads=args.n_head, dropout=args.drop_out, use_memory=False,
                       message_dimension=args.message_dim, memory_dimension=args.memory_dim,
                       memory_update_at_start=not args.memory_update_at_end,
                       embedding_module_type=args.embedding_module,
                       message_function_type=args.message_function_type,
                       aggregator_type=args.aggregator,
                       memory_updater_type=args.memory_updater,
                       n_neighbors=args.n_degree,
                       use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                       use_source_embedding_in_message=args.use_source_embedding_in_message,
                       dyrep=args.dyrep)
        self.sgn = SGN(device=device, n_layers=args.n_layer, n_heads=args.n_head, dropout=args.drop_out)
        self.Transformer = TransformerModel(ninp=768, nhead=2, nhid=768, nlayers=2, dropout=0.2)
        self.Transformer2 = TransformerModel(ninp=768 * 2, nhead=2, nhid=768, nlayers=2, dropout=0.2)
        self.fc2 = torch.nn.Linear(args.memory_dim, 2)

        self.fc1 = torch.nn.Linear(770, 2)  # Linear for forward Dynamic Interaction
        # self.BiNet = BiNet(768, 192, 192)
        self.fc3 = torch.nn.Linear(768 * 2, 768)

        self.act_tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        self.gated_fusion = Gated_fusion(input_size=768 + 2)
        self.batchsize = args.bs
        self.hawkes_module = hawkes(d_model=768, num_types=1)

    def forward(self, data, train_ngh_finder):
        """
        define the Transformer like BERT, take the 1 as result for classification.
        """
        # print(data.unique_features.shape)
        # print(data.sources.size)
        updated_embedding = self.tgn(train_ngh_finder, self.batchsize, data.sources, data.destinations,
                                     data.timestamps, data.unique_features, data.node_depth, data.edge_idxs,
                                     data.n_unique_nodes,
                                     data.adj_list)

        updated_embedding_struc1, updated_embedding_struc2 = self.sgn(data)

        avg_time = np.mean(data.time_spans)
        # print(type(avg_time))
        time = data.timestamps[-1]

        # print(type(time))
        temp_ele = np.append(avg_time, time)

        # print(temp_ele)
        temp_ele = (torch.from_numpy(temp_ele.astype(float)).to(self.device)).float().view(1,
                                                                                              -1)  # dtype=torch.float64
        updated_embedding = updated_embedding.unsqueeze(dim=1).float()
        # # updated_embedding_struc = updated_embedding_struc.unsqueeze(dim=1).float()
        timestamps = data.timestamps  # if len(data.timestamps) <=128 else data.timestamps[:128]
        timestamps = torch.from_numpy(timestamps).to(device)
        # print(timestamps.shape)

        node_depth = data.node_depth[1:]
        node_child = data.node_child[1:]
        over_depth = np.max(node_depth)
        num = node_depth.size
        node_struc = [node_child[i] * 0.1 + node_depth[i] * 0.9 for i in range(len(node_depth))]
        node_struc = np.array(node_struc)
        node_struc = torch.from_numpy(node_struc).to(device).float()

        struc_ele = np.append(over_depth, num)
        struc_ele = (torch.from_numpy(struc_ele.astype(float)).to(self.device)).float().view(1, -1)

        updated_embedding_time = self.Transformer(updated_embedding, timestamps, has_mask=True)

        event_ll = 0
        non_event_ll = 0


        out_feature_time = torch.mean(updated_embedding_time, dim=0)
        out_feature_time = out_feature_time.view(1,-1)

        out_feature_struc1 = updated_embedding_struc1
        out_feature_struc2 = updated_embedding_struc2
        out_feature_struc = torch.cat((out_feature_struc1, out_feature_struc2), dim=0)
        out_feature_struc = out_feature_struc.view(1, -1)
        #
        # # print(out_feature_struc.shape)
        # # print(temp_ele.shape)
        # # input()
        out_feature_time = torch.cat((out_feature_time, temp_ele), dim=1)
        out_feature_struc = self.fc3(out_feature_struc)
        out_feature_struc = torch.cat((out_feature_struc, struc_ele), dim=1)
        # print(out_feature_time.shape)
        # print(out_feature_struc.shape)
        # --------------------gated_fusion-----------------#
        out_feature = self.gated_fusion(out_feature_time, out_feature_struc)

        out_feature = self.fc1(self.act_tanh(out_feature))
        class_outputs = self.softmax(out_feature.view(1, -1))

        return class_outputs, event_ll, non_event_ll
torch.manual_seed(0)
np.random.seed(0)
np.set_printoptions(suppress=True)
### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--data', type=str, help='Dataset name',
                    default='Twitter')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='tgn-attn-weibo-ma', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=2, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true', default=False,
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function_type', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="TGNF_with_memory", help='Type of message '
                                                                               'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=768, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')

parser.add_argument('--use_destination_embedding_in_message', action='store_false',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_false',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--use_gcn', action='store_true',
                    help='Whether to run the GCN model')
parser.add_argument('--opt', type=str, default="RMSprop", choices=[
    "RMSprop", "Adam"], help='Type of optimizer')
parser.add_argument('--fd', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help='fold index')  # 5 denotes all data
parser.add_argument('--alpha', type=float, default=0, help='The coeffecient of multi-task learning.')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

print("Now using aggregator function is ", args.aggregator)
weight_decay = 1e-4
patience = 10

Path("./saved_models/{}/{}_fold/".format(args.aggregator, args.fd)).mkdir(parents=True, exist_ok=True)

get_model_path = lambda \
        epoch, train_accuracy, test_accuracy: f'./saved_models/{args.aggregator}/{args.fd}_fold/{epoch}-{train_accuracy}-{test_accuracy}.pth'

# Set device
device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# treeDic = loadTree(args.data)
# print("len(treeDic)", len(treeDic))
treeDic = None
# random.shuffle(treeDic)
fold5_x_train = []  # all data
fold5_x_test = []  # all data


# for i in treeDic:
#     fold5_x_train.append(i)
#     fold5_x_test.append(i)


def to_np(x):
    return x.cpu().detach().numpy()


def train_TGN(args, treeDic, x_test, x_train, weight_decay, patience, device):
    print('Training on device: ', device)
    model = Net(args, device)
    model = model.to(device)
    if args.opt == 'RMSprop':
        print("RMSprop")
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    else:
        print("Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()  # nn.NLLLoss()#

    traindata_list, testdata_list = loadFNNData(x_train, x_test)
    print("len(traindata_list)", len(traindata_list))
    print("len(testdata_list)", len(testdata_list))

    for epoch in range(args.n_epoch):
        start_epoch = time.time()
        num_item = 0
        total_train_loss = 0.0
        avg_train_loss = 0.0
        ok = 0
        train_pred = []
        train_true = []
        model = model.train()

        for item in traindata_list:
            num_item += 1
            optimizer.zero_grad()
            label = np.array([item.labels])
            label = torch.from_numpy(label).to(device)
            train_ngh_finder = get_neighbor_finder(item, uniform=False)
            model.tgn.set_neighbor_finder(train_ngh_finder)
            class_outputs, event_ll, non_event_ll = model(item, train_ngh_finder)
            class_loss = criterion(class_outputs, label.long())
            loss = class_loss
            loss.backward()
            optimizer.step()
            pred = torch.argmax(class_outputs, dim=1)

            if num_item % 1000 == 0:
                print("num_item", num_item)
            total_train_loss += class_loss

            if pred[0] == label[0]:
                ok += 1

            if num_item - 1 == 0:
                train_pred = to_np(pred)
                train_true = to_np(label)
            else:
                train_pred = np.concatenate((train_pred, to_np(pred)), axis=0)
                train_true = np.concatenate((train_true, to_np(label)), axis=0)

        print(metrics.classification_report(train_true, train_pred, digits=4, zero_division=1))
        print("total_train_loss:", total_train_loss)
        avg_train_loss = total_train_loss / num_item
        train_accuracy = round(ok / num_item, 3)

        num_item = 0
        ok = 0

        test_accuracy = 0.000
        test_pred = []
        test_true = []
        model = model.eval()
        for item in testdata_list:
            num_item += 1
            index = item.id
            label = np.array([item.labels])
            label = torch.from_numpy(label).to(device)
            test_ngh_finder = get_neighbor_finder(item, uniform=False)
            model.tgn.set_neighbor_finder(test_ngh_finder)
            class_outputs,_,_ = model(item, test_ngh_finder)
            pred = torch.argmax(class_outputs, dim=1)
            if pred[0] == label[0]:
                ok += 1

            if num_item - 1 == 0:
                test_pred = to_np(pred)
                test_true = to_np(label)
            else:
                test_pred = np.concatenate((test_pred, to_np(pred)), axis=0)
                test_true = np.concatenate((test_true, to_np(label)), axis=0)

        test_accuracy = round(ok / num_item, 3)
        print(metrics.classification_report(test_true, test_pred, digits=4, zero_division=1))

        epoch_time = (time.time() - start_epoch) / 60
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "Epoch id: {}, Epoch time: {:.3f} , avg_train_loss: {:.3f}, train_accuracy: {:.3f}, test_accuracy: {:.3f}".format(
                epoch, epoch_time, avg_train_loss, train_accuracy, test_accuracy))
        torch.save(model.state_dict(), get_model_path(epoch, round(train_accuracy, 3), round(test_accuracy, 3)))




if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # 5-fold cross validation.
    # fold0_x_test = np.load('fnn_5_fold_ids/weibo_fold0_test.npy')
    # fold0_x_train = np.load('fnn_5_fold_ids/weibo_fold0_train.npy')
    # fold1_x_test = np.load('fnn_5_fold_ids/weibo_fold1_test.npy')
    # fold1_x_train = np.load('fnn_5_fold_ids/weibo_fold1_train.npy')
    # fold2_x_test = np.load('fnn_5_fold_ids/weibo_fold2_test.npy')
    # fold2_x_train = np.load('fnn_5_fold_ids/weibo_fold2_train.npy')
    # fold3_x_test = np.load('fnn_5_fold_ids/weibo_fold3_test.npy')
    # fold3_x_train = np.load('fnn_5_fold_ids/weibo_fold3_train.npy')
    # fold4_x_test = np.load('fnn_5_fold_ids/weibo_fold4_test.npy')
    # fold4_x_train = np.load('fnn_5_fold_ids/weibo_fold4_train.npy')

    # fold0_x_test = np.load('fnn_5_fold_ids/Poli_fold0_test.npy')
    # fold0_x_train = np.load('fnn_5_fold_ids/Poli_fold0_train.npy')
    # fold1_x_test = np.load('fnn_5_fold_ids/Poli_fold1_test.npy')
    # fold1_x_train = np.load('fnn_5_fold_ids/Poli_fold1_train.npy')
    # fold2_x_test = np.load('fnn_5_fold_ids/Poli_fold2_test.npy')
    # fold2_x_train = np.load('fnn_5_fold_ids/Poli_fold2_train.npy')
    # fold3_x_test = np.load('fnn_5_fold_ids/Poli_fold3_test.npy')
    # fold3_x_train = np.load('fnn_5_fold_ids/Poli_fold3_train.npy')
    # fold4_x_test = np.load('fnn_5_fold_ids/Poli_fold4_test.npy')
    # fold4_x_train = np.load('fnn_5_fold_ids/Poli_fold4_train.npy')
    #
    fold0_x_test = np.load('fnn_5_fold_ids/Gossi_fold0_test.npy')
    fold0_x_train = np.load('fnn_5_fold_ids/Gossi_fold0_train.npy')
    fold1_x_test = np.load('fnn_5_fold_ids/Gossi_fold1_test.npy')
    fold1_x_train = np.load('fnn_5_fold_ids/Gossi_fold1_train.npy')
    fold2_x_test = np.load('fnn_5_fold_ids/Gossi_fold2_test.npy')
    fold2_x_train = np.load('fnn_5_fold_ids/Gossi_fold2_train.npy')
    fold3_x_test = np.load('fnn_5_fold_ids/Gossi_fold3_test.npy')
    fold3_x_train = np.load('fnn_5_fold_ids/Gossi_fold3_train.npy')
    fold4_x_test = np.load('fnn_5_fold_ids/Gossi_fold4_test.npy')
    fold4_x_train = np.load('fnn_5_fold_ids/Gossi_fold4_train.npy')

    is_train = True
    treeDic = 1
    if is_train:

        args.fd = 0
        train_TGN(args, treeDic, fold0_x_test, fold0_x_train, weight_decay, patience, device)
        args.fd = 1
        train_TGN(args, treeDic, fold1_x_test, fold1_x_train, weight_decay, patience, device)
        args.fd = 2
        train_TGN(args, treeDic, fold2_x_test, fold2_x_train, weight_decay, patience, device)
        args.fd = 3
        train_TGN(args, treeDic, fold3_x_test, fold3_x_train, weight_decay, patience, device)
        args.fd = 4
        train_TGN(args, treeDic, fold4_x_test, fold4_x_train, weight_decay, patience, device)


