import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from utils.utils import get_neighbor_finder


# from torch_geometric.data import Data
class Data:
    def __init__(self, sources, destinations, time_spans, current_time, parrent_time, edge_idxs, labels, node_features,
                 adj_list, id, node_depth, node_child, edge_index, BU_edge_index, x):
        self.sources = sources
        self.destinations = destinations
        self.time_spans = time_spans
        self.timestamps = current_time
        self.parrent_time = parrent_time
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)
        self.unique_features = node_features
        self.adj_list = adj_list
        self.id = id
        self.node_depth = node_depth
        self.node_child = node_child
        self.edge_index = edge_index
        self.BU_edge_index = BU_edge_index
        self.x = x

def find_depth(sources, destinations, node_size):
    node_depth = []
    node_child = []
    pa_dict = {}
    for ans in range(len(destinations)):
        pa_dict[destinations[ans]] = sources[ans]
    for i in range(node_size):
        child = 0
        for des, src in pa_dict.items():
            if src == i:
                child += 1
        node_child.append(child)
    for i in range(node_size):
        num = 1
        if i == 0 or i ==1:
            node_depth.append(0)
        else:
            x =  pa_dict[i]
            while x!=1 :
                num +=1
                x = pa_dict[x]
            node_depth.append(num)
    return node_depth, node_child
class UdGraphDataset(Dataset):
    def __init__(self, fold_x, droprate=0.2,
                 data_path=None):
        self.fold_x = fold_x
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]

        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        # if id == "581550667753504768":
        #     print(data['time_spa'])
        # 找各个比例用来做早期检测
        # 找到总数，再按比例分
        total_num = len(data['time_spa'])
        end = int(total_num / 5)*5
        # print(end)
        if end == 0 or end == 1:
            end = 2
        # edgeindex = data['edgeindex']
        edgeindex_0 = data['edgeindex'][0][:end-1]
        edgeindex_1 = data['edgeindex'][1][:end-1]
        edgeindex = np.array([edgeindex_0, edgeindex_1])
        edge_id = np.array(data['edg_index'])[:end]
        time_spa = np.array(data['time_spa'][1:end])
        cur_time = np.array(data['cur_time'])[1:end]
        par_time = np.array(data['par_time'])[1:end]
        node_features = np.array(data['x'])[:end]
        empty = np.zeros(node_features.shape[1])[np.newaxis, :]
        node_features = np.vstack([empty, node_features])
        label = int(data['y'])
        # print(edgeindex.shape)
        # print(edge_id.shape)
        # print(time_spa.shape)
        # print(node_features.shape)

        if self.droprate > 0:
            # dropout part
            length = len(edgeindex[0])
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            edge_id = edge_id[poslist]
            time_spa = time_spa[poslist]
            cur_time = cur_time[poslist]
            par_time = par_time[poslist]

            row = list(edgeindex[0][poslist])
            col = list(edgeindex[1][poslist])
            burow = list(edgeindex[1][poslist])
            bucol = list(edgeindex[0][poslist])
            sources = edgeindex[0][poslist] + 1
            destinations = edgeindex[1][poslist] + 1
            #
        else:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            sources = edgeindex[0] + 1
            destinations = edgeindex[1] + 1

        row.extend(burow)
        col.extend(bucol)
        new_edgeindex = np.array([row, col]).reshape(2, -1)
        node_depth, node_child = find_depth(sources, destinations, len(node_features))
        node_depth = np.array(node_depth)
        node_child = np.array(node_child)
        bunew_edgeindex = [burow, bucol]
        # print(new_edgeindex)
        # print(node_depth)
        # input()
        full_data = Data(sources, destinations, time_spa, cur_time, par_time, edge_id, label, node_features,
                         adj_list=new_edgeindex, id=id, node_depth=node_depth, node_child=node_child,
                         edge_index=torch.LongTensor(edgeindex), BU_edge_index=torch.LongTensor(bunew_edgeindex),
                         x=torch.tensor(data['x'][:end],dtype=torch.float32))
        return full_data


class UdTestGraphDataset(Dataset):
    def __init__(self, fold_x, droprate=0.2, subrate=0.2,
                 data_path=None):
        self.fold_x = fold_x
        self.data_path = data_path
        self.droprate = droprate
        self.subrate = subrate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)

        edgeindex = data['edgeindex']
        edge_id = np.array(data['edg_index'])[:-1]
        time_spa = np.array(data['time_spa'][1:])
        cur_time = np.array(data['cur_time'])[1:]
        par_time = np.array(data['par_time'])[1:]
        node_features = np.array(data['x'])

        subset_idxs = np.array(range(int(self.subrate * len(cur_time))))
        if len(subset_idxs) == 0:
            subset_idxs = np.array([0])
        edgeindex = np.array([edgeindex[0][subset_idxs], edgeindex[1][subset_idxs]])
        edge_id = edge_id[subset_idxs]
        time_spa = time_spa[subset_idxs]
        cur_time = cur_time[subset_idxs]
        par_time = par_time[subset_idxs]


        empty = np.zeros(node_features.shape[1])[np.newaxis, :]
        node_features = np.vstack([empty, node_features])
        label = int(data['y'])

        if self.droprate > 0:
            # dropout part
            length = len(edgeindex[0])
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            edge_id = edge_id[poslist]
            time_spa = time_spa[poslist]
            cur_time = cur_time[poslist]
            par_time = par_time[poslist]

            row = list(edgeindex[0][poslist])
            col = list(edgeindex[1][poslist])
            burow = list(edgeindex[1][poslist])
            bucol = list(edgeindex[0][poslist])
            sources = edgeindex[0][poslist] + 1
            destinations = edgeindex[1][poslist] + 1
            #
        else:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            burow = list(edgeindex[1])
            bucol = list(edgeindex[0])
            sources = edgeindex[0] + 1
            destinations = edgeindex[1] + 1

        row.extend(burow)
        col.extend(bucol)
        new_edgeindex = np.array([row, col]).reshape(2, -1)

        full_data = Data(sources, destinations, time_spa, cur_time, par_time, edge_id, label, node_features,
                         adj_list=new_edgeindex, id=id)
        return full_data

################################### load tree#####################################
def loadTree(dataname):
    data_eid = np.load('./labels.npy', allow_pickle=True)
    treeDic = []
    for eid in data_eid.item():
        treeDic.append(eid)
    return treeDic


################################# load data ###################################

def loadUdData(fold_x_train, fold_x_test):  # 加载的是双向 data
    data_path = './Twitter_15_and_16'
    print("loading train set")
    traindata_list = UdGraphDataset(fold_x_train, data_path=data_path)
    testdata_list = UdGraphDataset(fold_x_test, data_path=data_path)
    print("train no:", len(traindata_list))
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadFNNData(fold_x_train, fold_x_test):
    # data_path = './fakenewsnet'
    # data_path = './PHEME_new/'
    # data_path = './Weibo_finished/'
    # data_path = './Twitter_finished/'
    # data_path = './Poli_finished/'
    data_path = './Gossi_finished/'
    print("loading train set")
    traindata_list = UdGraphDataset(fold_x_train, data_path=data_path, droprate=0.0)
    # valdata_list = UdGraphDataset(fold_x_val, data_path=data_path, droprate=0.0)
    testdata_list = UdGraphDataset(fold_x_test, data_path=data_path, droprate=0.0)
    print("train no:", len(traindata_list))
    # print("val no:", len(valdata_list))
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


def loadPHEMEData(fold_x_train, fold_x_test):
    data_path = './PHEME/'
    print("loading PHEME dataset")
    traindata_list = UdGraphDataset(fold_x_train, data_path=data_path, droprate=0.0)
    testdata_list = UdGraphDataset(fold_x_test, data_path=data_path, droprate=0.0)
    print('train no: ', len(traindata_list))
    print('test no: ', len(testdata_list))
    return traindata_list, testdata_list


def loadFNNTestData(fold_x_train, fold_x_test, subrate=0.2):
    data_path = './fakenewsnet'
    print("loading train set")
    traindata_list = UdTestGraphDataset(fold_x_train, subrate=subrate, data_path=data_path, droprate=0.0)
    testdata_list = UdTestGraphDataset(fold_x_test, subrate=subrate, data_path=data_path, droprate=0.0)
    print("train no:", len(traindata_list))
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list