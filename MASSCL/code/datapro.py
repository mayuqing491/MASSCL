import numpy as np
import os
import torch
import csv
import scipy.sparse as sp
import torch.utils.data.dataset as Dataset
import pandas as pd
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def mp_data(cd):

    D, M = cd.shape
    dc = cd.T

    mdm = np.matmul(cd, dc)
    dmd = np.matmul(dc, cd)

    mdmdm = np.matmul(mdm, mdm)
    dmdmd = np.matmul(dmd, dmd)

    return mdm, dmd, mdmdm, dmdmd

def mp_data1(cg):
    gc = cg.T
    cgc = np.matmul(cg, gc)
    return cgc

def loading_data(param):
    ratio = param.ratio

    md_matrix = pd.read_csv('/data/mayuqing/MSMCDA/datasets/1CircR2Disease/c_d.csv', encoding='utf-8-sig', header=None)

    rng = np.random.default_rng(seed=99)
    pos_samples = np.where(md_matrix == 1)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    rng = np.random.default_rng(seed=42)
    neg_samples = np.where(md_matrix == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]

    idx_split = int(n_pos_samples * ratio)

    test_pos_edges = pos_samples_shuffled[:, :idx_split]
    test_neg_edges = neg_samples_shuffled[:, :idx_split]

    test_pos_edges = test_pos_edges.T
    test_neg_edges = test_neg_edges.T

    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))


    train_pos_edges = pos_samples_shuffled[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))


    edge_idx_dict['train_Edges'] = train_edges#
    edge_idx_dict['train_Labels'] = train_true_label

    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label

    edge_idx_dict['true_md'] = md_matrix
    non_zero_indices = np.transpose(np.nonzero(md_matrix))

    edge_idx_dict['train_md'] = non_zero_indices
    # edge_idx_dict['edges']=np.vstack((train_edges, test_edges))

    return edge_idx_dict

def read_csv(path):
    with open(path, 'r', newline='',encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):

    edge_index = matrix.nonzero(as_tuple=False).t()
    return edge_index

def Simdata_pro(param):

    dataset = dict()


    cc_f_matrix = read_csv('/data/mayuqing/MSMCDA/Feature/1CircR2Disease/circFSim.csv')
    cc_f_edge_index = get_edge_index(cc_f_matrix)
    dataset['cc_f'] = {'data_matrix': cc_f_matrix.to(device), 'edges': cc_f_edge_index.to(device)}

    dd_s_matrix = read_csv('/data/mayuqing/MSMCDA/Feature/1CircR2Disease/disSSim.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_s'] = {'data_matrix': dd_s_matrix.to(device), 'edges': dd_s_edge_index.to(device)}

    cc_g_matrix = read_csv('/data/mayuqing/MSMCDA/Feature/1CircR2Disease/circGIPSim.csv')
    cc_g_edge_index = get_edge_index(cc_g_matrix)
    dataset['cc_g'] = {'data_matrix': cc_g_matrix.to(device), 'edges': cc_g_edge_index.to(device)}

    dd_g_matrix = read_csv('/data/mayuqing/MSMCDA/Feature/1CircR2Disease/disGIPSim.csv')
    dd_g_edge_index = get_edge_index(dd_g_matrix)
    dataset['dd_g'] = {'data_matrix': dd_g_matrix.to(device), 'edges': dd_g_edge_index.to(device)}

    cd_a_matrix = read_csv('/data/mayuqing/MSMCDA/datasets/1CircR2Disease/c_d.csv')
    cd_edge_index = get_edge_index(cd_a_matrix)
    dataset['cd']= {'data_matrix': cd_a_matrix.to(device), 'edges': cd_edge_index.to(device)}

    cdc,dcd,cdcdc,dcdcd =mp_data(cd_a_matrix)

    dataset['cdc']={'data_matrix': cdc}
    dataset['dcd']={'data_matrix': dcd}
    dataset['cdcdc']={'data_matrix': cdcdc}
    dataset['dcdcd']={'data_matrix': dcdcd}

    cdc_edge_index = get_edge_index(cdc)
    dataset['cdc_I'] = {'data_matrix': cdc.to(device), 'edges': cdc_edge_index.to(device)}
    dcd_edge_index = get_edge_index(dcd)
    dataset['dcd_I'] = {'data_matrix': dcd.to(device), 'edges': dcd_edge_index.to(device)}
    cdcdc_edge_index = get_edge_index(cdcdc)
    dataset['cdcdc_I'] = {'data_matrix': cdcdc.to(device), 'edges': cdcdc_edge_index.to(device)}
    dcdcd_edge_index = get_edge_index(dcdcd)
    dataset['dcdcd_I'] = {'data_matrix': dcdcd.to(device), 'edges': dcdcd_edge_index.to(device)}

    edges_cdc_cdcdc = torch.cat([dataset['cdc_I']['edges'], dataset['cdcdc_I']['edges']], dim=1)
    edges_cdc_cdcdc = torch.unique(edges_cdc_cdcdc, dim=1)
    dataset['cdc_cdcdc_I'] = {'edges': edges_cdc_cdcdc.to(device)}

    edges_dcd_dcdcd = torch.cat([dataset['dcd_I']['edges'], dataset['dcdcd_I']['edges']], dim=1)
    edges_dcd_dcdcd = torch.unique(edges_dcd_dcdcd, dim=1)
    dataset['dcd_dcdcd_I'] = {'edges': edges_dcd_dcdcd.to(device)}

    return dataset


class CVEdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):

        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label

