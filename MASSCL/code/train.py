import time
import torch
import random
from datapro import CVEdgeDataset
from model import MASSCL, EmbeddingM, EmbeddingD, ACMF, GRID
import os
import numpy as np
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import copy
import warnings
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import json
import os
import pandas as pd

from scipy.interpolate import make_interp_spline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import torch as th
from torch_geometric.data import Data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.to(device).manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):  ##*
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret

def get_metrics(score, label):
    y_pre = score
    y_true = label
    metric,fpr,tpr = caculate_metrics(y_pre, y_true)
    return metric,fpr,tpr


def caculate_metrics(pre_score, real_score):
    y_true = real_score
    y_pre = pre_score

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)

    best_f1 = -1
    best_threshold = 0.5  # fallback
    for th in thresholds:
        pred = (y_pre >= th).astype(int)
        current_f1 = metrics.f1_score(y_true, pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = th

    y_score = (y_pre >= best_threshold).astype(int)

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score, zero_division=0)
    recall = metrics.recall_score(y_true, y_score, zero_division=0)
    precision = metrics.precision_score(y_true, y_score, zero_division=0)

    metric_result = [auc, aupr, acc, f1, recall, precision, best_threshold]

    print("One epoch metric（Auto threshold）:")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"AUC: {auc:.4f}  AUPR: {aupr:.4f}  ACC: {acc:.4f}  F1: {f1:.4f}  Recall: {recall:.4f}  Precision: {precision:.4f}")

    return metric_result, fpr, tpr


def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))

def build_train_adj(edge_idx_dict, num_circRNA, num_disease, device='cpu'):

    adj_train = torch.zeros((num_circRNA, num_disease), dtype=torch.float32)

    train_edges = edge_idx_dict['train_Edges']  # shape [num_edges, 2]
    train_labels = edge_idx_dict['train_Labels']  # shape [num_edges]

    for idx, (c_idx, d_idx) in enumerate(train_edges):
        if train_labels[idx] == 1.0:
            adj_train[c_idx, d_idx] = 1.0

    adj_train = adj_train.to(device)
    return adj_train


def train_test(simData, train_data, param, state):
    epo_metric = []
    valid_metric = []
    train_losses = []
    valid_losses = []
    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']

    all_valid_metrics = []
    all_fpr = []
    all_tpr = []
    kfolds = param.kfold
    # edgeIndex = train_data
    # trainEdges = EdgeDataset(edgeIndex, True)
    # testEdges = EdgeDataset(edgeIndex, False)
    # kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
    # setup_seed(42)
    torch.manual_seed(99)

    # trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)

        num_c = train_data['true_md'].shape[0]
        num_d = train_data['true_md'].shape[1]

        for i in range(kfolds):
            train_losses = []
            valid_losses = []
            a = i + 1
            model = MASSCL(EmbeddingM(param), EmbeddingD(param), ACMF(param), GRID(param))  ##*
            # model.cuda()
            model.to(device)

            # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  ###
            optimizer = torch.optim.AdamW(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]


            adj_train = build_train_adj(
                {'train_Edges': edges_train, 'train_Labels': labels_train},
                num_c, num_d, device)

            trainEdges = CVEdgeDataset(edges_train, labels_train)
            validEdges = CVEdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

            valid_metric = []

            print("-----training-----")

            for e in range(param.epoch):
                running_loss = 0.0  ###
                epo_label = []
                epo_score = []
                print("epoch：", e + 1)
                model.train()
                start = time.time()

                alpha = max(0.1, 1 - e / param.epoch)
                beta = max(0.1, 1 - e / param.epoch)

                for f, item in enumerate(trainLoader):
                    data, label = item
                    train_data = data.to(device)
                    true_label = label.to(device)
                    (pre_score, losss, lossc,lossd,mFea1, dFea1) = model(simData, train_data, adj_train)
                    # train_loss = torch.nn.BCELoss()

                    pos_weight = torch.tensor([(len(labels_train) - sum(labels_train)) / sum(labels_train)]).to(device)
                    train_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss1 = train_loss(pre_score, true_label)

                    # loss_cl1 = losss
                    loss_cl = lossc + lossd
                    #loss_mp = cmp1_loss + dmp1_loss
                    # loss_inter = interc_loss + interd_loss
                    loss = loss1 + losss
                    # loss=loss1
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    # print(f"After batch {f + 1}: loss= {loss:.3f};", end='\n')  ###

                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score, batch_score)
                    epo_label = np.append(epo_label, label.numpy())
                end = time.time()
                print('Time：%.2f \n' % (end - start))
                train_losses.append(running_loss / len(trainLoader))


                valid_loss = 0
                valid_score, valid_label = [], []  ###
                model.eval()
                with torch.no_grad():
                    print("-----validing-----")
                    for f, item in enumerate(validLoader):
                        data, label = item
                        train_data = data.to(device)  ##torch.Size([32, 2])
                        pre_score, losss, lossc, lossd, mFea1, dFea1 = model(simData, train_data, adj_train=None)

                        valid_loss += torch.nn.BCEWithLogitsLoss()(pre_score, label.to(device)).item()
                        batch_score = pre_score.cpu().detach().numpy()

                        valid_score = np.append(valid_score, batch_score)
                        valid_label = np.append(valid_label, label.numpy())
                    end = time.time()
                    print('Time：%.2f \n' % (end - start))
                    valid_losses.append(valid_loss / len(validLoader))

                    metric, fpr, tpr = get_metrics(valid_score, valid_label)
                    valid_metric.append(metric)
                    # 保存每个 fold 训练完的模型参数
                    save_path = "result/savemodel/circR2_share_KG_fold_{}.pkl".format(a)
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), save_path)  ##


            all_valid_metrics.append(np.array(valid_metric))
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
            plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curve for Fold {a}')
            plt.legend()
            plt.savefig(f'result/image/loss_curve_circR2_share_KG_fold_{a}.png')

            # if state == 'valid':
            valid_metric = np.array(valid_metric)
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 0], label='AUC')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 1], label='AUPR')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 2], label='Accuracy')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 3], label='F1')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 4], label='Recall')
            plt.plot(range(1, len(valid_metric) + 1), valid_metric[:, 5], label='Precision')
            plt.xlabel('Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.title(f'Validation Metrics for Fold {a}')
            plt.legend()
            plt.savefig(f'result/image/validation_metrics_circR2_share_KG_fold_{a}.png')
            with open('result/circR2/valid_metrics_circrna.txt', 'a') as file:  ##改动

                file.write('AUC\tAUPR\tAccuracy\tF1\tRecall\tPrecision\n')


                for epoch in range(valid_metric.shape[0]):

                    line = '\t'.join(map(str, valid_metric[epoch])) + '\n'
                    file.write(line)
                print('数据已写入 valid_metrics.txt 文件中')

            mean_valid_metrics = np.mean(all_valid_metrics, axis=0)

        with open('result/circR2/mean_valid_metrics_circrna.txt', 'w') as file:  ##改动

            file.write('AUC\tAUPR\tAccuracy\tF1\tRecall\tPrecision\n')
            for epoch in range(mean_valid_metrics.shape[0]):
                line = '\t'.join(map(str, mean_valid_metrics[epoch])) + '\n'  # 修改7：确保均值结果写入 7 个指标
                file.write(line)
            print('五折均值已保存到 mean_valid_metrics_circR2v2.0.txt 文件中')

    else:
    # else:



        all_test_scores = []

        all_test_labels = []

        os.makedirs("curve_results", exist_ok=True)

        for fold in range(1, param.kfold + 1):

            print(f"\n===== Testing Fold {fold}/{param.kfold} =====")

            model = MASSCL(EmbeddingM(param), EmbeddingD(param), ACMF(param), GRID(param))

            model_path = f'result/savemodel/circR2_share_KG_fold_{fold}.pkl'

            model.load_state_dict(torch.load(model_path))

            model.to(device)

            model.eval()

            test_score, test_label = [], []

            testLoader = DataLoader.DataLoader(

                CVEdgeDataset(test_edges, test_labels),

                batch_size=param.batchSize, shuffle=False

            )

            with torch.no_grad():

                for data, label in testLoader:
                    data = data.to(device)

                    pre_score, _, _, _, _, _ = model(simData, data, adj_train=None)

                    test_score.extend(pre_score.cpu().numpy())

                    test_label.extend(label.numpy())



            fpr, tpr, _ = roc_curve(test_label, test_score)

            precision, recall, _ = precision_recall_curve(test_label, test_score)

            roc_auc = auc(fpr, tpr)

            pr_auc = auc(recall, precision)

            json.dump({

                "dataset": param.dataset_name,

                "fold": fold,

                "AUC": float(roc_auc),

                "AUPR": float(pr_auc),

                "fpr": fpr.tolist(),

                "tpr": tpr.tolist(),

                "precision": precision.tolist(),

                "recall": recall.tolist()

            }, open(f"curve_results/{param.dataset_name}_TEST_fold{fold}.json", "w"), indent=4)

            print(f"Fold {fold} → AUC: {roc_auc:.4f}, AUPR: {pr_auc:.4f}")



            all_test_scores.append(np.array(test_score))

            all_test_labels.append(np.array(test_label))


        all_test_scores = np.array(all_test_scores)  # shape: [folds, num_samples]

        mean_scores = np.mean(all_test_scores, axis=0)


        test_labels_final = all_test_labels[0]

        fpr, tpr, _ = roc_curve(test_labels_final, mean_scores)

        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(test_labels_final, mean_scores)

        pr_auc = auc(recall, precision)



        json.dump({

            "dataset": param.dataset_name,

            "avg_AUC": float(roc_auc),

            "avg_AUPR": float(pr_auc),

            "fpr": fpr.tolist(),

            "tpr": tpr.tolist(),

            "precision": precision.tolist(),

            "recall": recall.tolist()

        }, open(f"curve_results/{param.dataset_name}_TEST_average.json", "w"), indent=4)



    return kfolds
