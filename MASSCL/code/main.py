import torch
from datapro import Simdata_pro,loading_data
from train import train_test


class Config:
    def __init__(self):
        self.datapath = './datasets/CircR2Disease'
        self.kfold = 5
        self.ratio = 0.2
        self.batchSize = 64
        self.lr = 0.01
        self.weight_decay = 1e-5
        self.epoch =125
        self.dropout = 0.1
        self.gcn_layers = 1
        self.view = 2
        self.fm = 128
        self.nhid=64
        self.hidden_dim=64
        self.fd = 128
        self.out_channels = 128
        self.circRNA_number=585
        self.disease_number=88
        self.fcDropout = 0.5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

def main():
    param = Config()
    simData = Simdata_pro(param)
    train_data = loading_data(param)
    result= train_test(simData, train_data, param, state='valid')


if __name__ == "__main__":
    main()

