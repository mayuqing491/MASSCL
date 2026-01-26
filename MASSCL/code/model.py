import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch as th
from attention import CBAM,CrossPathSLAFusion
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F


class MASSCL(nn.Module):
    def __init__(self, c_emd1, d_emd1, acmf, grid):
        super(MASSCL, self).__init__()
        self.device = device
        self.Xc1 = c_emd1.to(self.device)
        self.Xd1 = d_emd1.to(self.device)
        self.acmf = acmf.to(self.device)
        self.md_supernode = grid.to(self.device)

    def forward(self, sim_data, train_data, adj_train= None):
        Em1, Em2 = self.Xc1(sim_data)
        Ed1, Ed2 = self.Xd1(sim_data)

        losss, lossc,lossd,cm1, cm2, dm1, dm2 = self.acmf(sim_data, Em1, Ed1, Em2, Ed2,adj_train)

        mFea1, dFea1 = pro_data(train_data, cm1, dm1)
        mFea2, dFea2 = pro_data(train_data, cm2, dm2)
        pre_asso = self.md_supernode(mFea1, mFea2, dFea1, dFea2)

        return pre_asso, losss,lossc,lossd, mFea1, dFea1


def pro_data(data, em, ed):
    edgeData = data.t().to(device)
    m_index = edgeData[0]
    d_index = edgeData[1]
    Em = torch.index_select(em, 0, m_index)
    Ed = torch.index_select(ed, 0, d_index)
    return Em, Ed


class EmbeddingM(nn.Module):
    def __init__(self, args):
        super(EmbeddingM, self).__init__()
        self.args = args
        self.device = device

        if self.args.dropout > 0:
            self.feat_drop = nn.Dropout(self.args.dropout)
        else:
            self.feat_drop = lambda x: x

        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_g = GCNConv(self.args.fm, self.args.fm)
        self.gcn_Icdc1 = GCNConv(self.args.fm, self.args.fm)
        self.gcn_Icdcdc1 = GCNConv(self.args.fm, self.args.fm)

        self.fc_list = nn.Linear(self.args.circRNA_number, self.args.fm, bias=True)
        nn.init.xavier_uniform_(self.fc_list.weight)
        if self.fc_list.bias is not None:
            self.fc_list.bias.data.fill_(0.0)

        self.fusion_modules = nn.ModuleList([CrossPathSLAFusion()])

        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.circRNA_number), (1, 1))
        self.cbamx = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)

        self.cnn_x = nn.Conv2d(
            in_channels=self.args.view * self.args.gcn_layers,
            out_channels=self.args.out_channels,
            kernel_size=(self.args.fm, 1),
            stride=1,
            bias=True
        )

        self.cnn_x = self.cnn_x.to(device)
        self.cbamx = self.cbamx.to(device)
        self.to(self.device)

    def forward(self, data):
        torch.manual_seed(1)
        circRNA_number = len(data['cc_f']['data_matrix'])

        x_c = torch.randn(circRNA_number, self.args.fm, device=self.device)

        x_c_f1 = torch.relu(self.gcn_x1_f(
            x_c,
            data['cc_f']['edges'].to(self.device),
            data['cc_f']['data_matrix'][data['cc_f']['edges'][0], data['cc_f']['edges'][1]].to(self.device)
        ))

        x_c_g1 = torch.relu(self.gcn_x1_g(
            x_c,
            data['cc_g']['edges'].to(self.device),
            data['cc_g']['data_matrix'][data['cc_g']['edges'][0], data['cc_g']['edges'][1]].to(self.device)
        ))

        feat = self.fc_list(data['cdc_I']['data_matrix'].to(self.device))
        cdc_mp1 = torch.relu(self.gcn_Icdc1(feat, data['cdc_I']['edges'].to(self.device)))

        feat = self.fc_list(data['cdcdc_I']['data_matrix'].to(self.device))
        cdcdc_mp1 = torch.relu(self.gcn_Icdcdc1(feat, data['cdcdc_I']['edges'].to(self.device)))


        fusion_model = CrossPathSLAFusion(hidden_dim=128)
        c_mp1 = fusion_model(cdc_mp1, cdcdc_mp1).to(self.device)
        c_mp1 = self.feat_drop(c_mp1)


        XM = torch.cat((x_c_f1, x_c_g1), 1).t()
        XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, circRNA_number)

        XM = self.cbamx(XM)
        XM = self.feat_drop(XM)

        x = self.cnn_x(XM)
        x = x.view(self.args.out_channels, circRNA_number).t()

        return x, c_mp1

class EmbeddingD(nn.Module):
    def __init__(self, args):
        super(EmbeddingD, self).__init__()
        self.args = args
        self.device = device

        if self.args.dropout > 0:
            self.feat_drop = nn.Dropout(self.args.dropout)
        else:
            self.feat_drop = lambda x: x

        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_Idcd1 = GCNConv(self.args.fd, self.args.fd)
        self.gcn_Idcdcd1 = GCNConv(self.args.fd, self.args.fd)

        self.fc_list = nn.Linear(args.disease_number, self.args.fd, bias=True)
        nn.init.xavier_uniform_(self.fc_list.weight)
        if self.fc_list.bias is not None:
            self.fc_list.bias.data.fill_(0.0)


        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))
        self.cbamy = CBAM(self.args.view * self.args.gcn_layers, 5, no_spatial=False)

        self.cnn_y = nn.Conv2d(
            in_channels=self.args.view * self.args.gcn_layers,
            out_channels=self.args.out_channels,
            kernel_size=(self.args.fd, 1),
            stride=1,
            bias=True
        )

        self.cnn_y = self.cnn_y.to(device)
        self.cbamy = self.cbamy.to(device)
        self.to(self.device)

    def forward(self, data):
        torch.manual_seed(1)
        disease_number = len(data['dd_s']['data_matrix'])


        x_d = torch.randn(disease_number, self.args.fd, device=self.device)


        y_d_s1 = torch.relu(self.gcn_y1_s(
            x_d, data['dd_s']['edges'].to(self.device),
            data['dd_s']['data_matrix'][data['dd_s']['edges'][0], data['dd_s']['edges'][1]].to(self.device)
        ))

        y_d_g1 = torch.relu(self.gcn_y1_g(
            x_d, data['dd_g']['edges'].to(self.device),
            data['dd_g']['data_matrix'][data['dd_g']['edges'][0], data['dd_g']['edges'][1]].to(self.device)
        ))


        feat = self.fc_list(data['dcd_I']['data_matrix'].to(self.device))
        dcd_mp1 = torch.relu(self.gcn_Idcd1(feat, data['dcd_I']['edges'].to(self.device)))

        feat = self.fc_list(data['dcdcd_I']['data_matrix'].to(self.device))
        dcdcd_mp1 = torch.relu(self.gcn_Idcdcd1(feat, data['dcdcd_I']['edges'].to(self.device)))


        fusion_model = CrossPathSLAFusion(hidden_dim=128)
        d_mp1 = fusion_model(dcd_mp1, dcdcd_mp1).to(self.device)
        d_mp1 = self.feat_drop(d_mp1)


        YD = torch.cat((y_d_s1, y_d_g1), 1).t()
        YD = YD.view(1, self.args.view * self.args.gcn_layers, self.args.fd, self.args.disease_number)

        YD = self.cbamy(YD)
        YD = self.feat_drop(YD)

        y = self.cnn_y(YD)
        y = y.view(self.args.out_channels, self.args.disease_number).t()

        return y, d_mp1


def loss_contrastive_m(m1, m2, return_embedding=False):

    m1_norm = F.normalize(m1, p=2, dim=1)
    m2_norm = F.normalize(m2, p=2, dim=1)

    pos = (m1_norm * m2_norm).sum(dim=1, keepdim=True)

    sim_m1 = torch.matmul(m1_norm, m1_norm.t())
    sim_m2 = torch.matmul(m2_norm, m2_norm.t())

    neg1 = sim_m1 - torch.diag_embed(torch.diagonal(sim_m1))
    neg2 = sim_m2 - torch.diag_embed(torch.diagonal(sim_m2))

    pos_m = pos.mean(dim=1)
    neg_m = torch.cat([neg1, neg2], dim=1).mean(dim=1)

    loss = F.softplus(neg_m - pos_m).mean()

    if return_embedding:
        m1_new = m1_norm + (m2_norm - m1_norm) * pos.detach()
        m2_new = m2_norm + (m1_norm - m2_norm) * pos.detach()

        m1_new = F.normalize(m1_new, p=2, dim=1)
        m2_new = F.normalize(m2_new, p=2, dim=1)

        return loss, m1_new, m2_new

    return loss


def contrastive_loss(
    c_emb,
    d_emb,
    adj=None,
    mode="semi", # "self", "supervised", "semi"
    temp=0.1,
    lam=0.5,
    label_smooth=0.1,
    noise_scale=0.01,
    top_k=20
):

    def safe_log(x, eps=1e-8):
        return torch.log(x + eps)

    Nc, _ = c_emb.shape
    Nd, _ = d_emb.shape


    c_norm = F.normalize(c_emb, dim=1)
    d_norm = F.normalize(d_emb, dim=1)


    c_view = F.normalize(c_emb + noise_scale * torch.randn_like(c_emb), dim=1)
    d_view = F.normalize(d_emb + noise_scale * torch.randn_like(d_emb), dim=1)


    logits = torch.matmul(c_norm, d_view.t()) / temp
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits)

    loss_self = torch.tensor(0.0, device=c_emb.device)

    if mode in ["self", "semi"]:
        min_dim = min(Nc, Nd)
        pos_idx = torch.arange(min_dim, device=c_emb.device)
        pos_mask = torch.zeros_like(logits)
        pos_mask[pos_idx, pos_idx] = 1.0

        # Hard negative mining
        mask_neg = 1 - pos_mask
        neg_logits = logits.clone()
        neg_logits[mask_neg == 0] = -1e9
        _, topk_idx = torch.topk(neg_logits, k=min(top_k, Nd), dim=1)

        num_c = (exp_logits * pos_mask).sum(dim=1)
        hard_neg_exp = torch.gather(exp_logits, 1, topk_idx)
        den_c = num_c + hard_neg_exp.sum(dim=1)
        loss_c = -safe_log(num_c / den_c).mean()

        neg_logits_T = logits.clone().t()
        mask_neg_T = mask_neg.t()
        neg_logits_T[mask_neg_T == 0] = -1e9
        _, topk_idx_T = torch.topk(neg_logits_T, k=min(top_k, Nc), dim=1)

        num_d = (exp_logits * pos_mask).sum(dim=0)
        hard_neg_exp_T = torch.gather(exp_logits.t(), 1, topk_idx_T)
        den_d = num_d + hard_neg_exp_T.sum(dim=1)
        loss_d = -safe_log(num_d / den_d).mean()

        loss_self = 0.5 * (loss_c + loss_d)

    loss_sup = torch.tensor(0.0, device=c_emb.device)

    if mode in ["supervised", "semi"] and adj is not None:
        pos_mask = adj.float()
        pos_mask = pos_mask * (1 - label_smooth) + label_smooth / max(1, pos_mask.sum())

        num_c = (exp_logits * pos_mask).sum(dim=1)
        den_c = exp_logits.sum(dim=1)
        loss_c = -safe_log(num_c / den_c).mean()

        num_d = (exp_logits * pos_mask).sum(dim=0)
        den_d = exp_logits.sum(dim=0)
        loss_d = -safe_log(num_d / den_d).mean()

        loss_sup = 0.5 * (loss_c + loss_d)


    if mode == "self":
        return loss_self

    elif mode == "supervised":
        return loss_sup

    elif mode == "semi":
        return lam * loss_sup + (1 - lam) * loss_self

    else:
        raise ValueError("mode must be 'self', 'supervised', or 'semi'")



class ACMF(nn.Module):
    def __init__(self, args, device="cuda"):
        super(ACMF, self).__init__()
        self.args = args
        self.device = device

    def forward(self, sim_data, c_emb1, d_emb1, c_emb2, d_emb2,adj):
        """
        sim_data: 相似性矩阵信息（保留接口）
        c_emb1, c_emb2: circRNA 两个视图的嵌入
        d_emb1, d_emb2: Disease 两个视图的嵌入
        """
        lossc, c_emb1, c_emb2 = loss_contrastive_m(c_emb1, c_emb2, return_embedding=True)
        lossd, d_emb1, d_emb2 = loss_contrastive_m(d_emb1, d_emb2, return_embedding=True)
        losss = contrastive_loss(c_emb1, d_emb1, adj, mode="semi")
        # lossm = contrastive_loss(c_emb2, d_emb2, adj, mode="semi")
        # lossc, lossd = 0, 0

        return losss,lossc,lossd,c_emb1, c_emb2, d_emb1, d_emb2

class GRID(nn.Module):
    def __init__(self, args):
        super(GRID, self).__init__()
        self.args = args
        self.device = args.device if hasattr(args, "device") else "cuda"

        hidden_dim = args.fm

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(args.fcDropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.gate_layer.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, em1, em2, ed1, ed2):

        R, D = em1, ed1
        x = (R * D).squeeze(dim=1)

        gate = torch.sigmoid(self.gate_layer(x))
        h = self.relu(self.fc1(x))
        h = h * gate

        h = h + x

        h = self.dropout(h)

        out = self.sigmoid(self.fc_out(h)).squeeze(dim=1)

        return out

