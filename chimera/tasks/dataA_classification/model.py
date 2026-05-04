# -*- coding:utf-8 -*-
import os.path as osp
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GraphSAGE,GCNConv,GATConv,GAT,GCN,GIN,TransformerConv, global_add_pool,global_mean_pool,global_max_pool,SAGPooling,JumpingKnowledge
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
import math
from torch_geometric.utils import to_dense_batch

from .CDAN_packages import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from .domain_adaptator import Discriminator
from .domain_adaptator import ReverseLayerF
from torch_geometric import seed_everything
seed_everything(432)
# 设置种子 
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from sklearn.metrics import precision_score, recall_score, f1_score

def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

def _compute_entropy_weights(logits):
    entropy = entropy_logits(logits)
    entropy = ReverseLayerF.apply(entropy, 1)
    entropy_w = 1.0 + torch.exp(-entropy)
    return entropy_w

def group_node_rep(node_rep, batch_size, num_part):
    group = []
    motif_group = []
    super_group = []
    # print('num_part', num_part)
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
#        pd.DataFrame([num_atom]).to_csv('num_atom.csv')
#        pd.DataFrame([num_motif]).to_csv('num_motif.csv')
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom].float().mean(dim=0))
        motif_group.append(node_rep[count + num_atom : count + num_all-1].float().mean(dim=0))
        super_group.append(node_rep[count + num_atom : count + num_all].float().mean(dim=0))
        count += num_all
    return group, motif_group, super_group
    
# One training epoch for GNN model.
def train(train_loader, model, optimizer,loss_fn):
    loss_cal=[]
    R2_cal=[]
    corr_cal=[]
    model.train()

    for data in train_loader:
       # data = data.to(device)
        optimizer.zero_grad()
        output_rg,rep,perm_x1,score_x1 = model(data)
        pred=output_rg.to(torch.float64)
        y=data[0].y.view(pred.shape).to(torch.float64)
        #R2 = torch.sum((pred - torch.mean(y))**2) / torch.sum((y - torch.mean(y))**2)
        R2 = 1-(torch.sum((y-pred )**2) / torch.sum((y - torch.mean(y))**2))
#        loss = torch.sqrt(loss_fn(pred, y)) + 10*F.nll_loss(output_cf, data[0].label)
        loss = torch.sqrt(loss_fn(pred, y))
        loss.backward()
        optimizer.step()
        
        pred = pred.detach().reshape(-1).numpy()
        y = y.detach().reshape(-1).numpy()
        R2 = 1-(np.sum((y-pred )**2) / np.sum((y - np.mean(y))**2))
        corr = np.corrcoef(y, pred)[0,1]
        loss = loss.detach().numpy()
        
        R2_cal.append(R2)
        corr_cal.append(corr)
        loss_cal.append(loss)
        l = len(loss_cal)
                         
    return sum(loss_cal)/l, sum(R2_cal)/l, sum(corr_cal)/l

def train_simada(cdan_loader,train_loader, model, domain_dmm, optimizer,opt_da):
    model.train()
    record=[]
    loss_cal=[]
    correct = 0
    for i,(data,data_t) in enumerate(cdan_loader):
        v_d, v_p, labels = data[0],data[1], data[0].y
        v_d_t, v_p_t = data_t[0], data_t[1]
        optimizer.zero_grad()
        opt_da.zero_grad()
        output,f,_,_,_,score= model(data)
#        n, model_loss = cross_entropy_logits(score, labels)
#        print(output)
#        print(output.shape)
#        print(labels)
#        print(labels.shape)
        model_loss = F.nll_loss(output, labels)
        _,f_t,_,_,_,score_t  = model(data_t)
    
        reverse_f = ReverseLayerF.apply(f, 1)
        softmax_output = torch.nn.Softmax(dim=1)(score)
        softmax_output = softmax_output.detach()
        # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
    
        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
#        print(feature.shape)
        adv_output_src_score = domain_dmm(feature)
    
        reverse_f_t = ReverseLayerF.apply(f_t, 1)
        softmax_output_t = torch.nn.Softmax(dim=1)(score_t)
        softmax_output_t = softmax_output_t.detach()
        # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
    
        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
    
        adv_output_tgt_score = domain_dmm(feature_t)
        
        entropy_src = _compute_entropy_weights(score)
        entropy_tgt = _compute_entropy_weights(score_t)
        src_weight = entropy_src / torch.sum(entropy_src)
        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)    
    
        n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(128),
                                                    src_weight)
        n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(128),
                                                    tgt_weight)
        h=f.detach()
        h_t=f_t.detach()
                                                  
        
        da_loss = loss_cdan_src + loss_cdan_tgt 

        loss = model_loss+ 1*da_loss
        
        loss.backward()
        optimizer.step()
        opt_da.step()
        
        pred = output.max(dim=1)[1]
        loss = F.nll_loss(output, labels)
        loss_cal.append(loss)
        correct += pred.eq(labels).sum().item()
    loss_sum=sum(loss_cal)
    
    return loss_sum/len(loss_cal) ,correct / len(train_loader.dataset)


# Get acc. of GNN model.
def test(loader, model):
    model.eval()
    loss_cal=[]
    correct = 0
    pred_list = []
    true_list = []
    for data in loader:
        #data = data.to(device)
        output,h,_,perm_x1,score_x1,_ = model(data)
        pred = output.max(dim=1)[1]
        loss = F.nll_loss(output, data[0].y)
        loss_cal.append(loss)
        correct += pred.eq(data[0].y).sum().item()
        
        pred_list.extend(pred.tolist())
        true_list.extend(data[0].y.tolist())
    loss_sum=sum(loss_cal)
    accuracy = sum(pred_list[i] == true_list[i] for i in range(len(pred_list))) / len(pred_list)
    precision = precision_score(true_list, pred_list, average='macro')
    recall = recall_score(true_list, pred_list, average='macro')
    f1 = f1_score(true_list, pred_list, average='macro')
    return loss_sum/len(loss_cal),correct / len(loader.dataset),accuracy, precision, recall, f1



def auc_cal(loader,model):
    y_prediction=[]
    y_true=[]
    y_score=[]
    for data in loader:
        output,h,_,perm_x1,score_x1,_ = model(data)
        pred = output.max(dim=1)[1]
        ture= data[0].y
        y_prediction+=pred
        y_true+=ture
        output=torch.exp(output)
        output_list=output[:,1].detach().numpy().tolist()
        y_score+=output_list
    # 计算
    y_test=y_true
    y_score=y_score
    fpr, tpr, thread = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc,fpr,tpr

def auc_plot(roc_curves,mean_fpr_test,mean_tpr_test,tprs_lower_test,tprs_upper_test,mean_auc,std_auc):
    # 绘制ROC曲线
    plt.figure(figsize=(8, 8))
    for i, (fpr, tpr, roc_auc) in enumerate(roc_curves):
        plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')
    
    # 绘制平均ROC曲线
    plt.plot(mean_fpr_test, mean_tpr_test, color='b', label=f'GNN (AUC = {mean_auc:.2f}±{std_auc:.2f})')
    
    # 绘制标准差的阴影区域
    plt.fill_between(
        mean_fpr_test,
        tprs_lower_test,
        tprs_upper_test,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for GNN ')
    plt.legend(loc='lower right')
    

class SubExtractor(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()
        
        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)
        
        self.W_Q = Linear(hidden_dim, hidden_dim)
        self.W_K = Linear(hidden_dim, hidden_dim)
        self.W_V = Linear(hidden_dim, hidden_dim)
        self.W_O = Linear(hidden_dim, hidden_dim)
#
#        # 可学习的chirality权重因子 
#        self.chirality_weight_factor = nn.Parameter(torch.Tensor(1)) 
#        # 初始化为标量 
#        nn.init.constant_(self.chirality_weight_factor, 2.0) # 默认初始化为2.0
        self.residual = residual
    
    def forward(self, x, batch, chirality_feas, use_chiral= False):
        # 处理输入的chirality_features
        chirality_features = chirality_feas.view(list(chirality_feas.shape)[0]//100,100)  # (batch_size, num_atoms)
#        pd.DataFrame([chirality_features.shape]).to_csv('size_chirality_features.csv')
#        pd.DataFrame([x.shape]).to_csv('size_x.csv')
#        # 将chirality_features添加到输入特征中
#        x = torch.cat((x, chirality_features), dim=-1)  # (batch_size, num_atoms, hidden_dim + 1)
        
#        # 使用ReLU激活函数保证chirality_weight_factor大于1
#        chirality_weight_factor = torch.relu(self.chirality_weight_factor - 1) + 1
        
        K = self.W_K(x)
        V = self.W_V(x)
        
        K, mask = to_dense_batch(K, batch)
        # mask: (batch_size, max_num_nodes)
        V, _ = to_dense_batch(V, batch)
        
        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)
        
        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)
        
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
#        pd.DataFrame([A.shape]).to_csv('size_A.csv')
        # 增强手性中心的注意力
        chirality_weight = chirality_features.unsqueeze(1).repeat(1, 10, 1)[:,:,:A.size(2)]  # 计算手性特征权重 (batch_size,num_patterns, num_atoms)
         
        if use_chiral ==True:
        
            alpha = float(os.getenv("CHIMERA_CHIRALITY_ALPHA", "2.0"))
            A = A * (chirality_weight * alpha + (1 - chirality_weight))  # 增加手性特征的权重（根据需要调整此系数）/ # 使用learnable factor
#        pd.DataFrame(A[0].detach().numpy()).to_csv('A.csv')
#        pd.DataFrame(A.numpy()).to_csv('A.csv')
        A = A.softmax(dim=-2)
        # (batch_size, num_clusters, max_num_nodes)
        
        out = Q + A @ V
        
        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        
        return out, A.detach().argmax(dim=-2), mask




class Transformer(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, num_patterns=10, cfg=None):
        super(Transformer, self).__init__()

        self.ablation = dict((cfg or {}).get("ablation", {}))
        self.ablation.setdefault("no_fingerprint", False)
        self.ablation.setdefault("no_graph", False)
        self.ablation.setdefault("no_motif", False)
        self.ablation.setdefault("no_interaction", False)
        self.ablation.setdefault("no_loffi", False)
        self.ablation.setdefault("no_condition", False)

        self.conv1 = GIN(dataset[0][0].num_features,hidden,5,2*hidden,jk='cat')
        self.conv2 = GAT(dataset[0][5].num_features,hidden,2,2*hidden,jk='cat')
#        self.conv2 = TransformerConv(dataset[0][0].num_features,hidden,1,edge_dim=dataset[0][0].num_edge_features)
#        self.conv3 = TransformerConv(dataset[0][0].num_features,hidden,1,edge_dim=dataset[0][0].num_edge_features)
#        self.sage =  GraphSAGE(dataset[0][0].num_features, hidden, 2, 2*hidden)
#        self.convs1 = torch.nn.ModuleList()
#        self.convs2 = torch.nn.ModuleList()
#        self.convs3 = torch.nn.ModuleList()
#        self.convs4 = torch.nn.ModuleList()
#        for i in range(num_layers - 1):
#
#            self.convs1.append(TransformerConv(1*hidden, hidden, 1, edge_dim=dataset[0][0].num_edge_features))
#            self.convs2.append(TransformerConv(1*hidden, hidden, 1, edge_dim=dataset[0][0].num_edge_features))
#            self.convs3.append(TransformerConv(1*hidden, hidden, 1, edge_dim=dataset[0][0].num_edge_features))

        #self.bn3 = BN(6*hidden+20+1) 
        self.bn1 = BN(2048+16*hidden+6*num_patterns*num_patterns)
#        self.bn2 = BN(6*hidden+1*num_patterns*num_patterns+76+3+1024)
        #self.bn3 = BN(16*hidden)
        
        
        self.lin1 = Linear(2048+16*hidden+6*num_patterns*num_patterns, (2048)//2+16*hidden//2+6*num_patterns*num_patterns//2)
        self.lin2 = Linear((2048)//2+16*hidden//2+6*num_patterns*num_patterns//2, (2048)//4+16*hidden//4+6*num_patterns*num_patterns//4)
        self.lin3 = Linear((2048)//4+16*hidden//4+6*num_patterns*num_patterns//4, 2)
        
#        self.lin4 = Linear(2048, 2048//2)
#        self.lin5 = Linear(2048//2, 2048//4)
#        self.lin6 = Linear(2048//4, 2048//8)
#        self.lin7 = Linear(2048, 2048//8)
        
        self.lin8 = Linear(2048+16*hidden+6*num_patterns*num_patterns, 2048+16*hidden+6*num_patterns*num_patterns)
#        self.lin8 = Linear(16*hidden+6*num_patterns*num_patterns+7, 16*hidden+6*num_patterns*num_patterns+7)
#        self.lin1 = Linear(16*hidden, 8*hidden)
#        self.lin2 = Linear(8*hidden, 4*hidden)
#        self.lin3 = Linear(4*hidden, 1)
        
#        self.embedding1 = torch.nn.Embedding(10, min(10, 10//2))
#        self.embedding2 = torch.nn.Embedding(61, min(10, 61//2))
#        self.embedding3 = torch.nn.Embedding(18, min(10, 18//2))
#        self.embedding4 = torch.nn.Embedding(14, min(10, 14//2))
#        self.embedding1 = torch.nn.Embedding(10, min(10, 10//2))



        self.sagpool1= SAGPooling(2*hidden,0.8,GCNConv )
#        self.sagpool2 = SAGPooling(1*hidden,0.8,GATConv )
#        self.sagpool3= SAGPooling(1*hidden,0.8,GATConv )
#        self.sagpool4 = SAGPooling(1*hidden,0.8,GATConv )
        
        self.pool1 = SubExtractor(2*hidden, num_patterns, False)
        self.pool2 = SubExtractor(2*hidden, num_patterns, False)
#        self.pool3 = SubExtractor(2*hidden, num_patterns, False)
#        self.pool4 = SubExtractor(2*hidden, num_patterns, False)
#        self.jump = JumpingKnowledge(mode='cat')


    def _apply_ablation(
        self,
        fg_input,
        super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
        sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
        x, x1, x2, x3,
        loffi_input=None,
        cond_embedding=None,
    ):
        """Apply YAML-controlled ablations by zeroing selected feature blocks.

        This keeps the concatenated representation dimensionality unchanged, so
        BN/Linear layers remain compatible with the original CHIMERA topology.
        """
        abl = self.ablation
        if abl.get("no_fingerprint", False):
            fg_input = fg_input * 0.0

        if abl.get("no_interaction", False):
            sim_cp = sim_cp * 0.0
            sim_c1 = sim_c1 * 0.0
            sim_c2 = sim_c2 * 0.0
            sim_x12h = sim_x12h * 0.0
            sim_x13h = sim_x13h * 0.0
            sim_x14h = sim_x14h * 0.0

        if abl.get("no_motif", False):
            super_rep_x0 = super_rep_x0 * 0.0
            super_rep_x10 = super_rep_x10 * 0.0
            super_rep_x20 = super_rep_x20 * 0.0
            super_rep_x30 = super_rep_x30 * 0.0
            sim_x12h = sim_x12h * 0.0
            sim_x13h = sim_x13h * 0.0
            sim_x14h = sim_x14h * 0.0

        if abl.get("no_graph", False):
            x = x * 0.0
            x1 = x1 * 0.0
            x2 = x2 * 0.0
            x3 = x3 * 0.0
            sim_cp = sim_cp * 0.0
            sim_c1 = sim_c1 * 0.0
            sim_c2 = sim_c2 * 0.0

        if loffi_input is not None and abl.get("no_loffi", False):
            loffi_input = loffi_input * 0.0

        if cond_embedding is not None and abl.get("no_condition", False):
            cond_embedding = cond_embedding * 0.0

        return (
            fg_input,
            super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
            sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
            x, x1, x2, x3,
            loffi_input,
            cond_embedding,
        )

    def reset_parameters(self):
        
#        torch.nn.init.xavier_uniform_(self.embedding1.weight.data)
#        torch.nn.init.xavier_uniform_(self.embedding2.weight.data)
#        torch.nn.init.xavier_uniform_(self.embedding3.weight.data)
#        torch.nn.init.xavier_uniform_(self.embedding4.weight.data)
#        torch.nn.init.xavier_uniform_(self.embedding1.weight.data)
#        
#        self.sage.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
#        self.conv3.reset_parameters()
#        self.conv4.reset_parameters()
#        for conv in self.convs1:
#            conv.reset_parameters()
#        for conv in self.convs2:
#            conv.reset_parameters()
#        for conv in self.convs3:
#            conv.reset_parameters()
#        for conv in self.convs4:
#            conv.reset_parameters()
            
        self.bn1.reset_parameters()
#        self.bn2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        
#        self.lin4.reset_parameters()
#        self.lin5.reset_parameters()
#        self.lin6.reset_parameters()
#        self.lin7.reset_parameters()
        self.lin8.reset_parameters()
        self.sagpool1.reset_parameters()
#        self.sagpool2.reset_parameters()
#        self.sagpool3.reset_parameters()
#        self.sagpool4.reset_parameters()
#        self.jump.reset_parameters()
        
        
    

        
#temp_x,time_x,metal_x,solvent_x,additive_x,gm_x,elsi_x,
#, data[0].temp ,data[0].time ,data[0].metal ,data[0].solvent ,data[0].additive ,data[0].gm ,data[0].elsi 
    def forward(self, data):
        x, edge_index, edge_attr, batch,temp_x,time_x,metal_x,solvent_x,additive_x,gm_x,elsi_x, add,add1,chirality_fea = data[0].x, data[0].edge_index, data[0].edge_attr, data[0].batch, data[0].temp ,data[0].time ,data[0].metal ,data[0].solvent ,data[0].additive ,data[0].gm ,data[0].elsi ,data[0].add_fea ,data[0].add_fea1, data[0].chirality_fea
        x1, edge_index_x1, edge_attr_x1, batch_x1, chirality_fea1 = data[1].x, data[1].edge_index, data[1].edge_attr, data[1].batch, data[1].chirality_fea
        x2, edge_index_x2, edge_attr_x2, batch_x2, chirality_fea2 = data[2].x, data[2].edge_index, data[2].edge_attr, data[2].batch, data[2].chirality_fea
        x3, edge_index_x3, edge_attr_x3, batch_x3, chirality_fea3 = data[3].x, data[3].edge_index, data[3].edge_attr, data[3].batch, data[3].chirality_fea
        
        xh1, edge_index_xh1, edge_attr_xh1, batch_xh1, num_part_xh1 = data[4].x, data[4].edge_index, data[4].edge_attr, data[4].batch, data[4].num_part
        xh2, edge_index_xh2, edge_attr_xh2, batch_xh2, num_part_xh2 = data[5].x, data[5].edge_index, data[5].edge_attr, data[5].batch, data[5].num_part
        xh3, edge_index_xh3, edge_attr_xh3, batch_xh3, num_part_xh3 = data[6].x, data[6].edge_index, data[6].edge_attr, data[6].batch, data[6].num_part
        xh4, edge_index_xh4, edge_attr_xh4, batch_xh4, num_part_xh4 = data[7].x, data[7].edge_index, data[7].edge_attr, data[7].batch, data[7].num_part
        
        
#        xs = self.sage(x, edge_index, edge_attr=edge_attr)
#        x1s = self.sage(x1, edge_index_x1, edge_attr=edge_attr_x1)
#        x2s = self.sage(x2, edge_index_x2, edge_attr=edge_attr_x2)
#        x3s = self.sage(x3, edge_index_x3, edge_attr=edge_attr_x3)
  
        x0 = self.conv1(x, edge_index, edge_attr=edge_attr)
        x10 = self.conv1(x1, edge_index_x1, edge_attr=edge_attr_x1)
        x20 = self.conv1(x2, edge_index_x2, edge_attr=edge_attr_x2)
        x30 = self.conv1(x3, edge_index_x3, edge_attr=edge_attr_x3)
        
        xh1 = self.conv2(xh1, edge_index_xh1, edge_attr=edge_attr_xh1)
        xh2 = self.conv2(xh2, edge_index_xh2, edge_attr=edge_attr_xh2)
        xh3 = self.conv2(xh3, edge_index_xh3, edge_attr=edge_attr_xh3)
        xh4 = self.conv2(xh4, edge_index_xh4, edge_attr=edge_attr_xh4)
#        pd.DataFrame([x.shape]).to_csv('size_x.csv')
        bs_xh1 = int(batch_xh1.max().item()) + 1 if batch_xh1.numel() > 0 else 0
        bs_xh2 = int(batch_xh2.max().item()) + 1 if batch_xh2.numel() > 0 else 0
        bs_xh3 = int(batch_xh3.max().item()) + 1 if batch_xh3.numel() > 0 else 0
        bs_xh4 = int(batch_xh4.max().item()) + 1 if batch_xh4.numel() > 0 else 0
        node_rep_x0, motif_node_rep_x0, super_node_rep_x0 = group_node_rep(xh1, bs_xh1, num_part_xh1)
        node_rep_x10, motif_node_rep_x10, super_node_rep_x10 = group_node_rep(xh2, bs_xh2, num_part_xh2)
        node_rep_x20, motif_node_rep_x20, super_node_rep_x20 = group_node_rep(xh3, bs_xh3, num_part_xh3)
        node_rep_x30, motif_node_rep_x30, super_node_rep_x30 = group_node_rep(xh4, bs_xh4, num_part_xh4)
        
        
        motif_rep_x0 = torch.stack(motif_node_rep_x0, dim=0)
        motif_rep_x10 = torch.stack(motif_node_rep_x10, dim=0)
        motif_rep_x20 = torch.stack(motif_node_rep_x20, dim=0)
        motif_rep_x30 = torch.stack(motif_node_rep_x30, dim=0)
        
        super_rep_x0 = torch.stack(super_node_rep_x0, dim=0)
        super_rep_x10 = torch.stack(super_node_rep_x10, dim=0)
        super_rep_x20 = torch.stack(super_node_rep_x20, dim=0)
        super_rep_x30 = torch.stack(super_node_rep_x30, dim=0)
#        xj = [x]
#        xj1 = [x1]
#        xj2 = [x2]
#        xj3 = [x3]
        
#        for conv in self.convs1:
#            x = conv(x, edge_index, edge_attr=edge_attr)
#            x1 = conv(x1, edge_index_x1, edge_attr=edge_attr_x1)
#            x2 = conv(x2, edge_index_x2, edge_attr=edge_attr_x2)
#            x3 = conv(x3, edge_index_x3, edge_attr=edge_attr_x3)
#            xj += [x]
#            xj1 += [x1]
#            xj2 += [x2]
#            xj3 += [x3]
        
#        for conv in self.convs2:
#            x1 = conv(x1, edge_index_x1, edge_attr=edge_attr_x1)    
#        for conv in self.convs1:
#            x2 = conv(x2, edge_index_x2, edge_attr=edge_attr_x2)
#            x3 = conv(x3, edge_index_x3, edge_attr=edge_attr_x3)
#        for conv in self.convs1:
#            x3 = conv(x3, edge_index_x3, edge_attr_x3)
#        x = self.jump(xj)
#        x1 = self.jump(xj1)
#        x2 = self.jump(xj2)
#        x3 = self.jump(xj3)

        x, pool_edge_index, pool_edge_weight, pool_batch_x, perm_x ,score_perm_x= self.sagpool1(x0,edge_index,batch=batch)
        x1, pool_edge_index_x1, pool_edge_weight, pool_batch_x1, perm_x1 ,score_perm_x1= self.sagpool1(x10,edge_index_x1,batch=batch_x1)
        x2, pool_edge_index_x2, pool_edge_weight, pool_batch_x2, perm_x2 ,score_perm_x2= self.sagpool1(x20,edge_index_x2,batch=batch_x2)
        x3, pool_edge_index_x3, pool_edge_weight, pool_batch_x3, perm_x3 ,score_perm_x3= self.sagpool1(x30,edge_index_x3,batch=batch_x3)

        
        pool_x, *_ = self.pool1(x0, batch, chirality_fea, use_chiral= True)
        pool_x = F.normalize(pool_x, dim=-1)
        pool_x1, *_ = self.pool1(x10, batch_x1, chirality_fea1, use_chiral= True)
        pool_x1 = F.normalize(pool_x1, dim=-1)
        
        pool_x2, *_ = self.pool1(x20, batch_x2, chirality_fea2, use_chiral= True)
        pool_x2 = F.normalize(pool_x2, dim=-1)
        pool_x3, *_ = self.pool1(x30, batch_x3, chirality_fea3, use_chiral= True)
        pool_x3 = F.normalize(pool_x3, dim=-1)
        
        pool_x1h, *_ = self.pool2(xh1, batch_xh1, chirality_fea, use_chiral= False)
        pool_x1h = F.normalize(pool_x1h, dim=-1)
        pool_x2h, *_ = self.pool2(xh2, batch_xh2, chirality_fea1, use_chiral= False)
        pool_x2h = F.normalize(pool_x2h, dim=-1)
        
        pool_x3h, *_ = self.pool2(xh3, batch_xh3, chirality_fea2, use_chiral= False)
        pool_x3h = F.normalize(pool_x3h, dim=-1)
        pool_x4h, *_ = self.pool2(xh4, batch_xh4, chirality_fea3, use_chiral= False)
        pool_x4h = F.normalize(pool_x4h, dim=-1)
        
#        x = torch.cat((global_max_pool(x, pool_batch_x),global_mean_pool(x, pool_batch_x)),1) 
#        x1 = torch.cat((global_max_pool(x1, pool_batch_x1),global_mean_pool(x1, pool_batch_x1)),1) 
#        x2 = torch.cat((global_max_pool(x2, pool_batch_x2),global_mean_pool(x2, pool_batch_x2)),1) 
#        x3 = torch.cat((global_max_pool(x3, pool_batch_x3),global_mean_pool(x3, pool_batch_x3)),1) 
        x = global_mean_pool(x, pool_batch_x)
        x1 = global_mean_pool(x1, pool_batch_x1)
        x2 = global_mean_pool(x2, pool_batch_x2)
        x3 = global_mean_pool(x3, pool_batch_x3)
        
#        xs = global_mean_pool(xs, batch)
#        x1s = global_mean_pool(x1s, batch_x1) 
#        x2s = global_mean_pool(x2s, batch_x2) 
#        x3s = global_mean_pool(x3s, batch_x3)

#        solvent_embedding = self.embedding1(solvent_x)
        
        sim_cp = pool_x @ pool_x1.transpose(-1, -2)
        sim_cp = sim_cp.flatten(1)
        
#        sim_R12 = pool_x2 @ pool_x3.transpose(-1, -2)
#        sim_R12 = sim_R12.flatten(1)
#        
        sim_c1 = pool_x @ pool_x2.transpose(-1, -2)
        sim_c1 = sim_c1.flatten(1)
        
        sim_c2 = pool_x @ pool_x3.transpose(-1, -2)
        sim_c2 = sim_c2.flatten(1)
        
        
        
        sim_x12h = pool_x1h @ pool_x2h.transpose(-1, -2)
        sim_x12h = sim_x12h.flatten(1)
        
#        sim_R12 = pool_x2 @ pool_x3.transpose(-1, -2)
#        sim_R12 = sim_R12.flatten(1)
#        
        sim_x13h = pool_x1h @ pool_x3h.transpose(-1, -2)
        sim_x13h = sim_x13h.flatten(1)
        
        sim_x14h = pool_x1h @ pool_x4h.transpose(-1, -2)
        sim_x14h = sim_x14h.flatten(1)
        
#        interaction=torch.mul(x,x1)
#        pd.DataFrame([add.shape]).to_csv('size_add_check.csv')
#        interaction1=torch.mul(x,x2)
#        interaction2=torch.mul(x,x3)
#        interaction3=torch.mul(x2,x3)
        #z= torch.cat((x, x1,x2,x3,interaction,interaction1,interaction2,solvent_embedding,temp_x.view(len(temp_x),1)), 1)
       
#        metal_embedding = self.embedding1(metal_x)
#        solvent_embedding = self.embedding2(solvent_x)
#        additive_embedding = self.embedding3(additive_x)
#        gm_embedding = self.embedding4(gm_x)
#        elsi_embedding = self.embedding1(elsi_x)
        
#        z= torch.cat((sim_cp,x,x1,interaction,metal_embedding,solvent_embedding,additive_embedding,gm_embedding,elsi_embedding,time_x.view(len(time_x),1),temp_x.view(len(temp_x),1),add), 1)
        temp_x=temp_x.view(len(temp_x),1)
        time_x=time_x.view(len(time_x),1)
        metal_x=metal_x.view(len(metal_x),1)
        solvent_x=solvent_x.view(len(solvent_x),1)
        additive_x=additive_x.view(len(additive_x),1)
        gm_x=gm_x.view(len(gm_x),1)
        elsi_x=elsi_x.view(len(elsi_x),1)
        
        fg_input=add.view(list(add.shape)[0]//2048,2048)
        (
            fg_input,
            super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
            sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
            x, x1, x2, x3,
            _, _,
        ) = self._apply_ablation(
            fg_input,
            super_rep_x0, super_rep_x10, super_rep_x20, super_rep_x30,
            sim_cp, sim_c1, sim_c2, sim_x12h, sim_x13h, sim_x14h,
            x, x1, x2, x3,
        )
#        pd.DataFrame([interaction.shape]).to_csv('size_interaction_check.csv')
#        pd.DataFrame([add.shape]).to_csv('size_add_check.csv')
#        pd.DataFrame([fg_input.shape]).to_csv('size_fg_input_check.csv')
#        fg_rep= F.relu(self.lin7(fg_input))
#        fg_rep= F.relu(self.lin4(fg_input))
#        fg_rep= F.relu(self.lin5(fg_rep))
#        fg_rep= F.relu(self.lin6(fg_rep))
#        mole_embedding=torch.cat(( super_rep_x0, super_rep_x10,super_rep_x20, super_rep_x30,sim_cp,sim_c1,sim_c2,x,x1,x2,x3,xs,x1s,x2s,x3s,interaction), 1)
        mole_embedding=torch.cat((fg_input, super_rep_x0, super_rep_x10,super_rep_x20, super_rep_x30,sim_cp,sim_c1,sim_c2,sim_x12h,sim_x13h,sim_x14h,x,x1,x2,x3), 1)
#        mole_embedding=torch.cat((super_rep_x0, super_rep_x10,super_rep_x20, super_rep_x30,sim_cp,sim_c1,sim_c2,sim_x12h,sim_x13h,sim_x14h,x,x1,x2,x3), 1)
        # Data A follows the original Data-A CHIMERA code: no condition vector or Ir-Ni 328-D CSV block is concatenated.
        z = mole_embedding
#        pd.DataFrame([z.shape]).to_csv('size_z_check.csv')
        z_attn= F.log_softmax(self.lin8(z), dim=1)
        z = torch.mul(z_attn, z) 
        z1=self.bn1(z)
#        pd.DataFrame([z1.shape]).to_csv('size_z1_check.csv')
        z1 = F.relu(self.lin1(z1))
        
        # z1 = F.dropout(z1, p=0.3, training=self.training)
        z2 = F.relu(self.lin2(z1))
#        z2 = torch.cat((z2,fg_input,loffi_input),1)
        output = self.lin3(z2)
#        pd.DataFrame([output_cf.shape,z.shape]).to_csv('size_check.csv')
#        pd.DataFrame([output_cf[0].detach().numpy()]).to_csv('size_check.csv')



#        return output_rg, F.log_softmax(output_cf, dim=-1) , perm_x1, score_perm_x1
        return F.log_softmax(output, dim=-1), z,z2 , perm_x, score_perm_x, output

    def __repr__(self):
        return self.__class__.__name__

# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, dataset, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr = 0.000001, factor=0.1, patience=10,
                       num_repetitions=10, all_std=True):

    test_accuracies_all = []
    test_accuracies_complete = []
    
    test_auc_all = []
    test_auc_complete = []
    
    test_acc_complete = []
    test_precision_complete = []
    test_recall_complete = []
    test_f1_complete = []
    
    test_acc_all = []
    test_precision_all = []
    test_recall_all = []
    test_f1_all = []
    
    roc_curves = []
    all_fpr_test = []
    all_tpr_test = []
    all_aucs = []
    #record=[]
#    seed_list=[6,54,432]
    seed_list=[432]
    for i in range(num_repetitions):
        seed_everything(seed_list[i])
        # Test acc. over all folds.
        test_accuracies = []
        
        test_auces=[]
        
        test_acces=[]
        test_precisiones=[]
        test_recalles=[]
        test_f1es=[]
        kf = KFold(n_splits=10, shuffle=True,random_state=5)
#        random.shuffle(dataset)
        num=1
        fold=1
        embeding=[]
        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
#            train_index, val_index = train_test_split(train_index, test_size=0.2,random_state=1)
            best_test = 0.0
            best_test_auc = 0.0
            best_test_acc = 0.0
            best_test_precision = 0.0
            best_test_recall = 0.0
            best_test_f1 = 0.0
            embeding_train=[]
            embeding_test=[]
            # Split data.
            train_dataset = [dataset[i] for i in train_index.tolist()]
#            val_dataset = [dataset[i] for i in val_index.tolist()]
            test_dataset = [dataset[i] for i in test_index.tolist()]
            

            #             当前test_dataset的大小 
            current_size = len(test_dataset) 
            #             计算需要补充的样本数量，使得总大小成为32的倍数 
            remainder = current_size % 128 
            if remainder != 0: 
                #             计算需要补充的样本数量 
                num_to_add = 128 - remainder 
                #             复制test_dataset中的一部分数据，直到总数满足要求 
                test_dataset += test_dataset[:num_to_add]

            #             当前train_dataset的大小 
            current_size = len(train_dataset) 
            #             计算需要补充的样本数量，使得总大小成为32的倍数 
            remainder = current_size % 128
            if remainder != 0: 
                #             计算需要补充的样本数量 
                num_to_add = 128 - remainder 
                #             复制test_dataset中的一部分数据，直到总数满足要求 
                train_dataset += train_dataset[:num_to_add]
            
            pp=math.ceil(len(train_dataset)/len(test_dataset))
            data_CDAN=list(zip(train_dataset,test_dataset*(pp+10)))
            
            # Prepare batching.
            cdan_loader = DataLoader(data_CDAN, batch_size=batch_size,shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
            train_loader1 = DataLoader(train_dataset, batch_size=batch_size)
#            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                for h in hidden:
                    # Setup model.
                    model = gnn(train_dataset, l, h)
                    model.reset_parameters()
                    loss_fn = torch.nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay = 1e-4 )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=factor, patience=patience,min_lr=0.0000001)
                    domain_dmm = Discriminator(input_size=2*(2048+16*h+6*10*10),output_size=2*(2048+16*h+6*10*10), n_class=2)
                    domain_dmm.reset_parameters()
                    opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=0.001)
#                    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: (0.01 if epoch < 100 else 0.001),last_epoch=-1)
                    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_da, mode='min',factor=factor, patience=patience,min_lr=0.0000001)
                    for epoch in range(1, max_num_epochs + 1):
                        lr = scheduler.optimizer.param_groups[0]['lr']
#                        loss_train,R2_train,corr_train=train(train_loader, model, optimizer,loss_fn)
                        #loss_train,R2_train,corr_train=train_simada(cdan_loader,train_loader, model, optimizer,loss_fn)
                        loss_train,acc_train = train_simada(cdan_loader,train_loader, model, domain_dmm, optimizer,opt_da)
                        loss_test,test_acc,accuracy, precision, recall, f1 = test(test_loader, model)
                        test_auc,_,_ = auc_cal(test_loader,model)
                        scheduler.step(loss_test)
                        scheduler2.step(loss_test)
                        if test_auc > best_test_auc:
                            best_test_auc,fpr,tpr = auc_cal(test_loader,model)
                            
                            best_test=test_acc
                            best_test_acc=accuracy
                            best_test_precision=precision
                            best_test_recall=recall
                            best_test_f1=f1
                        
                            
                            for data in train_loader1:
                                _,_,h,_,_,_=model(data)
#                                pd.DataFrame(h.shape).to_csv("size_h.csv")
                                embeding_train.append(h.detach().numpy())
#                            pd.DataFrame([len(embeding_train)]).to_csv("len_embeding_train.csv") 
#                            pd.DataFrame(np.vstack(embeding_train)).to_csv("embeding_train_value.csv")    
                            
                            for data in test_loader:
                                _,_,h1,_,_,_=model(data)
#                                pd.DataFrame(h1.shape).to_csv("size_h1.csv")
                                embeding_test.append(h1.detach().numpy())
#                            pd.DataFrame([len(embeding_test)]).to_csv("len_embeding_test.csv") 
#                            pd.DataFrame(np.vstack(embeding_test)).to_csv("embeding_test_value.csv") 
#                            embeding_tr=[]
#                            for data in embeding_train:
#                                embeding_tr.append(data[0])
#                            embeding_tr=np.vstack(embeding_tr)
#                            embeding_te=[]
#                            for data in embeding_test:
#                                embeding_te.append(data[0])
#                            embeding_te=np.vstack(embeding_te)
                            
                            #record.append([pred,y])
                            #pd.DataFrame(record).to_csv("pred_check.csv")
                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr:
                            break
            pd.DataFrame(np.vstack(embeding_train)).to_csv("embedding_train"+str(fold)+".csv")  
            pd.DataFrame(np.vstack(embeding_test)).to_csv("embedding_test"+str(fold)+".csv")  
            fold+=1
            test_accuracies.append(best_test)
            
            test_auces.append(best_test_auc)
            
            test_acces.append(best_test_acc)
            test_precisiones.append(best_test_precision)
            test_recalles.append(best_test_recall)
            test_f1es.append(best_test_f1)
            
            roc_curves.append((fpr, tpr, best_test_auc))
            all_fpr_test.append(fpr)
            all_tpr_test.append(tpr)
            all_aucs.append(best_test_auc)
            
                
                
            if all_std:
                test_accuracies_complete.append(best_test)
                
                test_auc_complete.append(best_test_auc)
                
                test_acc_complete.append(best_test_acc)
                test_precision_complete.append(best_test_precision)
                test_recall_complete.append(best_test_recall)
                test_f1_complete.append(best_test_f1)
                
                
        # 计算平均的假正例率和真正例率
        min_fpr = min(map(len, all_fpr_test))
        interp_fpr_test = np.linspace(0, 1, min_fpr)
        
        interp_tpr_test = np.zeros((10, min_fpr))
        
        for i in range(10):
            interp_tpr_test[i, :] = np.interp(interp_fpr_test, all_fpr_test[i], all_tpr_test[i])
        
        mean_fpr_test = interp_fpr_test
        mean_tpr_test = interp_tpr_test.mean(axis=0)
        
        # 计算标准差
        tprs_lower_test = np.percentile(interp_tpr_test, 2.5, axis=0)
        tprs_upper_test = np.percentile(interp_tpr_test, 97.5, axis=0)
        
        # 计算平均的AUC和AUC的标准差
        mean_auc = np.mean(all_aucs)
        std_auc = np.std(all_aucs)
        
        auc_plot(roc_curves,mean_fpr_test,mean_tpr_test,tprs_lower_test,tprs_upper_test,mean_auc,std_auc)
        
        plt.savefig('P_split RF_ROC.png', bbox_inches='tight', dpi=600)
        
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))
        
        test_auc_all.append(float(np.array(test_auces).mean()))
        
        test_acc_all.append(float(np.array(test_acces).mean()))
        test_precision_all.append(float(np.array(test_precisiones).mean()))
        test_recall_all.append(float(np.array(test_recalles).mean()))
        test_f1_all.append(float(np.array(test_f1es).mean()))
        

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(), np.array(test_accuracies_complete).std(),
                test_accuracies_all,test_accuracies_complete,test_auc_all,test_auc_complete,
                test_acc_all,test_acc_complete,test_precision_all,test_precision_complete,test_recall_all,test_recall_complete,test_f1_all,test_f1_complete)
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
