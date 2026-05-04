# -*- coding:utf-8 -*-
import os.path as osp
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

from CDAN_packages import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from domain_adaptator import Discriminator
from domain_adaptator import ReverseLayerF
from torch_geometric import seed_everything
seed_everything(432)
# 设置种子 
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

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

#def train_simada(cdan_loader,train_loader, model, optimizer,loss_fn):
#    loss_cal=[]
#    R2_cal=[]
#    corr_cal=[]
#    model.train()
#    record=[]
#    for i,(data,data_t) in enumerate(cdan_loader):
#
#        optimizer.zero_grad()
#        output,h,_,_= model(data)
#        pred=output.to(torch.float64)
#        y=data[0].y.view(pred.shape).to(torch.float64)
#        #R2 = torch.sum((pred - torch.mean(y))**2) / torch.sum((y - torch.mean(y))**2)
#        R2 = 1-(torch.sum((y-pred )**2) / torch.sum((y - torch.mean(y))**2))
#        
#        model_loss = torch.sqrt(loss_fn(pred, y))- 0.3 * R2 + 0.3
#        _,h_t,_,_  = model(data_t)
#        f=h.detach()
#        f_t=h_t.detach()
#
#        sim_5_mat=torch.topk(EuclideanDistances(f,f_t), k=10, dim=0, largest=True)[0]
#        da_loss = 100*torch.norm(sim_5_mat)/(torch.max(sim_5_mat)*sim_5_mat.shape[0]*sim_5_mat.shape[1])
#        loss = model_loss+ da_loss
#        #record.append([f.detach().numpy(),f_t.detach().numpy(),da_loss,model_loss])
#        loss.backward()
#        optimizer.step()
#        pred = pred.detach().reshape(-1).numpy()
#        y = y.detach().reshape(-1).numpy()
#        R2 = 1-(np.sum((y-pred )**2) / np.sum((y - np.mean(y))**2))
#        corr = np.corrcoef(y, pred)[0,1]
#        loss = loss.detach().numpy()
#        
#        R2_cal.append(R2)
#        corr_cal.append(corr)
#        loss_cal.append(loss)
#        l = len(loss_cal)
#    #pd.DataFrame(record).to_csv("da_loss_value_check.csv")                     
#    return sum(loss_cal)/l, sum(R2_cal)/l, sum(corr_cal)/l


def train_simada(cdan_loader,train_loader, model, domain_dmm, optimizer,opt_da,loss_fn):
    loss_cal=[]
    R2_cal=[]
    corr_cal=[]
    model.train()
    record=[]
    for i,(data,data_t) in enumerate(cdan_loader):
        v_d, v_p, labels = data[0],data[1], data[0].y
        v_d_t, v_p_t = data_t[0], data_t[1]
        optimizer.zero_grad()
        opt_da.zero_grad()
        output,f,_,_= model(data)
        score=output
        pred=output.to(torch.float64)
        y=data[0].y.view(pred.shape).to(torch.float64)
        model_loss = torch.sqrt(loss_fn(pred, y))
        score_t,f_t,_,_ = model(data_t)
    
        reverse_f = ReverseLayerF.apply(f, 1)
        softmax_output = torch.nn.Softmax(dim=1)(score)
        softmax_output = softmax_output.detach()
        
    
        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
#        check=feature.size()
#        pd.DataFrame(check).to_csv("check.csv")  
        adv_output_src_score = domain_dmm(feature)
    
        reverse_f_t = ReverseLayerF.apply(f_t, 1)
        softmax_output_t = torch.nn.Softmax(dim=1)(score_t)
        softmax_output_t = softmax_output_t.detach()
        
    
        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
    
        adv_output_tgt_score = domain_dmm(feature_t)
               
        n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(128))
        n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(128))
#        h=f.detach()
#        h_t=f_t.detach()
#        sim_5_mat = torch.topk(EuclideanDistances(h,h_t), k=5, dim=0, largest=True)[0]
#        sim_5_loss = 50*torch.norm(sim_5_mat)/(torch.max(sim_5_mat)*sim_5_mat.shape[0]*sim_5_mat.shape[1])                                            
        
#        da_loss = loss_cdan_src + loss_cdan_tgt + 100*sim_5_loss
        da_loss = loss_cdan_src + loss_cdan_tgt 
        loss = model_loss+ da_loss
        
#        record.append([f.detach().numpy() , f_t.detach().numpy() , loss_cdan_src , loss_cdan_tgt , sim_5_loss , da_loss , model_loss,loss])
#        record.append([f.detach().numpy() , f_t.detach().numpy() , loss_cdan_src , loss_cdan_tgt , 100*da_loss , model_loss,loss])
        loss.backward()
        optimizer.step()
        opt_da.step()
    
        pred = pred.detach().reshape(-1).numpy()
        y = y.detach().reshape(-1).numpy()
        R2 = 1-(np.sum((y-pred )**2) / np.sum((y - np.mean(y))**2))
        corr = np.corrcoef(y, pred)[0,1]
        loss = loss.detach().numpy()
        
        R2_cal.append(R2)
        corr_cal.append(corr)
        loss_cal.append(loss)
        l = len(loss_cal)
#    pd.DataFrame(record).to_csv("da_loss_value_check.csv")                     
    return sum(loss_cal)/l, sum(R2_cal)/l, sum(corr_cal)/l


# Get acc. of GNN model.
def test(loader, model,loss_fn):
    model.eval()
    loss_cal=[]
    R2_cal=[]
    corr_cal=[]
    for data in loader:
        #data = data.to(device)
        output,h,perm_x1,score_x1 = model(data)
        pred=output.to(torch.float64)
        y=data[0].y.view(pred.shape).to(torch.float64)
        #R2 = torch.sum((pred - torch.mean(y))**2) / torch.sum((y - torch.mean(y))**2)
        R2 = 1-(torch.sum((y-pred )**2) / torch.sum((y - torch.mean(y))**2))
        loss = torch.sqrt(loss_fn(pred, y))
        
        pred = pred.detach().reshape(-1).numpy()
        y = y.detach().reshape(-1).numpy()
        R2 = 1-(np.sum((y-pred )**2) / np.sum((y - np.mean(y))**2))
        corr = np.corrcoef(y, pred)[0,1]
        loss = loss.detach().numpy()
        
        R2_cal.append(R2)
        corr_cal.append(corr)
        loss_cal.append(loss)
        
    l = len(loss_cal)
    y_all = []
    y_pred_all = []
    for step, data in enumerate(loader):
        output,h,perm_x1,score_x1 = model(data)
        pred=output.to(torch.float64)
        y = data[0].y.view(pred.shape).to(torch.float64)
        pred = list(pred.detach().reshape(-1).numpy())
        y = list(y.detach().reshape(-1).numpy())
        y_all = y_all + y
        y_pred_all = y_pred_all + pred
    
    return sum(loss_cal)/l, sum(R2_cal)/l, sum(corr_cal)/l,y_pred_all,y_all



def reg_plot_train_test(pred,y,pred1,y1,num,fold):
    plt.figure(num=num,figsize=(8, 6))
    plt.plot([min(y1), max(y1)], [min(y1), max(y1)], linestyle='--', color='black', linewidth=2)
    plt.scatter(y, pred, c='b',alpha=0.5,label='test data')
    plt.scatter(y1, pred1,c='red',alpha=0.5,label='train data')
    
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(" Predicted vs True Values", fontsize=16, color='red', pad=15)
    plt.legend()
    plt.savefig('reg_compare_train_test_'+'fold-'+str(fold)+'-'+str(num)+'.png',bbox_inches='tight',dpi=50) #����ͼƬ 600

def reg_plot_test_cv(pred,y):
    plt.figure(50,figsize=(8, 6))
    plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='black', linewidth=2)
    plt.scatter(y, pred,c='b',alpha=0.5)    
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(" Predicted vs True Values", fontsize=16, color='red', pad=15)
    plt.legend()
    plt.savefig('reg_compare_test_cv'+'.png',bbox_inches='tight',dpi=50) #����ͼƬ 600
    

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
        chirality_features = chirality_feas.view(list(chirality_feas.shape)[0]//60,60)  # (batch_size, num_atoms)
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
        
            A = A * (chirality_weight * 2 + (1 - chirality_weight))  # 增加手性特征的权重（根据需要调整此系数）/ # 使用learnable factor
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


class interact_Attention(nn.Module):
    def __init__(self, num_patterns):
        super(interact_Attention, self).__init__()

        # 初始化可学习的注意力权重矩阵
        # 形状是 (batch_size, output_dim, output_dim)
        self.learnable_attention = nn.Parameter(torch.empty(128, num_patterns, num_patterns))
        nn.init.xavier_uniform_(self.learnable_attention)
        
    def forward(self, a, b):
        # 计算 a^T * b，得到 (batch_size, output_dim, output_dim) 的矩阵
        attention_matrix = torch.bmm(a, b.transpose(1, 2))  # a的形状是 (batch_size, output_dim, input_dim)

        # 按位相乘：将可学习的注意力矩阵与 attention_matrix 进行按位乘法
        weighted_attention = self.learnable_attention * attention_matrix

        # Flatten 处理：展平矩阵，使其成为一个一维张量
        flattened_result = weighted_attention.flatten(1)

        return flattened_result


class Transformer(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, num_patterns=10):
        super(Transformer, self).__init__()

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
        self.bn1 = BN(2048+328+16*hidden+6*num_patterns*num_patterns+7)
#        self.bn2 = BN(6*hidden+1*num_patterns*num_patterns+76+3+1024)
        #self.bn3 = BN(16*hidden)
        
        
        self.lin1 = Linear(2048+328+16*hidden+6*num_patterns*num_patterns+7, (2048+328)//2+16*hidden//2+6*num_patterns*num_patterns//2+7)
        self.lin2 = Linear((2048+328)//2+16*hidden//2+6*num_patterns*num_patterns//2+7, (2048+328)//4+16*hidden//4+6*num_patterns*num_patterns//4+7)
        self.lin3 = Linear((2048+328)//4+16*hidden//4+6*num_patterns*num_patterns//4+7, 1)
        
#        self.lin4 = Linear(2048, 2048//2)
#        self.lin5 = Linear(2048//2, 2048//4)
#        self.lin6 = Linear(2048//4, 1)
#        self.lin7 = Linear(2048,2048)
        
        self.lin8 = Linear(2048+328+16*hidden+6*num_patterns*num_patterns+7, 2048+328+16*hidden+6*num_patterns*num_patterns+7)
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
#        self.interact_attn1 = interact_Attention(num_patterns)
#        self.interact_attn2 = interact_Attention(num_patterns)

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
        node_rep_x0, motif_node_rep_x0, super_node_rep_x0 = group_node_rep(xh1,  128, num_part_xh1)
        node_rep_x10, motif_node_rep_x10, super_node_rep_x10 = group_node_rep(xh2,  128, num_part_xh2)
        node_rep_x20, motif_node_rep_x20, super_node_rep_x20 = group_node_rep(xh3,  128, num_part_xh3)
        node_rep_x30, motif_node_rep_x30, super_node_rep_x30 = group_node_rep(xh4,  128, num_part_xh4)
        
        
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
        
#        sim_cp = self.interact_attn1(pool_x,pool_x1)
        
#        sim_R12 = pool_x2 @ pool_x3.transpose(-1, -2)
#        sim_R12 = sim_R12.flatten(1)
#        
        sim_c1 = pool_x @ pool_x2.transpose(-1, -2)
        sim_c1 = sim_c1.flatten(1)
        
#        sim_c1 = self.interact_attn1(pool_x,pool_x2)
        
        sim_c2 = pool_x @ pool_x3.transpose(-1, -2)
        sim_c2 = sim_c2.flatten(1)
#        
#        sim_c2 = self.interact_attn1(pool_x,pool_x3)
        
        sim_x12h = pool_x1h @ pool_x2h.transpose(-1, -2)
        sim_x12h = sim_x12h.flatten(1)
#        sim_x12h = self.interact_attn2(pool_x1h,pool_x2h)
        
        
#        sim_R12 = pool_x2 @ pool_x3.transpose(-1, -2)
#        sim_R12 = sim_R12.flatten(1)
#        
        sim_x13h = pool_x1h @ pool_x3h.transpose(-1, -2)
        sim_x13h = sim_x13h.flatten(1)
#        sim_x13h = self.interact_attn2(pool_x1h,pool_x3h)
        sim_x14h = pool_x1h @ pool_x4h.transpose(-1, -2)
        sim_x14h = sim_x14h.flatten(1)
#        sim_x14h = self.interact_attn2(pool_x1h,pool_x4h)
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
        loffi_input=add1.view(list(add1.shape)[0]//328,328)
#        pd.DataFrame([interaction.shape]).to_csv('size_interaction_check.csv')
#        pd.DataFrame([add.shape]).to_csv('size_add_check.csv')
#        pd.DataFrame([fg_input.shape]).to_csv('size_fg_input_check.csv')
#        
#        fg_attn= F.log_softmax(self.lin7(fg_input), dim=1)
#        fg_rep = torch.mul(fg_attn, fg_input) 
#        fg_rep= F.relu(self.lin4(fg_rep))
#        fg_rep= F.relu(self.lin5(fg_rep))
#        fg_rep= F.relu(self.lin6(fg_rep))
        
#        mole_embedding=torch.cat(( super_rep_x0, super_rep_x10,super_rep_x20, super_rep_x30,sim_cp,sim_c1,sim_c2,x,x1,x2,x3,xs,x1s,x2s,x3s,interaction), 1)
        mole_embedding=torch.cat((loffi_input,fg_input, super_rep_x0, super_rep_x10,super_rep_x20, super_rep_x30,sim_cp,sim_c1,sim_c2,sim_x12h,sim_x13h,sim_x14h,x,x1,x2,x3), 1)
#        mole_embedding=torch.cat((super_rep_x0, super_rep_x10,super_rep_x20, super_rep_x30,sim_cp,sim_c1,sim_c2,sim_x12h,sim_x13h,sim_x14h,x,x1,x2,x3), 1)
        cond_embedding=torch.cat((temp_x,time_x,metal_x,solvent_x,additive_x,gm_x,elsi_x),1)
        z= torch.cat((mole_embedding,cond_embedding), 1)
#        pd.DataFrame([z.shape]).to_csv('size_z_check.csv')
        z_attn= F.log_softmax(self.lin8(z), dim=1)
        z = torch.mul(z_attn, z) 
        z1=self.bn1(z)
        
#        pd.DataFrame([z1.shape]).to_csv('size_z1_check.csv')
        z1 = F.relu(self.lin1(z1))
        
#        z1 = F.dropout(z1, p=0.5, training=self.training)
        z2 = F.relu(self.lin2(z1))
#        z2 = torch.cat((z2,fg_input,loffi_input),1)
        output = self.lin3(z2)
#        pd.DataFrame([output_cf.shape,z.shape]).to_csv('size_check.csv')
#        pd.DataFrame([output_cf[0].detach().numpy()]).to_csv('size_check.csv')



#        return output_rg, F.log_softmax(output_cf, dim=-1) , perm_x1, score_perm_x1
        return output, z , z2, z2

    def __repr__(self):
        return self.__class__.__name__

# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, dataset, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr = 0.000001, factor=0.1, patience=10,
                       num_repetitions=10, all_std=True):



    test_R2es_all = []
    test_R2es_complete = []
    test_losses_all = []
    test_losses_complete = []
    #record=[]
#    seed_list=[6,54,432]
    seed_list=[432]
    for i in range(num_repetitions):
        seed_everything(seed_list[i])
        # Test acc. over all folds.
        test_R2es=[]
        test_losses=[]
        kf = KFold(n_splits=10, shuffle=True,random_state=5)
#        random.shuffle(dataset)
        num=1
        fold=1
        embeding=[]
        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
#            train_index, val_index = train_test_split(train_index, test_size=0.2,random_state=1)
            best_val_R2 = -100.0
            best_test_R2 = -100.0
            best_test_loss = 100.0
            best_R20 = 0.0
            best_R21 = 0.0
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
            cdan_loader = DataLoader(data_CDAN, batch_size=batch_size, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=factor, patience=patience,min_lr=0.0000001)
                    domain_dmm = Discriminator(input_size=2048+328+16*h+6*10*10+7,output_size=2048+328+16*h+6*10*10+7, n_class=2)
                    domain_dmm.reset_parameters()
                    opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=0.001)
#                    scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch: (0.01 if epoch < 100 else 0.001),last_epoch=-1)
                    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_da, mode='min',factor=factor, patience=patience,min_lr=0.0000001)
                    for epoch in range(1, max_num_epochs + 1):
                        lr = scheduler.optimizer.param_groups[0]['lr']
#                        loss_train,R2_train,corr_train=train(train_loader, model, optimizer,loss_fn)
                        #loss_train,R2_train,corr_train=train_simada(cdan_loader,train_loader, model, optimizer,loss_fn)
                        loss_train,R2_train,corr_train=train_simada(cdan_loader,train_loader, model, domain_dmm, optimizer,opt_da,loss_fn)
#                        loss_val,val_R2,val_corr,_,_ = test(val_loader, model,loss_fn)
                        loss_test,test_R2,test_corr,_,_ = test(test_loader, model,loss_fn)
                        scheduler.step(loss_test)
                        scheduler2.step(loss_test)
                        #record.append([val_R2])
                        #pd.DataFrame(record).to_csv("val_R2_check.csv")
#                        if val_R2 > best_val_R2 :
                        if loss_test < best_test_loss :
#                            best_val_R2 = val_R2
#                            loss_test,best_R2,corr,pred,y = test(test_loader, model,loss_fn)
#                            loss_test1,best_R21,corr1,pred1,y1 = test(train_loader, model,loss_fn)
                            
#                            best_test_R2 = test_R2
                            best_test_loss = loss_test
                            loss_test0,best_R20,corr,pred,y = test(test_loader, model,loss_fn)
                            loss_test1,best_R21,corr1,pred1,y1 = test(train_loader, model,loss_fn)
                            
                            for data in train_loader1:
                                _,_,h,_=model(data)
#                                pd.DataFrame(h.shape).to_csv("size_h.csv")
                                embeding_train.append(h.detach().numpy())
#                            pd.DataFrame([len(embeding_train)]).to_csv("len_embeding_train.csv") 
#                            pd.DataFrame(np.vstack(embeding_train)).to_csv("embeding_train_value.csv")    
                            
                            for data in test_loader:
                                _,_,h1,_=model(data)
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
            test_R2es.append(best_R20)
            test_losses.append(best_test_loss)
            try:
                
                reg_plot_train_test(pred,y,pred1,y1,num,i)
                
                reg_plot_test_cv(pred,y)
                num+=1
            except:
                num+=1
                
                
            if all_std:
                test_R2es_complete.append([best_R21,best_R20])
                test_losses_complete.append([loss_test1,loss_test0])

        
        test_R2es_all.append(float(np.array(test_R2es).mean()))
        test_losses_all.append(float(np.array(test_losses).mean()))

    if all_std:
        return (np.array(test_R2es_all).mean(), np.array(test_R2es_all).std(),
                np.array(test_R2es_complete).std(),test_R2es_all,test_R2es_complete,test_losses_all,test_losses_complete)
    else:
        return (np.array(test_R2es_all).mean(), np.array(test_R2es_all).std())
