# -*- coding:utf-8 -*-
import os.path as osp
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch
import torch_geometric
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GraphSAGE,GCNConv,GATConv,TransformerConv, global_add_pool,global_mean_pool,global_max_pool,SAGPooling,JumpingKnowledge
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

# One training epoch for GNN model.
def train(train_loader, model, optimizer):
    model.train()

    for data in train_loader:
       # data = data.to(device)
        optimizer.zero_grad()
        output,h,perm_x1,score_x1 = model(data)
        loss = F.nll_loss(output, data[0].y)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        loss_cal=[]
        correct = 0
        for batch in train_loader:
            output,h,perm_x1,score_x1 = model(batch)
            pred = output.max(dim=1)[1]
            loss = F.nll_loss(output, batch[0].y)
            loss_cal.append(loss)
            correct += pred.eq(batch[0].y).sum().item()
    loss_sum=sum(loss_cal)
                         
    return loss_sum/len(loss_cal) ,correct / len(train_loader.dataset)


# Get acc. of GNN model.
def test(loader, model):
    model.eval()
    loss_cal=[]
    correct = 0
    for data in loader:
        #data = data.to(device)
        output,h,perm_x1,score_x1 = model(data)
        pred = output.max(dim=1)[1]
        loss = F.nll_loss(output, data[0].y)
        loss_cal.append(loss)
        correct += pred.eq(data[0].y).sum().item()
    loss_sum=sum(loss_cal)
    return loss_sum/len(loss_cal),correct / len(loader.dataset)

def auc_cal(loader,model):
    y_prediction=[]
    y_true=[]
    y_score=[]
    for data in loader:
        output,h,perm_x1,score_x1 = model(data)
        pred = output.max(dim=1)[1]
        ture= data[0].y
        y_prediction+=pred
        y_true+=ture
        output=torch.exp(output)
        output_list=output[:,1].detach().numpy().tolist()
        y_score+=output_list
    # ¼ÆËã
    y_test=y_true
    y_score=y_score
    fpr, tpr, thread = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc,fpr,tpr

def auc_plot(fpr,tpr):
    # auc»æÍ¼
    lw = 2
    plt.plot(fpr, tpr, lw=lw)
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('10folds CV Test ROC')
    #plt.legend(loc="lower right")


class Transformer(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(Transformer, self).__init__()
        self.conv1 = TransformerConv(dataset[0][0].num_features,hidden,1,edge_dim=dataset[0][0].num_edge_features)
        self.conv2 = TransformerConv(dataset[0][0].num_features,hidden,1,edge_dim=dataset[0][0].num_edge_features)
        self.conv3 = TransformerConv(dataset[0][0].num_features,hidden,1,edge_dim=dataset[0][0].num_edge_features)
        self.conv4 = TransformerConv(dataset[0][0].num_features,hidden,1,edge_dim=dataset[0][0].num_edge_features)
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        self.convs3 = torch.nn.ModuleList()
        self.convs4 = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs1.append(TransformerConv(1*hidden, hidden, 1, edge_dim=dataset[0][0].num_edge_features))
            self.convs2.append(TransformerConv(1*hidden, hidden, 1, edge_dim=dataset[0][0].num_edge_features))
            self.convs3.append(TransformerConv(1*hidden, hidden, 1, edge_dim=dataset[0][0].num_edge_features))
            self.convs4.append(TransformerConv(1*hidden, hidden, 1, edge_dim=dataset[0][0].num_edge_features))

        self.bn3 = BN(14*hidden+20+1) 

        
        self.lin1 = Linear(14*hidden+20+1, 8*hidden)
        self.lin2 = Linear(8*hidden, 4*hidden)
        self.lin3 = Linear(4*hidden, 2)
        self.embedding1 = torch.nn.Embedding(41, min(50, 41//2))

        self.sagpool1= SAGPooling(hidden,0.8,GATConv )
        self.sagpool2 = SAGPooling(hidden,0.8,GATConv )
        self.sagpool3= SAGPooling(hidden,0.8,GATConv )
        self.sagpool4 = SAGPooling(hidden,0.8,GATConv )

    def reset_parameters(self):
            
        torch.nn.init.xavier_uniform_(self.embedding1.weight.data)

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        for conv in self.convs1:
            conv.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        for conv in self.convs3:
            conv.reset_parameters()
        for conv in self.convs4:
            conv.reset_parameters()
            
        self.bn3.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.sagpool1.reset_parameters()
        self.sagpool2.reset_parameters()
        self.sagpool3.reset_parameters()
        self.sagpool4.reset_parameters()
    

        

    def forward(self, data):
        x, edge_index, edge_attr, batch,temp_x,solvent_x  = data[0].x, data[0].edge_index, data[0].edge_attr, data[0].batch, data[0].temp, data[0].solvent
        x1, edge_index_x1, edge_attr_x1, batch_x1 = data[1].x, data[1].edge_index, data[1].edge_attr, data[1].batch
        x2, edge_index_x2, edge_attr_x2, batch_x2 = data[2].x, data[2].edge_index, data[2].edge_attr, data[2].batch
        x3, edge_index_x3, edge_attr_x3, batch_x3 = data[3].x, data[3].edge_index, data[3].edge_attr, data[3].batch
             
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x1 = self.conv2(x1, edge_index_x1, edge_attr=edge_attr_x1)
        x2 = self.conv3(x2, edge_index_x2, edge_attr=edge_attr_x2)
        x3 = self.conv4(x3, edge_index_x3, edge_attr=edge_attr_x3)
        for conv in self.convs1:
            x = conv(x, edge_index, edge_attr)
        for conv in self.convs2:
            x1 = conv(x1, edge_index_x1, edge_attr_x1)
        for conv in self.convs3:
            x2 = conv(x2, edge_index_x2, edge_attr_x2)
        for conv in self.convs4:
            x3 = conv(x3, edge_index_x3, edge_attr_x3)

        x, pool_edge_index, pool_edge_weight, pool_batch_x, perm_x ,score_perm_x= self.sagpool1(x,edge_index,batch=batch)
        x1, pool_edge_index_x1, pool_edge_weight, pool_batch_x1, perm_x1 ,score_perm_x1= self.sagpool2(x1,edge_index_x1,batch=batch_x1)
        x2, pool_edge_index_x2, pool_edge_weight, pool_batch_x2, perm_x2 ,score_perm_x2= self.sagpool3(x2,edge_index_x2,batch=batch_x2)
        x3, pool_edge_index_x3, pool_edge_weight, pool_batch_x3, perm_x3 ,score_perm_x3= self.sagpool4(x3,edge_index_x3,batch=batch_x3)
        
        x = torch.cat((global_max_pool(x, pool_batch_x),global_mean_pool(x, pool_batch_x)),1) 
        x1 = torch.cat((global_max_pool(x1, pool_batch_x1),global_mean_pool(x1, pool_batch_x1)),1) 
        x2 = torch.cat((global_max_pool(x2, pool_batch_x2),global_mean_pool(x2, pool_batch_x2)),1) 
        x3 = torch.cat((global_max_pool(x3, pool_batch_x3),global_mean_pool(x3, pool_batch_x3)),1) 
    
        solvent_embedding = self.embedding1(solvent_x)

        interaction=torch.mul(x,x1)
        interaction1=torch.mul(x,x2)
        interaction2=torch.mul(x,x3)
        z= torch.cat((x, x1,x2,x3,interaction,interaction1,interaction2,solvent_embedding,temp_x.view(len(temp_x),1)), 1)

        z=self.bn3(z)
        z1 = F.relu(self.lin1(z))
        
        z1 = F.dropout(z1, p=0.5, training=self.training)
        z2 = F.relu(self.lin2(z1))
        output = self.lin3(z2)

        return F.log_softmax(output, dim=-1), z2 , perm_x1, score_perm_x1

    def __repr__(self):
        return self.__class__.__name__

# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, dataset,refindices,Refnum, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr = 0.000001, factor=0.5, patience=5,
                       num_repetitions=10, all_std=True):

    # Set device.
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracies_all = []
    test_accuracies_complete = []
    test_auc_all = []
    test_auc_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        test_auc=[]
        plt.figure()
        kf = KFold(n_splits=10, shuffle=True)
        #random.shuffle(dataset)

        for i,(train_idx, test_idx) in enumerate(kf.split(refindices)):
            train_refindices, test_refindices = np.array(refindices)[[train_idx]],np.array(refindices)[[test_idx]]
            train_indeces = [i for i in range(len(Refnum)) if Refnum[i] in train_refindices]
            test_indeces = [i for i in range(len(Refnum)) if Refnum[i] in test_refindices]
            
            # Sample 10% split from training split for validation.
            train_indeces, val_indeces = train_test_split(train_indeces, test_size=0.10)
            best_val_acc = 0.0
            best_val_auc=0.0
            best_test = 0.0
            best_auc = 0.0

            # Split data.
            train_dataset = [dataset[i] for i in train_indeces]
            val_dataset = [dataset[i] for i in val_indeces]
            test_dataset = [dataset[i] for i in test_indeces]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                for h in hidden:
                    # Setup model.
                    model = gnn(train_dataset, l, h)
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=factor, patience=patience,
                                                                           min_lr=0.0000001)
                    for epoch in range(1, max_num_epochs + 1):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        loss_train,acc_train=train(train_loader, model, optimizer)
                        loss_val,val_acc = test(val_loader, model)
                        val_auc,_,_ = auc_cal(val_loader,model)
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            loss_test,best_test = test(test_loader, model)
                        if val_auc > best_val_auc:
                            best_val_auc = val_auc
                            best_test_auc,fpr,tpr = auc_cal(test_loader, model) 

                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr:
                            break

            test_accuracies.append(best_test)
            test_auc.append(best_test_auc)
            auc_plot(fpr,tpr)

            if all_std:
                test_accuracies_complete.append(best_test)
                test_auc_complete.append(best_test_auc)
                
        plt.savefig('ROC curve_test_data_10folds_paper_split.tif')
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))
        test_auc_all.append(float(np.array(test_auc).mean()))

    if all_std:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std(),test_accuracies_all,test_accuracies_complete,test_auc_all,test_auc_complete)
    else:
        return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
