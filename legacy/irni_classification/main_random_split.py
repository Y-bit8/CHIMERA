# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from packages import data_info_cal,y_label_cal,stratified_sampling,extract_subsmiles_chiral,extract_subsmiles_metal,create_pyg,pyg_data_generation
from model_evaluation_random_split import gnn_evaluation,Transformer

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from torch_geometric import seed_everything
import torch
import random
seed_everything(432)
# ÉèÖÃÖÖ×Ó 
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

our_data = pd.read_excel('Ir-Ni reaction.xlsx',header=0)
condition = our_data['ddG']==0   
our_data = our_data[~condition].reset_index(drop=True)  
our_data['label']=our_data['ddG'].apply(lambda x: 2 if x>-4 else( 1 if x>-8 else  0 ))
#our_data=our_data[our_data['ddG'] >-12].reset_index(drop=True)  
our_data['bondary']=our_data['tem']

column_index = our_data.columns.get_loc('bondary')


L_Atom = pd.read_csv('Ir-Ni L Atom.csv',header=0)
L_Motif = pd.read_csv('Ir-Ni L Motif.csv',header=0)
P_Atom = pd.read_csv('Ir-Ni P Atom.csv',header=0)
P_Motif = pd.read_csv('Ir-Ni P Motif.csv',header=0)
R1_Atom = pd.read_csv('Ir-Ni R1 Atom.csv',header=0)
R1_Motif = pd.read_csv('Ir-Ni R1 Motif.csv',header=0)
R2_Atom = pd.read_csv('Ir-Ni R2 Atom.csv',header=0)
R2_Motif = pd.read_csv('Ir-Ni R2 Motif.csv',header=0)
our_data=pd.concat([our_data,L_Atom,L_Motif,P_Atom,P_Motif,R1_Atom,R1_Motif,R2_Atom,R2_Motif],axis=1).reset_index(drop=True) 


mol_list=['product']

fps=[]
for i in range(len(our_data)):
    try:
        ms=our_data["ligand"].tolist()
        m = Chem.MolFromSmiles(ms[i],sanitize=False)
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        ms_add=m
        reac_embedding=AllChem.GetMorganFingerprintAsBitVect(ms_add,2,1024)
        for mol in mol_list:
            ms=our_data[mol].tolist()
            m = Chem.MolFromSmiles(ms[i],sanitize=False)
            m.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
            ms_add=m
            reac_embedding+=AllChem.GetMorganFingerprintAsBitVect(ms_add,2,1024)

        refp = np.zeros((0,), dtype=int)
        refp_array=DataStructs.ConvertToNumpyArray(reac_embedding,refp)
        
        fps.append(refp)
    except:
        pass





temp=our_data['tem']
time=our_data['Time']
metal=our_data['metal']
solvent=our_data['solvent']
#pd.DataFrame([len(set(solvent))]).to_csv('solvent_set_check.csv')
additive=our_data['additive']
gm=our_data['gm']
elsi=our_data['elsi']
label=our_data['label']
add_fea=fps
add_fea1=our_data.iloc[:, column_index+1:].values
dataset=pyg_data_generation(our_data,temp,time,metal,solvent,additive,gm,elsi,label,add_fea,add_fea1)
######################################################################################


# data_describe
#s1 = pd.Series([len(our_data)], name='data_length')
#s2 = pd.Series([data_info_cal(our_data)[1]], name='positive_proportion')
##s3 = pd.Series([len(add_test)], name='add_test_data_length')
##s4 = pd.Series([data_info_cal(add_test)[1]], name='add_test_positive_proportion')
#df = pd.concat([s1,s2], axis=1)
#df.to_csv("data_describe.csv")



results = []
# GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
acc, s_1, s_2,test_accuracies_all,test_accuracies_complete,test_auc_all,test_auc_complete,test_acc_all,test_acc_complete,test_precision_all,test_precision_complete,test_recall_all,test_recall_complete,test_f1_all,test_f1_complete = gnn_evaluation(Transformer, dataset, [2], [32], max_num_epochs=60, batch_size=128,
                               start_lr=0.001, num_repetitions=1, all_std=True)

results.append( "Transformer " + str(acc) + " " + str(s_1) + " " + str(s_2))

result_dict={"result":results} 
result_frame=pd.DataFrame(result_dict)
result_frame.to_csv("model_result_random_split.csv")


s1 = pd.Series(test_accuracies_all, name='test_accuracies_all')
s2 = pd.Series(test_accuracies_complete, name='test_accuracies_complete')
s3 = pd.Series(test_auc_all, name='test_auc_all')
s4 = pd.Series(test_auc_complete, name='test_auc_complete')

s5 = pd.Series(test_acc_all, name='test_acc_all')
s6 = pd.Series(test_acc_complete, name='test_acc_complete')

s7 = pd.Series(test_precision_all, name='test_precision_all')
s8 = pd.Series(test_precision_complete, name='test_precision_complete')

s9 = pd.Series(test_recall_all, name='test_recall_all')
s10 = pd.Series(test_recall_complete, name='test_recall_complete')

s11 = pd.Series(test_f1_all, name='test_f1_all')
s12 = pd.Series(test_f1_complete, name='test_f1_complete')

df = pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12], axis=1)
df.to_csv("reult_for_plot_random_split.csv")