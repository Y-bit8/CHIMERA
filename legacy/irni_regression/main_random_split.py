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
# 设置种子 
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

## 定义一个函数，用于从每个组中选出标签绝对值最大的行
#def select_max_abs_label(group):
#    return group.loc[group['ee'].abs().idxmax()]
#
## 使用groupby分组，然后应用自定义函数
#our_data = our_data.groupby(['ligand','product'], as_index=False).apply(select_max_abs_label).reset_index(drop=True)




mol_list=['product']

fps=[]
for i in range(len(our_data)):
    try:
        ms=our_data['ligand'].tolist()
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



## 异常值检测
#from sklearn.ensemble import IsolationForest
#
#X=np.array(fps)
#iforest = IsolationForest(n_estimators=300, max_samples='auto',contamination=0.05, max_features=4,bootstrap=False, n_jobs=-1, random_state=1)
#
#pred= iforest.fit_predict(X)
#df=pd.DataFrame()
#df['scores']=iforest.decision_function(X)
#df['anomaly_label']=pred
#
#our_data=our_data[df.anomaly_label==1].reset_index(drop=True) 



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
acc, s_1, s_2,test_R2es_all,test_R2es_complete,test_losses_all,test_losses_complete = gnn_evaluation(Transformer, dataset, [2], [32], max_num_epochs=150, batch_size=128,
                               start_lr=0.001, num_repetitions=1, all_std=True)

results.append( "Transformer " + str(acc) + " " + str(s_1) + " " + str(s_2))

result_dict={"result":results} 
result_frame=pd.DataFrame(result_dict)
result_frame.to_csv("model_result_random_split.csv")


s1 = pd.Series(test_R2es_all, name='test_accuracies_all')
s2 = pd.Series(test_R2es_complete, name='test_accuracies_complete')
s3 = pd.Series(test_losses_all, name='test_losses_all')
s4 = pd.Series(test_losses_complete, name='test_losses_complete')
df = pd.concat([s1,s2,s3,s4], axis=1)
df.to_csv("reult_for_plot_random_split.csv")