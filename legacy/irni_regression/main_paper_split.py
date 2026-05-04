# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from packages import data_info_cal,y_label_cal,stratified_sampling,create_pyg,pyg_data_generation
from model_evaluation_paper_split import gnn_evaluation,Transformer
our_data = pd.read_excel('Rea box1.xlsx',header=0)
our_data=our_data.dropna(subset=["Catalyst_Smiles",'R1','R2',"product",'Yield (optical)','Solvent (Reaction Details)','Temperature [C]']).reset_index(drop=True)
our_data=our_data.drop_duplicates(subset=["Catalyst_Smiles",'R1','R2',"product",'Yield (optical)','Solvent (Reaction Details)','Temperature [C]'], keep="first", inplace= False)
our_data = our_data.reset_index(drop=True)


one_hot_frame=pd.get_dummies(our_data,columns=['Solvent (Reaction Details)'])
one_hot_frame=one_hot_frame.iloc[:,one_hot_frame.columns.get_loc('last')+1:]
one_hot_solvent=pd.concat([our_data,one_hot_frame],axis=1)
one_hot_solvent_cal=one_hot_frame
df=pd.DataFrame({      
    'solvent': [0]*len(one_hot_solvent_cal)  
})
for i in range(len(one_hot_solvent_cal)):
    
    c = np.where(one_hot_solvent_cal.iloc[i,:] == 1)
    b=c[0].tolist()
    df["solvent"][i]=b[0]
    
    
one_hot_solvent["solvent_index"]=[0]*len(one_hot_solvent)
for i in range(len(one_hot_solvent)):
    c = np.where(one_hot_solvent_cal.iloc[i,:] == 1)
    b=c[0].tolist()
    one_hot_solvent["solvent_index"][i]=b[0]
    
solvent=one_hot_solvent["solvent_index"]
temp=our_data['Temperature [C]']



dataset=pyg_data_generation(our_data,temp,solvent)


from sklearn.model_selection import train_test_split
refs = []
ENTRY_NUM=len(our_data)
Refnum = np.zeros(ENTRY_NUM)

for i in range(ENTRY_NUM):
    x = our_data['Links to Reaxys'][i]
    if x not in refs:
        refs.append(x)
    Refnum[i] = refs.index(x)

# # create a list of indices
refindices = list(range(len(refs)))


results = []
# GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
acc, s_1, s_2, test_accuracies_all,test_accuracies_complete = gnn_evaluation(Transformer, dataset,refindices,Refnum, [2,3], [32,64,128], max_num_epochs=40, batch_size=256,start_lr=0.01, num_repetitions=1, all_std=True)

results.append( "Transformer " + str(acc) + " " + str(s_1) + " " + str(s_2))

result_dict={"result":results} 
result_frame=pd.DataFrame(result_dict)
result_frame.to_csv("model_result_paper_split.csv")

s1 = pd.Series(test_accuracies_all, name='test_accuracies_all')
s2 = pd.Series(test_accuracies_complete, name='test_accuracies_complete')
s3 = pd.Series(test_auc_all, name='test_auc_all')
s4 = pd.Series(test_auc_complete, name='test_auc_complete')
df = pd.concat([s1,s2,s3,s4], axis=1)
df.to_csv("reult_for_plot_paper_split.csv")