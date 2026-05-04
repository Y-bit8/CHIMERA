# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
# import packages
# general tools

# RDkit
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from .data_utils import MolGraph

# Pytorch and Pytorch Geometric
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader

#from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


# basic func define

# 计算data长度和标签比例
def data_info_cal(df_data):
    y_label=[]
    y=df_data["ddG"]
    for i in range(len(y)):
        if y[i]>90:
            y_label.append(1)

        else:
            y_label.append(0)
    result=[len(y_label),sum(y_label)/len(y_label)]
    return result

# 计算data标签
def y_label_cal(df_data):
    """Return the continuous regression target.

    The original script accidentally routed a thresholded auxiliary label into
    Data.y.  For regression, Data.y must be the continuous ddG/target_value.
    """
    y = df_data["target_value"] if "target_value" in df_data.columns else df_data["ddG"]
    y_values = list(y.astype(float))
    return y_values, y

# 分层抽样
def stratified_sampling(df_data, stratify, proportion =0.5):
    
    vc = df_data[stratify].value_counts()
    sam = pd.DataFrame(columns = df_data.columns.tolist())
    
    for vi in vc.index:
    
        dd = df_data[df_data[stratify] == vi ].sample(n = round(vc[vi] * proportion))
        sam = pd.concat([sam, dd ], ignore_index = True)
    return sam


def compute_chirality_features(molecule):
    chirality_features = []
    for atom in molecule.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            chirality_features.append(1)  # 手性中心标记为1
        else:
            chirality_features.append(0)  # 非手性中心标记为0
    return chirality_features


def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms
    
#     permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Ru','Unknown']
#     permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','Al','I', 'B', 'Bi', 'Tl','Yb','Pd','Co','Zn', 'Li','Cu','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Ru','Unknown']
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','Al','I', 'B', 'Bi', 'Ir','Yb','Pd','Co','Zn', 'Li','Cu','Ni','Cd','In','Mn','Zr','Cr','Pt','Rh','Pb','Ru','Unknown'] 
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
#     atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
#     vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
#     covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

    atom_feature_vector = atom_type_enc  + n_heavy_neighbors_enc +formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc 
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
#     if hydrogens_implicit == True:
#         n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
#         atom_feature_vector += n_hydrogens_enc
        
####加强手性中心的学习
    # if chirality_type_enc.index(1) in [1,2,3]:
    #     atom_feature_vector=[data*1 for data in atom_feature_vector[:70]]+atom_feature_vector[70:]
    # else:
    #     atom_feature_vector=[data*1 for data in atom_feature_vector[:70]]+atom_feature_vector[70:]

#    if chirality_type_enc.index(1) in [1,2,3]:
#        atom_feature_vector=[data*0.8 for data in atom_feature_vector]
#    else:
#        atom_feature_vector=[data*0.2 for data in atom_feature_vector]
####        
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)
def create_pyg(x_smiles, y,temp,time,metal,solvent,additive,gm,elsi,label,add_fea,add_fea1):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    # choose 200 molecular descriptors
#     chosen_descriptors = []
#     # create molecular descriptor calculator
#     mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)
    data_list = []

#     idx=0
#     del_error=[]
    for (smiles, y_val,temp_val,time_val,metal_val,solvent_val,additive_val,gm_val,elsi_val,label_val,add_fea_val,add_fea1_val) in zip(x_smiles, y,temp,time,metal,solvent,additive,gm,elsi,label,add_fea,add_fea1):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles,sanitize=False)
#         mol = Chem.AddHs(mol)
    
#         road=r'D:/分子表征学习/AHO-Dataset(1)/AHO-Dataset/geoms/'
#         if type_smiles=='cat':
#             road=road+cat_files[idx]+'.xyz'
#         elif type_smiles=="pr":
#             road=road+pr_files[idx]+'.xyz'

#         xyz_data = np.loadtxt(road,usecols=(1,2,3), skiprows=2, dtype=float)
        
        chirality_features = compute_chirality_features(mol)

        
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
            
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
     
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
#        for atom in mol.GetAtoms():
#            X[atom.GetIdx(), :] = np.concatenate([get_atom_features(atom),add_fea_val],axis=0)
            
#             try:
#                 X[atom.GetIdx(), :] =np.concatenate([get_atom_features(atom),xyz_data[atom.GetIdx()]],axis=0)
#             except:
#                 del_error.append(idx)
#                 print(idx)
#         idx+=1    
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        
        # use molecular descriptor calculator on RDKit mol object
#         list_of_descriptor_vals = list(mol_descriptor_calculator.CalcDescriptors(mol))
        
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        temp_tensor = torch.tensor(np.array([temp_val]), dtype = torch.long)
        time_tensor = torch.tensor(np.array([time_val]), dtype = torch.long)
        metal_tensor= torch.tensor(np.array([metal_val]), dtype = torch.long)
        solvent_tensor = torch.tensor(np.array([solvent_val]), dtype = torch.long)
        additive_tensor = torch.tensor(np.array([additive_val]), dtype = torch.long)
        gm_tensor = torch.tensor(np.array([gm_val]), dtype = torch.long)
        elsi_tensor = torch.tensor(np.array([elsi_val]), dtype = torch.long)
        label_tensor = torch.tensor(np.array([label_val]), dtype = torch.long)
        add_fea_tensor = torch.tensor(add_fea_val, dtype = torch.float)
#        pd.DataFrame([add_fea_tensor.shape]).to_csv('size_add_fea_tensor.csv')
        
        add_fea1_tensor = torch.tensor(add_fea1_val, dtype = torch.float)
        # high-level feature representations for molecular
#         hlr_tensor =torch.tensor(np.array(list_of_descriptor_vals), dtype = torch.float)
        # 将手性特征作为一个额外的特征
        chirality_tensor = torch.tensor(chirality_features, dtype=torch.float32)
        # 计算填充的大小
        padding_size = 60 - chirality_tensor.size(0)
        
        # 使用 padding 在张量末尾补充零
        chirality_tensor = F.pad(chirality_tensor, (0, padding_size), value=0)
        pd.DataFrame([chirality_tensor.shape]).to_csv('size_chirality_tensor.csv')
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor,temp=temp_tensor,time=time_tensor,metal=metal_tensor,solvent=solvent_tensor,additive=additive_tensor,gm=gm_tensor,elsi=elsi_tensor, label=label_tensor, add_fea=add_fea_tensor, add_fea1=add_fea1_tensor,chirality_fea=chirality_tensor))
    

    return data_list

def create_pyg_himol(x_smiles):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    data_list = []


    for smiles in x_smiles:
        
        mol_graph = MolGraph(smiles)
        
        
        # construct label tensor
#        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
#        
#        temp_tensor = torch.tensor(np.array([temp_val]), dtype = torch.long)
#        time_tensor = torch.tensor(np.array([time_val]), dtype = torch.long)
#        
#        
#        metal_tensor= torch.tensor(np.array([metal_val]), dtype = torch.long)
#        solvent_tensor = torch.tensor(np.array([solvent_val]), dtype = torch.long)
#        additive_tensor = torch.tensor(np.array([additive_val]), dtype = torch.long)
#        gm_tensor = torch.tensor(np.array([gm_val]), dtype = torch.long)
#        elsi_tensor = torch.tensor(np.array([elsi_val]), dtype = torch.long)
        
        
        # high-level feature representations for molecular
#         hlr_tensor =torch.tensor(np.array(list_of_descriptor_vals), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = mol_graph.x, edge_index = mol_graph.edge_index, edge_attr = mol_graph.edge_attr, num_part=mol_graph.num_part))
    
    return data_list
    
def extract_subsmiles_chiral(smiles,radius=3):
    if not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    chiral_center = None
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED:
            chiral_center = atom.GetIdx()
            break

    if chiral_center is not None:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, chiral_center)
        atoms_to_use = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, chiral_center)
        submol = Chem.PathToSubmol(mol, atoms_to_use)
        submol_smiles = Chem.MolToSmiles(submol)
        return submol_smiles
    else:
        return None
    
    
    
def extract_subsmiles_metal(smiles):
    metal_atoms = ["Cu", "Fe", "Ni", "Co", "Pd", "Mg", "Ca", "Ru", "Ce"
                  , "In", "Ir", "Cr", "La", "Li", "Mn", "Nd", "Os", "Re"
                  , "Rh", "Sc", "Yb", "Zn"]  # 请根据需要添加或删除金属原子符号
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    metal_atom_index = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in metal_atoms:
            metal_atom_index = atom.GetIdx()
            break

    if metal_atom_index is None:
        return None

    radius = 3
    substructure_atoms = set()
    substructure_atoms.add(metal_atom_index)
    for _ in range(radius):
        neighbors = set()
        for atom_index in substructure_atoms:
            atom = mol.GetAtomWithIdx(atom_index)
            neighbors.update([neighbor.GetIdx() for neighbor in atom.GetNeighbors()])
        substructure_atoms.update(neighbors)

    submol_smiles = Chem.MolFragmentToSmiles(mol, list(substructure_atoms), canonical=False, allBondsExplicit=True, allHsExplicit=True)
    return submol_smiles


def pyg_data_generation(df_data,temp,time,metal,solvent,additive,gm,elsi,label,add_fea,add_fea1):
    data_list1=[]
    data_list2=[]
    data_list3=[]
    data_list4=[]
    data_listh1=[]
    data_listh2=[]
    data_listh3=[]
    data_listh4=[]
    y_label,y = y_label_cal(df_data)

    data_list1=create_pyg(df_data['ligand'],y,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1)
    data_list2=create_pyg(df_data['product'],y,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1)
    data_list3=create_pyg(df_data['R1'],y,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1)
    data_list4=create_pyg(df_data['R2'],y,temp,time,metal,solvent,additive,gm,elsi,label, add_fea, add_fea1)
    data_listh1=create_pyg_himol(df_data['ligand'])
    data_listh2=create_pyg_himol(df_data['product'])
    data_listh3=create_pyg_himol(df_data['R1'])
    data_listh4=create_pyg_himol(df_data['R2'])
    
      
    return list(zip(data_list1,data_list2,data_list3,data_list4,data_listh1,data_listh2,data_listh3,data_listh4))
    
    
    
# 查看训练的和测试数据有多相似？以此证明模型确实学到一定的泛化能力

def sim_train_test(mol,train_data,test_data,fold):
    ms=train_data[mol].tolist()
    ms1=test_data[mol].tolist()
    ms_add=[]
    ms1_add=[]
    for i in range(len(ms)):
        m = Chem.MolFromSmiles(ms[i],sanitize=False)
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        ms_add.append(m)
    for i in range(len(ms1)):
        m = Chem.MolFromSmiles(ms1[i],sanitize=False)
        m.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(m,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        ms1_add.append(m)
        
    ms=ms_add
    fps = [AllChem.GetMorganFingerprintAsBitVect(x,4,2048) for x in ms]
    ms1=ms1_add
    fps1 = [AllChem.GetMorganFingerprintAsBitVect(x,4,2048) for x in ms1]

    sim_matrix = [DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps1]
    sim_matrix_array=np.array(sim_matrix)

    coef_sum_cal=[]
    sim_percent=[]
    sim_correct_percent=[]
    sim_wrong_percent=[]
    ind_sim=[]
    for bond in [0.1*data for data in list(range(5,10))]:
        ind_sim_subset=[]
        coef_count_cal=0
        for i in range(len(sim_matrix_array)):
            data=sim_matrix_array[i]
            count=np.sum(data>=bond)
            coef_sum_cal.append(count/len(data))
            if count>=1:
                coef_count_cal+=1
                ind_sim_subset.append(i)

        ind_sim.append(ind_sim_subset)
        percent=coef_count_cal/len(test_data)

        sim_percent.append(percent)


    plt.plot([0.1*data for data in list(range(5,10))],sim_percent,label="test_sim_percent_"+"fold"+str(fold))  

    plt.xlabel("bondary")
    plt.ylabel("sim_percent")
    plt.legend(loc = 1,prop={'size':8})
    

    


