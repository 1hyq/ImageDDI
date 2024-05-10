'''
Date: 2023-10-19 18:54:06
LastEditors: xiaomingaaa && tengfei.mm@gmail.com
LastEditTime: 2023-11-16 20:07:37
FilePath: /mol_ddi/utils/motif_utils.py
Description: 
'''
from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import re
import networkx as nx
import dgl
import torch
 
# --------------------- 将分子拆分为不带数字或者自定义 ---------------------   
def fragment_recursive(mol, frags):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))
        if len(bonds) == 0:
            frags.append(Chem.MolToSmiles(mol))
            return frags
 
        idxs, labs = list(zip(*bonds))
        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())
        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]
        broken = Chem.FragmentOnBonds(mol, bondIndices=[bond_idxs[0]], dummyLabels=[(0, 0)])
        head, tail = Chem.GetMolFrags(broken, asMols=True)
        frags.append(Chem.MolToSmiles(head,isomericSmiles=True, kekuleSmiles=True))
        return fragment_recursive(tail, frags)
    except Exception as e:
        return [Chem.MolToSmiles(mol,isomericSmiles=True, kekuleSmiles=True)]

# --------------------- 将*号去掉 ---------------------
def remove_dummy(smiles):
    try:
        stripped_smi=smiles.replace('*','[H]')
        mol=Chem.MolFromSmiles(stripped_smi)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(e)
        return None

def extract_fragment(smiles, pattern = '\[\d+\*\]'):
    try:
        mol_t = Chem.MolFromSmiles(smiles)
        mol_s = copy.deepcopy(mol_t)
        for i in mol_t.GetAtoms():
            i.SetIntProp("atom_idx", i.GetIdx())
        for i in mol_t.GetBonds():
            i.SetIntProp("bond_idx", i.GetIdx())
        ringinfo = mol_t.GetRingInfo()
        bondrings = ringinfo.BondRings()

        if len(bondrings) == 0:
            bondring_list = []
        elif len(bondrings) == 1:
            bondring_list = list(bondrings[0])
        else:
            bondring_list = list(bondrings[0]+bondrings[1])
        # bondring_list

        all_bonds_idx = [bond.GetIdx() for bond in mol_t.GetBonds()]
        none_ring_bonds_list = []
        for i in all_bonds_idx:
            if i not in bondring_list:
                none_ring_bonds_list.append(i)
        # none_ring_bonds_list
        mol_t.GetAtomWithIdx(4).IsInRing()
        cut_bonds = []
        for bond_idx in none_ring_bonds_list:
            bgn_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetBeginAtomIdx()
            ebd_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetEndAtomIdx()
            if mol_t.GetBondWithIdx(bond_idx).GetBondTypeAsDouble() == 1.0:
                if mol_t.GetAtomWithIdx(bgn_atom_idx).IsInRing()+mol_t.GetAtomWithIdx(ebd_atom_idx).IsInRing() == 1:
                    t_bond = mol_t.GetBondWithIdx(bond_idx)
                    t_bond_idx = t_bond.GetIntProp("bond_idx")
                    cut_bonds.append(t_bond_idx)
        # if len(cut_bonds) == 0:
        #     return [smiles]
        mol_frag = Chem.FragmentOnBonds(mol_s, cut_bonds)
        frgs = Chem.GetMolFrags(mol_frag, asMols=True)
        # largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        motifs = []
        for m in frgs:
            m_smiles = Chem.MolToSmiles(m,isomericSmiles=True, kekuleSmiles=True)
            m_smiles = re.sub(pattern, '', m_smiles)
            motifs.append(m_smiles)
        # print(motifs)
        return motifs
    except:
        return [smiles]

def extract_scaffold_motif(dataset="Deng's_dataset"):
    motif_vacb = {}
    
    with open(f'datasets/{dataset}/drug_smiles.csv', 'r') as f:
        f.readline()
        bar = tqdm(f, total=1729)
        for idx, line in enumerate(bar):
            # if idx == 3:
            #     break
            did, smiles = line.strip().split(',')
            mol = Chem.MolFromSmiles(smiles)
            motifs = fragment_recursive(mol, [])
            motifs = [remove_dummy(smi) for smi in motifs]
            motifs.extend(extract_fragment(smiles))
            # print(chem.MolToSmiles(largest_mol))
            # print(motifs)
            for m in motifs:
                if m not in motif_vacb:
                    motif_vacb[m] = len(motif_vacb)
            motif_vacb[smiles] = len(motif_vacb)
            bar.set_description(f'{smiles}, len: {len(motif_vacb)}')
    
    print('vacb size: ', len(motif_vacb))
    data = pd.DataFrame.from_dict(motif_vacb, orient='index', columns=['motif_id'])
    data = data.reset_index().rename(columns={'index':'order_num'})
    data.to_csv('datasets/motif_vacb.csv', sep=',', index=False)

def get_substructure_molecule(smiles, motif_id):
    mol = Chem.MolFromSmiles(smiles)
    motifs = fragment_recursive(mol, [])
    motifs = [remove_dummy(smi) for smi in motifs]
    motifs.extend(extract_fragment(smiles))
    m_ids = []
    for m in motifs:
        # if m in motif_id:
        m_ids.append(motif_id[m])
        # else:
        #     print('not found')
        #     raise ValueError(f"Motif '{m}' not found in motif_id dictionary.")
    return m_ids

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# convert smiles string to graph

def smiles2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # 使用dgl.graph来创建图
    src, dst = zip(*edges)
    g = dgl.graph((src, dst))  # 创建图
    g.ndata['feat'] = torch.from_numpy(np.array(features))  # 添加节点特征
    g = dgl.add_self_loop(g)  # 添加自环

    return g

if __name__ == '__main__':
    # 单个smiles拆分为fragment
    # aspirin = Chem.MolFromSmiles('C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC(=C3)CC4=NNC(=O)C5=CC=CC=C54)F')
    # fragments = fragment_recursive(aspirin, [])
    # clean_fragments = [remove_dummy(smi) for smi in fragments]
    # print(clean_fragments)
    extract_scaffold_motif()