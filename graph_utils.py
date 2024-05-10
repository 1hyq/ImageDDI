'''
Date: 2023-12-29 13:40:03
LastEditors: xiaomingaaa && tengfei.mm@gmail.com
LastEditTime: 2023-12-29 14:12:51
FilePath: /safe_ddi/utils/graph_utils.py
Description: 
'''
from rdkit.Chem import BRICS
from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm

def brics_decomp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges



def extract_scaffold_cliques(dataset="Deng's_dataset"):
    clique_vocab = {}
    
    with open(f'datasets/{dataset}/drug_smiles.csv', 'r') as f:
        f.readline()
        bar = tqdm(f, total=1729)
        for idx, line in enumerate(bar):
            # if idx == 3:
            #     break
            did, smiles = line.strip().split(',')
            
            cliques, edges = brics_decomp(smiles)
            
            for clique in cliques:
                # 将原子团转换为字符串形式，以便比较
                clique_str = ','.join(map(str, clique))
                if clique_str not in clique_vocab:
                    clique_vocab[clique_str] = len(clique_vocab)
                    
    data = pd.DataFrame(list(clique_vocab.items()), columns=['Clique', 'ID'])
    data.to_csv('datasets/clique_vocab.csv', sep=',', index=False)




if __name__ == '__main__':
    smiles = 'CC(=O)C1=C(C)N(Cc2ccco2)C(=O)C1(NC(=O)c1ccc(Cl)cc1Cl)C(F)(F)F'
    brics_decomp(smiles)
    # cliques, edges = brics_decomp(smiles)
    
    # print("Cliques:", cliques)
    # print("Edges:", edges)
    extract_scaffold_cliques()