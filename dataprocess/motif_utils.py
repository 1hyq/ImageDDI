import re
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem


def fragment_recursive(mol, frags):
    try:
        bonds = list(Chem.BRICS.FindBRICSBonds(mol))
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
        frags.append(Chem.MolToSmiles(head, isomericSmiles=True, kekuleSmiles=True))
        return fragment_recursive(tail, frags)
    except Exception as e:
        return [Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)]


def remove_dummy(smiles):
    try:
        stripped_smi = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(stripped_smi)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(e)
        return None


def extract_fragment(smiles, pattern='\[\d+\*\]'):
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
            bondring_list = list(bondrings[0] + bondrings[1])

        all_bonds_idx = [bond.GetIdx() for bond in mol_t.GetBonds()]
        none_ring_bonds_list = []
        for i in all_bonds_idx:
            if i not in bondring_list:
                none_ring_bonds_list.append(i)

        mol_t.GetAtomWithIdx(4).IsInRing()
        cut_bonds = []
        for bond_idx in none_ring_bonds_list:
            bgn_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetBeginAtomIdx()
            ebd_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetEndAtomIdx()
            if mol_t.GetBondWithIdx(bond_idx).GetBondTypeAsDouble() == 1.0:
                if mol_t.GetAtomWithIdx(bgn_atom_idx).IsInRing() + mol_t.GetAtomWithIdx(ebd_atom_idx).IsInRing() == 1:
                    t_bond = mol_t.GetBondWithIdx(bond_idx)
                    t_bond_idx = t_bond.GetIntProp("bond_idx")
                    cut_bonds.append(t_bond_idx)

        mol_frag = Chem.FragmentOnBonds(mol_s, cut_bonds)
        frgs = Chem.GetMolFrags(mol_frag, asMols=True)

        motifs = []
        for m in frgs:
            m_smiles = Chem.MolToSmiles(m, isomericSmiles=True, kekuleSmiles=True)
            m_smiles = re.sub(pattern, '', m_smiles)
            motifs.append(m_smiles)
        return motifs
    except:
        return [smiles]


def extract_scaffold_motif(datafile):
    motif_vacb = {}
    drug_substructure_dict = {}

    df = pd.read_csv(datafile)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        did, smiles = row['drug_id'], row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        motifs = fragment_recursive(mol, [])
        motifs = [remove_dummy(smi) for smi in motifs]
        motifs.extend(extract_fragment(smiles))

        motif_ids = []
        for m in motifs:
            if m not in motif_vacb:
                motif_vacb[m] = len(motif_vacb)
            motif_ids.append(motif_vacb[m])

        drug_substructure_dict[did] = motif_ids

    # Save motif_vacb to a file
    print('vacb size: ', len(motif_vacb))
    data = pd.DataFrame.from_dict(motif_vacb, orient='index', columns=['motif_id'])
    data = data.reset_index().rename(columns={'index': 'motif'})
    data.to_csv("datasets/motif_vacb.csv", sep=',', index=False)

    return drug_substructure_dict


datafile = "datasets/Deng's_dataset/drug_smiles.csv"
drug_substructure_dict = extract_scaffold_motif(datafile)

np.savez('datasets/drug_substructure_dict.npz', drug_substructure_dict=drug_substructure_dict, allow_pickle=True)
