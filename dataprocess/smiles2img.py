import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.utils import smiles
from tqdm import tqdm

def loadSmilesAndSave(smis, path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png

        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================
    '''
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224,224))
    img.save(path)

def main():
    '''
    demo of raw_file_path:
        index,k_100,k_1000,k_10000,smiles
        1,97,861,4925,CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1
        2,37,524,4175,CC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1
        3,77,636,6543,COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC
        ...
    :return:
    '''
    parser =argparse.ArgumentParser(description='processing Data Gneration for ImageMol')
    parser.add_argument('--dataroot', type=str, default="./datasets/Deng's_dataset/", help='data root')
    parser.add_argument('--dataset', type=str, default="drug_smiles", help='dataset name, e.g. data')
    args = parser.parse_args()

    raw_file_path = os.path.join(args.dataroot,  "{}.csv".format(args.dataset))
    img_save_root = os.path.join(args.dataroot, "drug_image", "ImageDDI")
    csv_save_path = os.path.join(args.dataroot,  "drug_image.csv")
    error_save_path = os.path.join(args.dataroot, args.dataset, "error_smiles.csv")
    if not os.path.exists(img_save_root):
        os.makedirs(img_save_root)

    df = pd.read_csv(raw_file_path)
    drug_id, smiles = df["drug_id"].values, df["smiles"].values

    processed_ac_data = []
    error_smiles = []
    '''
    for i, (drug_id, s) in enumerate(tqdm(zip(drug_id, smiles), total=len(drug_id))):
        filename = "{}.png".format(i)
        img_save_path = os.path.join(img_save_root, filename)
        try:
            loadSmilesAndSave(s, img_save_path)
            processed_ac_data.append([drug_id, filename])
        except:
            pass
    '''
    for i, s in tqdm(zip(drug_id, smiles), total=len(drug_id)):
        filename = "{}.png".format(i)
        img_save_path = os.path.join(img_save_root, filename)
        try:
            loadSmilesAndSave(s, img_save_path)
            processed_ac_data.append([i, filename])
        except:
            print(f'error smiles : {s}')
            pass

    processed_ac_data = np.array(processed_ac_data)
    pd.DataFrame({
        "drug_id": processed_ac_data[:, 0],
        "filename": processed_ac_data[:, 1],
    }).to_csv(csv_save_path, index=False)

    if len(error_smiles) > 0:
        pd.DataFrame({"smiles": error_smiles}).to_csv(error_save_path, index=False)

if __name__ == '__main__':
    main()
